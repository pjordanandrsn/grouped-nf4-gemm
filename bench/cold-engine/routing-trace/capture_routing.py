"""Capture a REAL decode routing sequence from an MoE.

The committed olmoe_profile.jsonl is aggregate mass -- tokens_routed per
(layer, expert) -- which cannot answer a temporal-locality question. A cache
lives or dies on WHEN an expert is re-routed, not how often. This writes the
sequence: for each decode step, for each layer, the expert ids the router
actually picked.

Autoregressive on purpose. Teacher-forced routing is not the routing a served
model sees, because after the first token the sequence conditions on what the
model itself generated.
"""
import argparse
import json
import sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="prose",
                    choices=["prose", "code", "math", "dialogue"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # Every offline result in this campaign rests on ONE captured trace, and
    # "one prompt" is the standing limit on all of them. These four are
    # deliberately unlike each other -- expository prose, source code, formal
    # mathematics, dialogue -- because a conclusion about MoE ROUTING rather
    # than about one generation has to survive the change.
    PROMPTS = {
        "prose": ("The question of how memory works has occupied philosophers "
                  "and scientists for centuries. When we recall an event, we "
                  "do not replay a recording; we reconstruct it, and the "
                  "reconstruction is shaped by everything we have learned "
                  "since. This is why eyewitness testimony is less reliable "
                  "than juries assume. "),
        "code": ("def merge_intervals(intervals):\n"
                 "    intervals.sort(key=lambda x: x[0])\n"
                 "    out = []\n"
                 "    for start, end in intervals:\n"
                 "        if out and start <= out[-1][1]:\n"
                 "            out[-1][1] = max(out[-1][1], end)\n"
                 "        else:\n"
                 "            out.append([start, end])\n"
                 "    return out\n\n"),
        "math": ("Theorem. Let G be a finite group and H a subgroup of G. "
                 "Then the order of H divides the order of G. Proof. The "
                 "left cosets of H partition G, and each coset gH has the "
                 "same cardinality as H, since h -> gh is a bijection. "
                 "Hence |G| = [G:H] * |H|. "),
        "dialogue": ("A: Have you tried restarting it?\n"
                     "B: Twice. Same error.\n"
                     "A: What does the log say right before it fails?\n"
                     "B: Nothing useful, just a timeout after thirty "
                     "seconds.\n"
                     "A: Thirty seconds is the default. Something upstream "
                     "is not answering.\n"),
    }
    if a.prompt not in PROMPTS:
        sys.exit("--prompt must be one of %s" % sorted(PROMPTS))
    text = PROMPTS[a.prompt]
    reps = max(2, -(-a.prompt_tokens // max(1, len(tk(text).input_ids))) + 1)
    ids = tk(text * reps, return_tensors="pt").input_ids[:, :a.prompt_tokens].to("cuda")

    # Hook every router gate. Reading the gate's OWN logits and redoing topk
    # with the config's k is the only way to see the ids without depending on
    # a particular HF version's return shape.
    layers, gates = [], []
    for name, mod in model.named_modules():
        if name.endswith("mlp.gate"):
            layers.append(name)
            gates.append(mod)
    if not gates:
        sys.exit("no `mlp.gate` modules found — wrong architecture?")
    k = getattr(model.config, "num_experts_per_tok", None) or \
        getattr(model.config, "top_k", None)
    if k is None:
        sys.exit("cannot determine num_experts_per_tok from config")
    print(f"{len(gates)} routers, top_k={k}, "
          f"E={getattr(model.config,'num_experts',None)}")

    step_rec = {}

    def _indices(out):
        """The router hands back (scores[T,E], weights[T,k], indices[T,k]).

        Pick the tensor by SHAPE AND DTYPE rather than by position: an
        integer tensor whose last dim is k is the index tensor and nothing
        else is. Reading position 2 would break silently the first time a
        transformers version reorders the tuple, and the failure would be a
        plausible-looking routing trace rather than an exception.
        """
        cands = [t for t in (out if isinstance(out, (tuple, list)) else [out])
                 if isinstance(t, torch.Tensor)
                 and t.dtype in (torch.int32, torch.int64)
                 and t.shape[-1] == k]
        if len(cands) != 1:
            raise RuntimeError(
                f"expected exactly one integer [*, k={k}] tensor in the "
                f"router output, found {len(cands)}: "
                f"{[(tuple(t.shape), str(t.dtype)) for t in (out if isinstance(out,(tuple,list)) else [out]) if isinstance(t, torch.Tensor)]}")
        return cands[0]

    def mk(i):
        def hook(_m, _inp, out):
            idx = _indices(out).reshape(-1, k)
            # decode: one token; take the LAST row so a prefill call that
            # slips through cannot silently contribute 64 tokens of routing
            step_rec.setdefault(i, []).append(sorted(idx[-1].tolist()))
        return hook

    hs = [g.register_forward_hook(mk(i)) for i, g in enumerate(gates)]

    out = []
    with torch.no_grad():
        past, cur = None, ids
        for s in range(a.steps + 1):
            step_rec.clear()
            r = model(cur, past_key_values=past, use_cache=True)
            past = r.past_key_values
            cur = r.logits[:, -1:].argmax(-1)
            if s == 0:
                continue            # the prefill; not a decode step
            out.append({"step": s - 1,
                        "routed": {str(i): step_rec[i][-1]
                                   for i in range(len(gates))}})
    for h in hs:
        h.remove()

    meta = {"model": a.model, "prompt": a.prompt, "steps": len(out),
            "layers": len(gates),
            "top_k": k, "n_experts": getattr(model.config, "num_experts", None),
            "prompt_tokens": int(ids.shape[1]), "decode": True}
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", a.out, meta)


if __name__ == "__main__":
    main()
