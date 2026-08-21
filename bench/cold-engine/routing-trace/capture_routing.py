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


def routed_ids(out, k):
    """The routed expert ids, however this transformers version spells them.

    Two shapes exist in the wild and both have to work, because the
    module named `mlp.gate` is not the same object across versions:

    * a **router module** (OLMoE's `OlmoeTopKRouter` in current
      transformers) returns `(scores[T,E], weights[T,k], indices[T,k])`
      and hands the ids over directly;
    * a bare **`nn.Linear`** (older transformers) returns float logits
      `[T, E]`, and the ids have to be recomputed with top-k.

    Indices are preferred when present and picked by SHAPE AND DTYPE, not
    by position -- an integer tensor whose last dim is k is the index
    tensor and nothing else is. Reading position 2 would break silently
    the first time the tuple is reordered, and a silent break here is a
    plausible-looking routing trace rather than an exception.
    """
    ts = [t for t in (out if isinstance(out, (tuple, list)) else [out])
          if isinstance(t, torch.Tensor)]
    idx = [t for t in ts if t.dtype in (torch.int32, torch.int64)
           and t.shape[-1] == k]
    if len(idx) == 1:
        return idx[0]
    if len(idx) > 1:
        raise RuntimeError(
            f"ambiguous router output: {len(idx)} integer [*, k={k}] "
            f"tensors, cannot tell which holds the routed ids "
            f"({[tuple(t.shape) for t in idx]})")
    # No index tensor: this is the raw-logits form. Take the widest
    # float tensor -- scores over all E experts -- and redo the top-k.
    lg = [t for t in ts if t.is_floating_point()]
    if not lg:
        raise RuntimeError(
            "router output has neither an integer [*, k] index tensor "
            f"nor a float logit tensor: "
            f"{[(tuple(t.shape), str(t.dtype)) for t in ts]}")
    wide = max(lg, key=lambda t: t.shape[-1])
    if wide.shape[-1] < k:
        raise RuntimeError(
            f"router logits have {wide.shape[-1]} columns, fewer than "
            f"top_k={k}; this is not a router output")
    return torch.topk(wide.reshape(-1, wide.shape[-1]), k=k, dim=-1).indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--router-suffix", default=None,
                    help="restrict router discovery to modules whose name "
                         "ends with this; probing finds them without it")
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

    k = (getattr(model.config, "num_experts_per_tok", None)
         or getattr(model.config, "top_k", None)
         or getattr(model.config, "moe_top_k", None)
         or getattr(model.config, "num_experts_per_token", None))
    if k is None:
        sys.exit("cannot determine the routed-expert count from config; "
                 "looked for num_experts_per_tok, top_k, moe_top_k, "
                 "num_experts_per_token")

    # Router discovery is BY PROBE, not by name. `mlp.gate` is OLMoE's
    # spelling; GraniteMoE says `block_sparse_moe.router.layer`, Mixtral
    # `block_sparse_moe.gate`, and a name list would need editing for every
    # new architecture -- which is how a capture harness quietly becomes
    # single-model. Candidates are anything plausibly a router; the ones kept
    # are those whose output a probe forward can actually read routed ids out
    # of, one per layer.
    cands = [(n, m) for n, m in model.named_modules()
             if n.rsplit(".", 1)[-1] in ("gate", "router", "layer", "gate_proj")
             or n.endswith("router.layer")]
    if a.router_suffix:
        cands = [(n, m) for n, m in cands if n.endswith(a.router_suffix)]
    seen, probe = {}, {}

    def _probe(nm):
        def h(_m, _i, out):
            probe[nm] = out
        return h

    hs = [m.register_forward_hook(_probe(n)) for n, m in cands]
    with torch.no_grad():
        model(ids[:, :2])
    for h in hs:
        h.remove()

    layers, gates = [], []
    for n, m in cands:
        out = probe.get(n)
        if out is None:
            continue
        try:
            routed_ids(out, k)
        except Exception:
            continue
        # One router per layer: keyed on the `model.layers.N` prefix, so a
        # nested `router.layer` does not also register its parent.
        key = ".".join(n.split(".")[:3])
        if key in seen:
            continue
        seen[key] = n
        layers.append(n)
        gates.append(m)
    if not gates:
        sys.exit("no readable router modules found. Probed %d candidates: %s"
                 % (len(cands), [n for n, _ in cands[:6]]))
    print("routers: %s ..." % ", ".join(layers[:2]))
    print(f"{len(gates)} routers, top_k={k}, "
          f"E={getattr(model.config,'num_experts',None)}")

    step_rec = {}

    def _indices(out):
        return routed_ids(out, k)

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
