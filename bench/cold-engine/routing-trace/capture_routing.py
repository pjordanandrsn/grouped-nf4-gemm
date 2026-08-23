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
import os
import sys
import torch


def env_fingerprint(model_path, model, repo_id=None):
    """What a trace needs in order to be regenerable later.

    A routing trace is only reproducible if the model weights AND the library
    that runs them are pinned. Neither was recorded, and it mattered: the
    committed OLMoE traces agree with a fresh capture of the same model, same
    prompt, same seed on only 18% of layer-steps under transformers 5.15.1 --
    the MODULE path disagrees by exactly as much as the derived one, so it is
    the environment that moved, not the harness
    (bench/cold-engine/RESULTS-topk-frequency.md). Nothing in the repo said
    which transformers those traces came from, so the drift was invisible
    until something happened to compare across environments.

    The weight index is hashed rather than the weights: it names every shard
    and its tensor map, so a changed checkpoint changes it, and it is
    kilobytes rather than gigabytes. `config.json` is hashed too because a
    config-only revision (a changed `num_experts_per_tok`, say) leaves the
    weights untouched and changes the routing completely.
    """
    import hashlib
    import sys as _sys
    import transformers as _tf

    def sha16(*names):
        for n in names:
            fp = os.path.join(model_path, n)
            if os.path.exists(fp):
                h = hashlib.sha256()
                with open(fp, "rb") as f:
                    for blk in iter(lambda: f.read(1 << 20), b""):
                        h.update(blk)
                return "%s:%s" % (n, h.hexdigest()[:16])
        return None

    # WHICH MODEL, not just which environment. Without this a trace records
    # only a local path like /root/models/granite, and the model becomes
    # unidentifiable: four Hub checkpoints share Granite's 32x40x8 geometry,
    # and establishing that the ambiguity happened to be harmless cost a
    # rented box and sixteen captures
    # (bench/cold-engine/RESULTS-trace-reproducibility.md).
    #
    # Two sources, both checkable, and no guessing at cache layouts. `repo_id`
    # is whatever the operator passed --repo-id; `name_or_path` is what
    # transformers recorded, which IS the Hub id when the model was loaded by
    # id and merely echoes the directory when it was loaded from disk. A local
    # path is not an identity, so it is reported under its own key rather than
    # being allowed to sit in a field that reads like one.
    return {"repo_id": repo_id,
            "name_or_path": getattr(model.config, "_name_or_path", None),
            "model_path": model_path,
            "transformers": _tf.__version__,
            "torch": torch.__version__,
            "python": _sys.version.split()[0],
            "cuda": getattr(torch.version, "cuda", None),
            "config": sha16("config.json"),
            "weight_index": sha16("model.safetensors.index.json",
                                  "model.safetensors", "pytorch_model.bin"),
            "architectures": getattr(model.config, "architectures", None)}


def check_rank_invariants(rec, n_layers):
    """The preregistered capture check. Returns an error string, or None.

    Rank order must be a PERMUTATION of the sorted routed set -- same experts,
    different order -- and the near-miss band must not intersect the selected
    set. A capture that reorders anything else is not what it claims to be
    (bench/cold-engine/PREREG-router-rank.md).

    A function rather than a loop inlined in main() so it can be tested
    without a model: main() only runs after a multi-gigabyte download, which
    is the worst place to discover a check is wrong. The inline version also
    bound `a` and clobbered the argparse namespace (Bugbot, gnf4#199).
    """
    for i in range(n_layers):
        key = str(i)
        sel = rec["routed"][key]
        ranked = rec["routed_rank"][key]
        if sorted(ranked) != sorted(sel):
            return ("rank order is not a permutation of the routed set at "
                    "step %s layer %d: %s vs %s"
                    % (rec.get("step"), i, sel, ranked))
        if len(set(ranked)) != len(ranked):
            return ("rank order repeats an expert at step %s layer %d: %s"
                    % (rec.get("step"), i, ranked))
        nm = (rec.get("near_miss") or {}).get(key)
        if nm is not None and set(nm) & set(sel):
            return ("near-miss band overlaps the selected set at step %s "
                    "layer %d: %s vs %s" % (rec.get("step"), i, sel, nm))
    return None


def n_experts_of(model):
    """E, however this architecture spells it. Used to identify the router by
    the shape of its weight rather than by a name list."""
    for nm in ("num_experts", "num_local_experts", "n_routed_experts",
               "moe_num_experts"):
        v = getattr(model.config, nm, None)
        if v is not None:
            return int(v)
    return None


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
    ap.add_argument("--repo-id", default=None,
                    help="the Hub id the weights came from, e.g. "
                         "ibm-granite/granite-3.1-3b-a800m-instruct. Recorded "
                         "verbatim in the trace. Pass it: a local --model path "
                         "does not identify a model, and four Hub checkpoints "
                         "share Granite's geometry.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="override the config's routed-expert count. ONLY "
                         "honoured on the derive path, where routing is "
                         "recomputed as topk(linear(h, W_router), k) from the "
                         "router's own weights and k is therefore a free "
                         "parameter over a fixed logit distribution. A module "
                         "that emits its own indices emits them at ITS k, and "
                         "silently relabelling that would be a fabricated "
                         "trace, so this forces derivation instead.\n"
                         "RECORDING ONLY: this changes what the hook WRITES, "
                         "never what the model COMPUTES. The forward still "
                         "routes at native k, so the hidden states and the "
                         "generated tokens are identical for every k. The "
                         "trace is a counterfactual readout of the router's "
                         "own ranking -- 'which k experts would this router "
                         "have picked' -- along one fixed decode trajectory. "
                         "That is the intended manipulation and it is what "
                         "makes the k values comparable "
                         "(bench/cold-engine/PREREG-topk-frequency.md)."),
    ap.add_argument("--device", default="cuda",
                    help="where to run the capture. 'cpu' is not a fallback "
                         "for a small box -- gpt-oss-20b ships mxfp4 and "
                         "dequantizes to ~40 GB without triton_kernels, so a "
                         "32 GB card spills the experts and torch._grouped_mm "
                         "dies on mixed devices. Routing is exact either way; "
                         "only speed differs.")
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
        a.model, dtype=torch.bfloat16, device_map=a.device)
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
    ids = tk(text * reps, return_tensors="pt").input_ids[:, :a.prompt_tokens].to(a.device)

    k = (getattr(model.config, "num_experts_per_tok", None)
         or getattr(model.config, "top_k", None)
         or getattr(model.config, "moe_top_k", None)
         or getattr(model.config, "num_experts_per_token", None))
    native_k = k
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
    routers = None

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
    # FALLBACK: derive routing from the router's own weights.
    #
    # A fused MoE path may never CALL the router submodule at all. gpt-oss on
    # transformers 5.x with `kernels` installed computes routing inside the
    # mxfp4 kernel and hands `mlp.experts` a triton_kernels RoutingData; the
    # `mlp.router` module exists, is hookable, and never fires, so the
    # output-probe above finds 24 candidates and reads nothing from any of
    # them. Dequantizing to bf16 restores the module call, but that is a
    # different numerical path from the one the arena was baked from -- and on
    # a 32 GB card it does not fit.
    #
    # The router is a plain Linear over the MLP's input, and gpt-oss's own
    # config excludes it from quantization (`modules_to_not_convert` lists
    # `model.layers.*.mlp.router`), so its weights are bf16 either way and
    # top-k over `linear(h, W, b)` is exactly what the kernel does.
    #
    # VERIFIED, not assumed: against the kernel's own `RoutingData.expt_hist`
    # over 8 tokens, the derived assignment reproduces the histogram on
    # 24 layers of 24.
    # `--top-k` FORCES derivation even when the module emits its own indices.
    # A module emits them at ITS k and cannot be re-topk'd, but the router is
    # a Linear over the MLP's input in both cases, so recomputing from its
    # weights gives the same logits and leaves the RECORDED k free.
    #
    # This changes what is written, not what is computed: the forward still
    # routes at native k, so every k shares one decode trajectory and the
    # traces differ only in how deep the router's ranking is read. That is
    # stronger than re-running the model at each k, which would give each k a
    # different token stream and confound the comparison.
    #
    # Correctness is checked, not asserted: a derived capture at NATIVE k must
    # reproduce the committed trace for that model EXACTLY, id for id --
    # possible precisely because the trajectory does not move
    # (PREREG-topk-frequency.md). The same derivation was separately validated
    # on gpt-oss against the mxfp4 kernel's own RoutingData.expt_hist, 24
    # layers of 24.
    forced_derive = a.top_k is not None and bool(gates)
    if forced_derive:
        gates, routers, layers = [], [], []

    if not gates:
        derived = []
        for n, mod in model.named_modules():
            if not n.endswith("mlp"):
                continue
            # The router is whichever child holds an [E, hidden] weight.
            # Named `router` on gpt-oss and `gate` on OLMoE, so match on the
            # SHAPE rather than on a name list that needs editing per model.
            for cn, cm in mod.named_children():
                w = getattr(cm, "weight", None)
                if w is not None and w.dim() == 2 and w.shape[0] == n_experts_of(model):
                    derived.append((n, mod, cm))
                    break
        if derived:
            derived.sort(key=lambda t: int(t[0].split(".")[2]))
            layers = [n + ".router (derived)" for n, _, _ in derived]
            gates = [mod for _, mod, _ in derived]
            routers = [r for _, _, r in derived]
            # Two different situations reach this branch and the capture log
            # is the artifact someone reads later to reconstruct the run, so
            # they must not print the same sentence: the modules DID fire on
            # OLMoE and were discarded on purpose (Bugbot, gnf4#191).
            print("%s; deriving from router weights for %d layers"
                  % ("--top-k given, so the readable router modules were "
                     "discarded" if forced_derive
                     else "router modules never fired", len(derived)))
            if a.top_k is not None and a.top_k != k:
                print("top_k READOUT OVERRIDE %d -> %d (recording only; the "
                      "model still routes at %d, so the decode trajectory is "
                      "unchanged)" % (k, a.top_k, k))
                k = a.top_k

    if not gates:
        sys.exit("no readable router modules found. Probed %d candidates: %s"
                 % (len(cands), [n for n, _ in cands[:6]]))
    # Every architecture spells the expert count differently and reading only
    # `num_experts` recorded null for Granite, which then had to be inferred
    # from the largest id actually routed -- a lower bound, and one that
    # propagates into any k/E normalization downstream. Mixtral and Granite
    # both use `num_local_experts`.
    n_exp = next((int(getattr(model.config, n))
                  for n in ("num_experts", "num_local_experts",
                            "n_routed_experts", "moe_num_experts")
                  if getattr(model.config, n, None) is not None), None)
    print("routers: %s ..." % ", ".join(layers[:2]))
    # The geometry line an operator checks against the preregistration, so it
    # reports the count actually found rather than one spelling of it.
    print(f"{len(gates)} routers, top_k={k}, E={n_exp}, "
          f"per_step={len(gates) * k}, arena={len(gates) * (n_exp or 0)}")

    step_rec = {}

    def _indices(out):
        return routed_ids(out, k)

    def mk(i):
        def hook(_m, _inp, out):
            idx = _indices(out).reshape(-1, k)
            # decode: one token; take the LAST row so a prefill call that
            # slips through cannot silently contribute 64 tokens of routing.
            #
            # `routed` stays SORTED. positional_transfers compares routed sets
            # BY INDEX POSITION, and replay_dev_cache documents the sorted
            # order as a deliberate, conservative choice for that baseline --
            # rewriting it in rank order would silently move the denominator
            # of every ratio this program has published. Rank is additive.
            r = idx[-1].tolist()                 # topk is score-descending
            step_rec.setdefault(i, []).append((sorted(r), r, None))
        return hook

    def mk_derived(i, router):
        def hook(_m, args):
            h = args[0]
            lg = torch.nn.functional.linear(
                h.reshape(-1, h.shape[-1]), router.weight,
                getattr(router, "bias", None))
            # 2k, so the near-miss band -- the experts that just failed to
            # make the cut -- falls out of the same sort at no extra cost.
            wide = torch.topk(lg, k=min(2 * k, lg.shape[-1]), dim=-1).indices
            row = wide[-1].tolist()
            r, near = row[:k], row[k:]
            step_rec.setdefault(i, []).append((sorted(r), r, near))
        return hook

    if routers is not None:
        hs = [g.register_forward_pre_hook(mk_derived(i, routers[i]))
              for i, g in enumerate(gates)]
    else:
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
            # The token this step produced, so a degenerate generation is
            # visible in the trace itself. Qwen's math trace was a period-2
            # repetition loop and that was only found by inspecting routing
            # lag-overlap afterwards; with the ids here it is one diff
            # (bench/cold-engine/routing-trace/RESULTS-third-model.md).
            rec = {"step": s - 1,
                   "token": int(cur.reshape(-1)[-1].item()),
                   "routed": {str(i): step_rec[i][-1][0]
                              for i in range(len(gates))},
                   "routed_rank": {str(i): step_rec[i][-1][1]
                                   for i in range(len(gates))}}
            if step_rec[0][-1][2] is not None:      # derive path only
                rec["near_miss"] = {str(i): step_rec[i][-1][2]
                                    for i in range(len(gates))}
            # The preregistered capture check, asserted at write time rather
            # than left to a test that cannot run without a model: rank order
            # must be a PERMUTATION of the sorted ids -- same experts, different
            # order. A capture that reorders anything else is not what it says
            # it is (bench/cold-engine/PREREG-router-rank.md).
            err = check_rank_invariants(rec, len(gates))
            if err:
                sys.exit(err)
            out.append(rec)
    for h in hs:
        h.remove()

    tok = [r["token"] for r in out]
    meta = {"model": a.model, "prompt": a.prompt, "steps": len(out),
            "layers": len(gates),
            "top_k": k, "n_experts": n_exp,
            "top_k_native": None if a.top_k is None else native_k,
            "top_k_overridden": a.top_k is not None and a.top_k != native_k,
            "distinct_tokens": len(set(tok)),
            "prompt_tokens": int(ids.shape[1]), "decode": True,
            # Provenance. Without this a trace cannot be regenerated and,
            # worse, cannot be KNOWN to have drifted -- see env_fingerprint.
            "env": env_fingerprint(a.model, model, a.repo_id)}
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", a.out, meta)


if __name__ == "__main__":
    main()
