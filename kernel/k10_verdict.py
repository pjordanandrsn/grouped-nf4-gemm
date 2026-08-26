# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k10-decode-router.

Stage A names the OWNER of the census `router` row and gates
everything; Stage B1 is the sorted=False probe. K9 died twice from
designing a treatment before proving who owned a cost, so an
unidentified owner here REFUSES rather than degrading to a guess.

Report shape:
  {"stage_a": {"topk_calls_per_step": float, "layers": int,
               "router_shape_calls_per_step": float,
               "sort_calls_per_step": int,
               "gather_calls_per_step": int,
               "sort_calls_per_step_ablated": int},
   "b1": {"sets_identical": bool, "layers_checked": int,
          "ppl_base": float, "ppl_sorted_false": float,
          "ppl_tokens_base": int, "ppl_tokens_false": int},
   "step": {"base_a": ARM, "base_b": ARM, "sf_a": ARM, "sf_b": ARM},
   "cert_knob_ms": float}
"""

import argparse
import json
import math
import sys

PPL_EPS = 0.05          # TR2 / K8 epsilon, same question
AA_TOL = 0.02
ANCHOR_TOL = 0.05


def _pos_finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def verdict(rep):
    out = {"stage_a": {}, "gates": {}, "verdict": None}
    a = rep.get("stage_a") or {}

    layers = a.get("layers")
    if not isinstance(layers, int) or layers <= 0:
        return _refuse(out, "A: layer count missing")
    for k in ("topk_calls_per_step", "router_shape_calls_per_step",
              "sort_calls_per_step", "gather_calls_per_step"):
        if not _pos_finite(a.get(k)):
            return _refuse(out, f"A: {k} missing or non-positive")

    # A1: one topk per layer per step
    if abs(a["topk_calls_per_step"] - layers) > 1e-9:
        return _refuse(out, f"A1: {a['topk_calls_per_step']} topk "
                            f"calls/step against {layers} layers -- not "
                            "one per layer, so the router is not the "
                            "sole caller; refusing to treat an "
                            "unidentified cost")
    # A2: those calls are the ROUTER shape, not some other topk
    if abs(a["router_shape_calls_per_step"] - layers) > 1e-9:
        return _refuse(out, f"A2: only {a['router_shape_calls_per_step']}"
                            f" of {layers} calls carry the router shape")
    # A3: the census kernels match the call count
    for k, name in (("sort_calls_per_step", "bitonicSortKVInPlace"),
                    ("gather_calls_per_step", "sbtopk::gatherTopK")):
        if a[k] != layers:
            return _refuse(out, f"A3: {name} fires {a[k]}x/step against "
                                f"{layers} topk calls -- the kernels do "
                                "not resolve to this call site")
    # A4: the ABLATION -- forcing sorted=False must delete the sort
    abl = a.get("sort_calls_per_step_ablated")
    if abl is None:
        return _refuse(out, "A4: no ablation run -- the attribution is "
                            "unproven without it")
    if abl != 0:
        return _refuse(out, f"A4: sorted=False still leaves {abl} "
                            f"{'sort'} kernels/step; the sort is NOT "
                            "this call site's, and any B1 delta would "
                            "be measuring something else")

    out["stage_a"] = {"owner": "torch.topk(sorted=True) in the router",
                      "layers": layers,
                      "sort_calls_per_step": a["sort_calls_per_step"],
                      "gather_calls_per_step": a["gather_calls_per_step"]}

    b = rep.get("b1") or {}
    if not b:
        out["verdict"] = ("STAGE-A-ONLY", "owner identified; no B1 run")
        return out

    # B1-C: the SET must be unchanged (a changed set is a different
    # model, and refuses regardless of perplexity)
    if not isinstance(b.get("layers_checked"), int) or \
            b["layers_checked"] < 1:
        return _refuse(out, "B1-C: no layers checked -- a set-equality "
                            "gate that examined nothing passes vacuously")
    if not b.get("sets_identical"):
        return _refuse(out, "B1-C: selected expert SETS differ between "
                            "arms -- that is a different model, not a "
                            "reordering")
    for k in ("ppl_base", "ppl_sorted_false"):
        if not math.isfinite(b.get(k, float("nan"))):
            return _refuse(out, f"B1-Q: {k} missing or non-finite")
    if b.get("ppl_tokens_base") != b.get("ppl_tokens_false"):
        return _refuse(out, "B1-Q: the arms scored different token "
                            "budgets")

    s = rep.get("step") or {}
    for label, (x, y) in (("base", ("base_a", "base_b")),
                          ("sorted_false", ("sf_a", "sf_b"))):
        for k in (x, y):
            arm = s.get(k)
            if not arm or not _pos_finite(arm.get("step_ms_clean")):
                return _refuse(out, f"A/A: arm {k} missing or non-positive")
        p, q = s[x]["step_ms_clean"], s[y]["step_ms_clean"]
        spread = abs(p - q) / min(p, q)
        out["gates"][f"{label}_aa"] = spread
        if spread > AA_TOL:
            return _refuse(out, f"A/A: {label} spread {spread * 100:.2f}%"
                                f" > {AA_TOL * 100:.0f}%")
        # per-arm determinism: the prereg requires tokens identical as
        # well as spread, and a non-deterministic pair that happens to
        # land inside AA_TOL would otherwise clear the gate and produce
        # a verdict instead of a REFUSE (Bugbot, gnf4#275)
        if s[x].get("tokens") != s[y].get("tokens"):
            return _refuse(out, f"A/A: {label} token streams differ "
                                "between identical runs -- the arm is "
                                "not deterministic, so its timing pair "
                                "is not an A/A")
    base = (s["base_a"]["step_ms_clean"] + s["base_b"]["step_ms_clean"]) / 2
    sf = (s["sf_a"]["step_ms_clean"] + s["sf_b"]["step_ms_clean"]) / 2
    cert = rep.get("cert_knob_ms")
    if not _pos_finite(cert):
        return _refuse(out, "anchor: cert_knob_ms missing or non-finite")
    drift = abs(base - cert) / cert
    out["gates"]["anchor_drift"] = drift
    if drift > ANCHOR_TOL:
        return _refuse(out, f"anchor: base {base:.3f} ms is "
                            f"{drift * 100:.1f}% off {cert:.3f} ms")

    d_ppl = b["ppl_sorted_false"] - b["ppl_base"]
    out["gates"].update({"ppl_delta": d_ppl, "step_base_ms": base,
                         "step_sf_ms": sf, "step_delta_ms": base - sf})
    if d_ppl > PPL_EPS:
        out["verdict"] = ("REFUTED", f"quality: ppl +{d_ppl:.4f} > "
                                     f"{PPL_EPS}")
    else:
        out["verdict"] = ("PASS", base - sf)
    return out


def _refuse(out, why):
    out["verdict"] = ("REFUSE", why)
    return out


def render(out):
    tag, x = out["verdict"]
    if tag == "REFUSE":
        return f"K10 VERDICT: REFUSE\n  {x}"
    a = out["stage_a"]
    lines = [f"K10 STAGE A: owner = {a['owner']} "
             f"({a['sort_calls_per_step']} sort + "
             f"{a['gather_calls_per_step']} gather per step, "
             f"{a['layers']} layers); ablation drove the sort to 0"]
    if tag == "STAGE-A-ONLY":
        lines.append("K10 VERDICT: STAGE-A-ONLY (no B1 arms in report)")
        return "\n".join(lines)
    g = out["gates"]
    lines.append(f"K10 VERDICT: {tag}  (step {g['step_base_ms']:.3f} -> "
                 f"{g['step_sf_ms']:.3f} ms = {g['step_delta_ms']:+.3f} ms; "
                 f"ppl {g['ppl_delta']:+.4f} vs eps {PPL_EPS})")
    return "\n".join(lines)


def _mk(layers=48, topk=48, shape=48, sort=48, gather=48, abl=0,
        b1=True, sets=True, checked=48, ppl_d=0.01, tok=(1024, 1024),
        base=6.46, sf=6.32, aa=0.0, cert=6.476):
    r = {"stage_a": {"topk_calls_per_step": topk, "layers": layers,
                     "router_shape_calls_per_step": shape,
                     "sort_calls_per_step": sort,
                     "gather_calls_per_step": gather,
                     "sort_calls_per_step_ablated": abl},
         "cert_knob_ms": cert}
    if b1:
        toks = list(range(20))
        r["b1"] = {"sets_identical": sets, "layers_checked": checked,
                   "ppl_base": 4.94, "ppl_sorted_false": 4.94 + ppl_d,
                   "ppl_tokens_base": tok[0], "ppl_tokens_false": tok[1]}
        r["step"] = {"base_a": {"step_ms_clean": base, "tokens": toks},
                     "base_b": {"step_ms_clean": base * (1 + aa),
                                "tokens": toks},
                     "sf_a": {"step_ms_clean": sf, "tokens": toks},
                     "sf_b": {"step_ms_clean": sf, "tokens": toks}}
    return r


def self_test():
    assert verdict(_mk())["verdict"][0] == "PASS"
    assert verdict(_mk(b1=False))["verdict"][0] == "STAGE-A-ONLY"
    assert verdict(_mk(ppl_d=0.049))["verdict"][0] == "PASS"
    assert verdict(_mk(ppl_d=0.051))["verdict"][0] == "REFUTED"
    # Stage A refusals -- each is an attribution failure, the K9 lesson
    for bad, why in ((_mk(topk=96), "A1"),
                     (_mk(shape=24), "A2"),
                     (_mk(sort=96), "A3"),
                     (_mk(gather=24), "A3"),
                     (_mk(abl=48), "A4"),
                     (_mk(abl=1), "A4"),
                     (_mk(layers=0), "A:")):
        r = verdict(bad)
        assert r["verdict"][0] == "REFUSE" and \
            r["verdict"][1].startswith(why), (why, r["verdict"])
    r = _mk(); del r["stage_a"]["sort_calls_per_step_ablated"]
    assert verdict(r)["verdict"][1].startswith("A4"), "missing ablation"
    # B1 refusals
    # a non-deterministic pair inside AA_TOL must still REFUSE
    nd = _mk()
    nd["step"]["base_b"]["tokens"] = [99] + nd["step"]["base_b"]["tokens"][1:]
    r = verdict(nd)
    assert r["verdict"][0] == "REFUSE" and "token streams differ" in \
        r["verdict"][1], r["verdict"]
    nd2 = _mk()
    nd2["step"]["sf_b"]["tokens"] = []
    assert verdict(nd2)["verdict"][0] == "REFUSE", "sf pair unchecked"
    for bad, why in ((_mk(sets=False), "B1-C"),
                     (_mk(checked=0), "B1-C"),
                     (_mk(tok=(1024, 512)), "B1-Q"),
                     (_mk(aa=0.03), "A/A"),
                     (_mk(cert=7.39), "anchor"),
                     (_mk(cert=float("nan")), "anchor")):
        r = verdict(bad)
        assert r["verdict"][0] == "REFUSE" and \
            r["verdict"][1].startswith(why), (why, r["verdict"])
    for line in render(verdict(_mk())).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("k10_verdict self-test OK (eight Stage-A attribution "
          "refusals including the ablation, six B1 refusals, the "
          "quality epsilon either side, per-arm token determinism "
          "on both pairs, and the Stage-A-only path)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.report:
        ap.error("report path or --self-test")
    out = verdict(json.load(open(a.report)))
    print(render(out))
    if out["verdict"][0] == "REFUSE":
        sys.exit(3)


if __name__ == "__main__":
    main()
