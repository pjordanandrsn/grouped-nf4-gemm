"""Lane K27 bench (``PREREG-k27-nf4-tree-precision.md``): does K25 with the select tree keep its speed at the served
kernel's weight precision?

experts4bit-qlora lane P92 read K25 (bf16 weight operand) QUALITY_FAIL on OLMoE, its c4val1 K8 moved −0.107 ppl. The
served NF4 GEMM multiplies TF32 on fp32 weights. Lane K26 read the select-tree decode at 0.37-0.38 of K25's paired lookup.
K25's ``dot_bf16=False`` keeps the fp32 weight and runs TF32 MMA (the served precision class). This bench reads, at the
NF4 families' B=16 shapes (K26's harness: per-layer synthetic stores, seeded top-8 routing, both GEMMs per layer in one
CUDA graph):

- ``pair16``  K25 as P92 ran it: paired lookup, bf16 weight operand
- ``tree16``  K25 with the select tree, bf16 (bit-identical to ``pair16``)
- ``tree32``  K25 with the select tree, fp32 weights through TF32 MMA (``dot_bf16=False``)
- ``served``  ``nf4_grouped.gemm_4bit_grouped_captured`` (TF32 on fp32 weights, the route P92's OFF arm ran)

Each tree arm takes the best of a small plan grid, selected on steps 0-3 and read on steps 4-7. The error proxy: every
row of step 0, layer 0, both projections, against an fp64 product of the fp32 dequant; rms error per arm.

    python k27_bench.py out.json [--quick]
    python k27_bench.py --self-test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

FAMILIES = {
    "granite": dict(E=40, k=8, L=32, gu=(1024, 1536), dn=(1536, 512)),
    "olmoe": dict(E=64, k=8, L=16, gu=(2048, 2048), dn=(2048, 1024)),
}
B = 16
STEPS = 8
SELECT, EVAL = (0, 1, 2, 3), (4, 5, 6, 7)
ITERS = 20
DEFAULT = dict(block_n=32, kc=256, warps=4, stages=2)
GRID = [dict(block_n=bn, kc=kc, warps=wp, stages=st) for bn, kc, wp, st in
        [(32, 256, 4, 2), (32, 128, 4, 2), (32, 128, 4, 3), (64, 128, 4, 2), (32, 64, 4, 3), (16, 256, 4, 2)]]
SPEED_BAR = 0.60              # tree arm / served, both families
ERR_BAR = 1.10                # tree32 rms error / served rms error, both families


def verdict(rec):
    fams = rec.get("families") or {}
    if set(fams) != set(FAMILIES):
        return "VOID", f"families measured {sorted(fams)}, registered {sorted(FAMILIES)}"
    for n, f in fams.items():
        if not f.get("tree16_bit_equal_pair16"):
            return "VOID", f"{n}: tree16 is not bit-equal to pair16"
        if not f.get("served_sane"):
            return "VOID", f"{n}: the served kernel's error is not within 2x of pair16's (instrument)"
        missing = [a for a in ("pair16", "tree16", "tree32", "served") if not (f.get("eval_ms") or {}).get(a)]
        if missing:
            return "VOID", f"{n}: arms missing {missing}"
    s32 = {n: f["eval_ms"]["tree32"] / f["eval_ms"]["served"] for n, f in fams.items()}
    s16 = {n: f["eval_ms"]["tree16"] / f["eval_ms"]["served"] for n, f in fams.items()}
    e32 = {n: f["rms_err"]["tree32"] / f["rms_err"]["served"] for n, f in fams.items()}
    txt = (", ".join(f"{n} tree32/served {s32[n]:.3f} err {e32[n]:.3f}" for n in fams)
           + "; " + ", ".join(f"{n} tree16/served {s16[n]:.3f}" for n in fams))
    if all(v <= SPEED_BAR for v in s32.values()) and all(v <= ERR_BAR for v in e32.values()):
        return "TF32_PATH", txt + f" -- next: K25-tree in TF32 end to end on both families (<= {SPEED_BAR}, err <= {ERR_BAR})"
    if all(v <= SPEED_BAR for v in s16.values()):
        return "BF16_ONLY", txt + " -- next: K25-tree in bf16 end to end (the families whose K8 already passed)"
    return "NONE", txt


def self_test():
    def fam(t32=4.0, t16=3.5, served=10.0, e32=1.0, eq=True, sane=True, drop=None):
        ev = {"pair16": 9.0, "tree16": t16, "tree32": t32, "served": served}
        if drop:
            ev.pop(drop)
        return {"tree16_bit_equal_pair16": eq, "served_sane": sane, "eval_ms": ev,
                "rms_err": {"pair16": 1.37, "tree16": 1.37, "tree32": e32, "served": 1.0}}

    def rec(g=None, o=None):
        return {"families": {"granite": fam(**(g or {})), "olmoe": fam(**(o or {}))}}
    assert verdict(rec())[0] == "TF32_PATH"
    assert verdict(rec(g={"t32": 6.0}))[0] == "TF32_PATH"                                      # the boundary
    assert verdict(rec(g={"t32": 6.5}))[0] == "BF16_ONLY"                                      # too slow in one family
    assert verdict(rec(o={"e32": 1.2}))[0] == "BF16_ONLY"                                      # error too large
    assert verdict(rec(g={"t32": 7.0, "t16": 7.0}))[0] == "NONE"
    assert verdict(rec(o={"eq": False}))[0] == "VOID"
    assert verdict(rec(g={"sane": False}))[0] == "VOID"
    assert verdict(rec(g={"drop": "tree32"}))[0] == "VOID"
    assert verdict({"families": {"granite": fam()}})[0] == "VOID"
    print("k27_bench self-test OK (9 cases)")


def main(out_path, quick=False):
    import torch
    from int4_b32 import build_group_tiles_fused
    from nf4_grouped import dequant_ref, gemm_4bit_grouped_captured
    from nf4_smallm import gemm_nf4_grouped_smallm
    dev = "cuda"

    def capture(fn, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        torch.cuda.synchronize()
        return gr

    def graph_ms(fn):
        gr = capture(fn)
        ts = []
        for _ in range(ITERS):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            gr.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        del gr
        torch.cuda.empty_cache()
        return statistics.median(ts)

    t0 = time.time()
    rec = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "B": B, "steps": STEPS, "quick": quick,
           "grid": GRID, "families": {}}
    for fam, c in FAMILIES.items():
        E, k, L = c["E"], c["k"], (2 if quick else c["L"])
        g = torch.Generator(device=dev).manual_seed(27)
        WL = []
        for _li in range(L):
            Wl = {}
            for proj in ("gu", "dn"):
                N, K = c[proj]
                Wl[proj] = (torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                            torch.rand(E, N, K // 64, generator=g, device=dev) * 0.05 + 0.01)
            WL.append(Wl)
        xg = (torch.randn(B, c["gu"][1], generator=g, device=dev) / 4).to(torch.bfloat16).repeat_interleave(k, 0)
        xd = (torch.randn(B * k, c["dn"][1], generator=g, device=dev) / 4).to(torch.bfloat16)
        gcpu = torch.Generator().manual_seed(2027)
        steps = []
        for _s in range(STEPS):
            per_layer = []
            for _li in range(L):
                ids = torch.stack([torch.randperm(E, generator=gcpu)[:k] for _ in range(B)]).reshape(-1).to(torch.int32).to(dev)
                row0, rows, grp, order, _c = build_group_tiles_fused(ids, E, 16)
                per_layer.append((ids, row0, rows, grp, order, xg.index_select(0, order).contiguous()))
            steps.append(per_layer)

        def k25(lut, bf16, plan):
            def call(x, w, a, row0, rows, grp, order):
                return gemm_nf4_grouped_smallm(x, w, a, row0, rows, grp, order, lut=lut, dot_bf16=bf16, **plan)
            return call

        def arm(fn_gu_dn, sl):
            def f():
                for li, (ids, row0, rows, grp, order, xs) in enumerate(sl):
                    fn_gu_dn(li, row0, rows, grp, order, xs)
            return f

        def k25_arm(lut, bf16, plan):
            call = k25(lut, bf16, plan)

            def both(li, row0, rows, grp, order, xs):
                (gp, ga), (dp, da) = WL[li]["gu"], WL[li]["dn"]
                call(xg, gp, ga, row0, rows, grp, order)
                call(xd, dp, da, row0, rows, grp, None)
            return both

        def served_both(li, row0, rows, grp, order, xs):
            (gp, ga), (dp, da) = WL[li]["gu"], WL[li]["dn"]
            gemm_4bit_grouped_captured(xs, gp, ga, row0, rows, grp, 16)
            gemm_4bit_grouped_captured(xd, dp, da, row0, rows, grp, 16)

        # numerics and the error proxy on step 0, layer 0: every row, both projections, against fp64(fp32 dequant)
        ids, row0, rows, grp, order, xs = steps[0][0]
        sids = ids.index_select(0, order).cpu()
        err = {}
        outs = {}
        for name, fn in (("pair16", k25("pair", True, DEFAULT)), ("tree16", k25("tree", True, DEFAULT)),
                         ("tree32", k25("tree", False, DEFAULT))):
            outs[name] = (fn(xg, *WL[0]["gu"], row0, rows, grp, order), fn(xd, *WL[0]["dn"], row0, rows, grp, None))
        outs["served"] = (gemm_4bit_grouped_captured(xs, *WL[0]["gu"], row0, rows, grp, 16),
                          gemm_4bit_grouped_captured(xd, *WL[0]["dn"], row0, rows, grp, 16))
        refs = []
        for proj, x_sorted in (("gu", xs), ("dn", xd)):
            N, K = c[proj]
            w, a = WL[0][proj]
            deq = {int(e): dequant_ref(w[int(e)].cpu(), a[int(e)].cpu(), N, K).double() for e in torch.unique(sids)}
            refs.append(torch.stack([x_sorted[r].double().cpu() @ deq[int(sids[r])].t() for r in range(x_sorted.shape[0])]))
        for name, (yg, yd) in outs.items():
            d = torch.cat([(yg.double().cpu() - refs[0]).reshape(-1), (yd.double().cpu() - refs[1]).reshape(-1)])
            err[name] = float(d.pow(2).mean().sqrt())
        f = {"E": E, "k": k, "layers": L, "rms_err": err,
             "tree16_bit_equal_pair16": bool(torch.equal(outs["tree16"][0], outs["pair16"][0])
                                             and torch.equal(outs["tree16"][1], outs["pair16"][1])),
             "served_sane": err["served"] <= 2 * err["pair16"]}
        # timing: pair16 and served at their fixed plans; each tree arm selects its plan on SELECT, reads it on EVAL
        f["select"] = {}
        best = {}
        for name, lut, bf16 in (("tree16", "tree", True), ("tree32", "tree", False)):
            rows_ = []
            for plan in GRID:
                try:
                    ms = statistics.median(graph_ms(arm(k25_arm(lut, bf16, plan), steps[s])) for s in SELECT)
                    rows_.append({"plan": plan, "select_ms": ms})
                except Exception as e:                                   # recorded, never dropped
                    rows_.append({"plan": plan, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            f["select"][name] = rows_
            ok = [r for r in rows_ if "select_ms" in r]
            best[name] = min(ok, key=lambda r: r["select_ms"])["plan"] if ok else None
        f["best_plan"] = best
        ev = {"pair16": [graph_ms(arm(k25_arm("pair", True, DEFAULT), steps[s])) for s in EVAL],
              "served": [graph_ms(arm(served_both, steps[s])) for s in EVAL]}
        for name, lut, bf16 in (("tree16", "tree", True), ("tree32", "tree", False)):
            if best[name] is not None:
                ev[name] = [graph_ms(arm(k25_arm(lut, bf16, best[name]), steps[s])) for s in EVAL]
        f["eval_ms_all"] = ev
        f["eval_ms"] = {a: statistics.median(v) for a, v in ev.items()}
        rec["families"][fam] = f
        WL = steps = None
        torch.cuda.empty_cache()
        print(f"K27 {fam}: " + ", ".join(f"{a} {m:.3f}" for a, m in f["eval_ms"].items()) + " ms/step; best "
              + json.dumps(best) + "; rms err " + ", ".join(f"{a} {v:.3e}" for a, v in err.items())
              + f"; tree16==pair16 {f['tree16_bit_equal_pair16']}", flush=True)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["verdict"], rec["verdict_reason"] = verdict(rec)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K27_VERDICT {rec['verdict']}: {rec['verdict_reason']}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    if not a.out:
        ap.error("out is required")
    sys.exit(main(a.out, quick=a.quick))
