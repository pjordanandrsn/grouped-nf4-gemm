# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k7 bench harness — the committed instrument
([[commit-the-instrument-not-just-receipts]]).

Census pair at the decode shapes (gate_up 1536x2048, down 2048x768;
E=8, T=8 one-token routes): gates the split-K dot-pad candidate
against the fp32 dequant reference (G1, the K6 2^-7 band), checks
run-to-run bitwise determinism on the winner (G2), times dot-pad
baseline (A/A noise gate, G3) and the config sweep on CUDA-graph
replay (chunked median, the S2/S3 basis), and writes the
k7_verdict-shaped report plus the full sweep table.

Run on the box:  python kernel/k7_bench.py --out k7_report.json \
                     --sweep-out k7_sweep.json
"""

import argparse
import json
import os

os.environ["GNF4_GEMV_DOTPAD"] = "1"   # the K6-B baseline path

import torch

import nf4_grouped
from nf4_grouped import _sm_count, dequant_ref, gemm_4bit_grouped
from nf4_pack_ref import make_stack

CELLS = {"gate_up": (1536, 2048), "down": (2048, 768)}
SEEDS = {"gate_up": 11, "down": 22}   # fixed: str hash() is per-process
E, T = 8, 8
# sk=1 rows are the prereg's SECOND treatment (retune at the census
# shapes with no split-K); without them a pure-config win could not
# be found and the sweep would silently pre-judge the mechanism
SWEEP = [(bn, w, s, sk) for bn in (16, 32) for w in (2, 4)
         for s in (2, 3) for sk in (1, 2, 4, 8, 16)]
CHUNKS, REPLAYS = 10, 8
NOISE_BAR = 0.02
REL_BAR = 2.0 ** -7


def build_cell(N, K, seed):
    B, A = make_stack(E, N, K, seed=seed, device="cuda")
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    a = torch.randn(T, K, generator=g, dtype=torch.float32)
    a_cat = a.to(torch.bfloat16).cuda()
    eids = torch.arange(T, dtype=torch.int32, device="cuda") % E
    ref = torch.empty(T, N, dtype=torch.float32, device="cuda")
    for t in range(T):
        w = dequant_ref(B[eids[t]], A[eids[t]], N, K)   # [N, K] fp32
        # the kernel consumes the bf16 activation; reference reads the
        # SAME post-cast value so the gate measures the kernel, not the
        # input cast
        ref[t] = a_cat[t].float() @ w.t()
    return {"B": B, "A": A, "a": a_cat, "eids": eids, "ref": ref,
            "N": N, "K": K}


def call(cell, split_k=None):
    return gemm_4bit_grouped(cell["a"], cell["B"], cell["A"], [1] * T,
                             cell["eids"], split_k=split_k)


def gate(y, ref):
    d = (y.float() - ref).abs().max().item()
    mr = ref.abs().max().item()
    agree = int((y.float().argmax(-1) == ref.argmax(-1)).sum().item())
    return {"max_abs_delta": d, "max_abs_ref": mr,
            "argmax_agree": agree, "argmax_total": T,
            "rel_ok": d <= mr * REL_BAR and agree / T >= 0.99}


def time_us(fn):
    """CUDA-graph replay, chunked median per call, microseconds."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    spans = []
    for _ in range(CHUNKS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(REPLAYS):
            g.replay()
        e1.record()
        e1.synchronize()
        spans.append(e0.elapsed_time(e1) * 1e3 / REPLAYS)
    spans.sort()
    return spans[len(spans) // 2]


def self_test():
    """Exercise the SHIPPED pipeline on any CUDA box, tiny shape, no
    report written: build -> gate -> sweep -> G2 -> compose -> feed
    k7_verdict. Catches harness bugs on hardware that cannot
    adjudicate the prereg (e.g. a 26-SM A2000) before a census box
    spends money discovering them. The SM guard is lifted HERE ONLY,
    and no verdict-shaped file leaves this path."""
    import k7_verdict

    N, K = 64, 128
    nf4_grouped._sm_count = lambda dev: 200
    nf4_grouped._DOTPAD_CONFIGS[(N, K)] = (16, 2, 2)
    c = build_cell(N, K, seed=1)
    base = gate(call(c, split_k=1), c["ref"])
    assert base["rel_ok"], base
    t_a, t_b = time_us(lambda: call(c, split_k=1)), \
        time_us(lambda: call(c, split_k=1))
    y1, y2 = call(c, split_k=2), call(c, split_k=2)
    g2 = gate(y1, c["ref"])
    g2["bitwise_repeat"] = bool(torch.equal(y1, y2))
    assert g2["rel_ok"] and g2["bitwise_repeat"], g2
    cell = {"gate": g2, "dotpad_us": min(t_a, t_b),
            "best_us": time_us(lambda: call(c, split_k=2))}
    rep = {"summary": {"noise_gate_pass": True,
                       "dotpad_pair_us": cell["dotpad_us"] * 2,
                       "candidate_pair_us": cell["best_us"] * 2},
           "cells": {"gate_up": cell, "down": cell}}
    tag, x = k7_verdict.verdict(rep)
    assert tag != "REFUSE", (tag, x)
    print(f"k7_bench self-test OK (report validates; this box's "
          f"non-census ratio {x:.3f} is NOT a verdict) "
          f"aa=[{t_a:.1f}, {t_b:.1f}] us", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--sweep-out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.out and a.sweep_out):
        ap.error("--out and --sweep-out (or --self-test)")
    assert _sm_count(torch.device("cuda")) >= 160, \
        "K7 census cells are gated to >= 160-SM parts (the dot-pad " \
        "config guard); this box cannot adjudicate the prereg"

    cells, sweep_tbl = {}, {}
    noise = []
    for name, (N, K) in CELLS.items():
        c = build_cell(N, K, seed=SEEDS[name])
        base_gate = gate(call(c, split_k=1), c["ref"])
        assert base_gate["rel_ok"], (name, "K6-B baseline failed its "
                                     "own certified gate", base_gate)
        t_a = time_us(lambda: call(c, split_k=1))
        t_b = time_us(lambda: call(c, split_k=1))
        noise.append(abs(t_a - t_b) / min(t_a, t_b))
        dotpad_us = min(t_a, t_b)

        rows, best = [], None
        orig = nf4_grouped._DOTPAD_CONFIGS[(N, K)]
        try:
            for bn, w, s, sk in SWEEP:
                nf4_grouped._DOTPAD_CONFIGS[(N, K)] = (bn, w, s)
                try:
                    gg = gate(call(c, split_k=sk), c["ref"])
                except Exception as ex:  # noqa: BLE001 -- recorded, not judged
                    rows.append({"cfg": [bn, w, s, sk],
                                 "error": repr(ex)[:120]})
                    continue
                if not gg["rel_ok"]:
                    rows.append({"cfg": [bn, w, s, sk], "gate": gg,
                                 "gated_out": True})
                    continue
                t = time_us(lambda: call(c, split_k=sk))
                rows.append({"cfg": [bn, w, s, sk], "us": t})
                if best is None or t < best["us"]:
                    best = {"cfg": [bn, w, s, sk], "us": t, "gate": gg}
        finally:
            nf4_grouped._DOTPAD_CONFIGS[(N, K)] = orig
        assert best is not None, (name, "no split-K config passed the "
                                  "correctness gate -- REFUSE upstream")

        # G2 on the winner: two invocations, bitwise
        nf4_grouped._DOTPAD_CONFIGS[(N, K)] = tuple(best["cfg"][:3])
        try:
            y1 = call(c, split_k=best["cfg"][3])
            y2 = call(c, split_k=best["cfg"][3])
        finally:
            nf4_grouped._DOTPAD_CONFIGS[(N, K)] = orig
        best["gate"]["bitwise_repeat"] = bool(torch.equal(y1, y2))

        cells[name] = {"gate": best["gate"], "dotpad_us": dotpad_us,
                       "dotpad_aa": [t_a, t_b], "best_us": best["us"],
                       "best_config": best["cfg"],
                       "baseline_gate": base_gate}
        sweep_tbl[name] = rows

    rep = {"summary": {"noise_gate_pass": max(noise) <= NOISE_BAR,
                       "noise_spreads": noise,
                       "dotpad_pair_us": sum(c["dotpad_us"]
                                             for c in cells.values()),
                       "candidate_pair_us": sum(c["best_us"]
                                                for c in cells.values())},
           "cells": cells,
           "basis": "graph-replayed chunked-median (10 chunks x 8 "
                    "replays), same-box dot-pad denominator, "
                    "PREREG-k7"}
    json.dump(rep, open(a.out, "w"), indent=1)
    json.dump(sweep_tbl, open(a.sweep_out, "w"), indent=1)
    s = rep["summary"]
    print(f"K7_BENCH pair dotpad {s['dotpad_pair_us']:.1f} us -> "
          f"candidate {s['candidate_pair_us']:.1f} us "
          f"(ratio {s['candidate_pair_us'] / s['dotpad_pair_us']:.3f}) "
          f"noise {['%.4f' % n for n in noise]} out={a.out}",
          flush=True)


if __name__ == "__main__":
    main()
