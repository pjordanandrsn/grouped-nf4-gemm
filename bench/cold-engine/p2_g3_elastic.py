"""P2-G3: elasticity under real VRAM pressure. Registered in
bench/cold-engine/PREREG-p2-g3.md.

Real components end to end: SegmentedRowPool (real shrink()/grow()),
DevRowCache segments, the G1b burst copy path (side stream, one event per
layer burst, pre-staged ids), the MXFP4 CPU and GPU kernels on the same
packed bytes, and real torch ballast against the real allocator. Per spec
S4.4 a row filled this step executes on CPU this step -- promotion never
stalls -- so GPU work is resident hits only, per-segment gemms.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))    # repo root: gnf4_native
sys.path.insert(0, os.path.join(HERE, "..", "..", "kernel"))
sys.path.insert(0, os.path.join(HERE, "routing-trace"))
sys.path.insert(0, HERE)

import cpu_grouped as cg                                    # noqa: E402
import gnf4_native                                          # noqa: E402
from dev_row_cache import StepTag                           # noqa: E402
from mxfp4_grouped import gemm_mxfp4_grouped                # noqa: E402
from segmented_pool import SegmentedRowPool                 # noqa: E402
from replay_dev_cache import load                           # noqa: E402
import p2_g1_promotion as g1                                # noqa: E402

N, K = 5760, 2880
PB = N * (K // 2)
SB = N * (K // 32)
ROWBYTES = PB + SB
SEG_ROWS = 64
RESERVE = 2 * 2**30
CONVERGE, HOLD, RECOVER = 64, 64, 64
PRESSURE_AT = CONVERGE
RELEASE_AT = CONVERGE + HOLD
TOTAL = CONVERGE + HOLD + RECOVER


def pair_index(recs):
    pairs = sorted({(int(L), e) for r in recs
                    for L, ex in r["routed"].items() for e in ex})
    return {p: i for i, p in enumerate(pairs)}


def run_trace(meta, recs, pk, sc, pidx, threads, shrink_enabled=True):
    k = int(meta["top_k"])
    layers = int(meta["layers"])
    m = layers * k
    pairs = len(pidx)
    segments = -(-int(0.7 * pairs) // SEG_ROWS)
    pool = SegmentedRowPool(segments, SEG_ROWS, ROWBYTES, device="cuda",
                            routed=k)
    side = torch.cuda.Stream()
    smooth_cap = -(-m // 4)
    a32 = torch.randn(m, K, dtype=torch.float32)
    a16 = a32.to(torch.bfloat16).cuda()
    stage = torch.empty(SEG_ROWS, dtype=torch.int32).pin_memory()

    # ---- no-cache baseline: 16 all-CPU steps
    base = []
    for r in recs[:16]:
        eids = [pidx[(int(L), e)] for L, ex in r["routed"].items() for e in ex]
        t0 = time.perf_counter()
        g1.cpu_exec(a32, pk, sc, eids, threads)
        base.append(time.perf_counter() - t0)
    wall_nocache = statistics.median(base)

    walls, caps, misses_s = [], [], []
    ballast = None
    oom = None
    shrink_wall = None
    for t in range(TOTAL):
        r = recs[t % len(recs)]
        if t == PRESSURE_AT:
            want_free = -(-segments // 2)
            t0 = time.perf_counter()
            if shrink_enabled:
                pool.shrink(want_free)
            torch.cuda.synchronize()
            shrink_wall = time.perf_counter() - t0
            # freed segments sit in torch's cache, invisible to mem_get_info;
            # hand them to the driver so ballast sizing (and the spoiler's
            # guaranteed-OOM sizing) is measured against real free VRAM
            torch.cuda.empty_cache()
            free_now, _ = torch.cuda.mem_get_info()
            bal_bytes = (free_now - RESERVE) if shrink_enabled else \
                        (free_now - RESERVE + want_free * pool.seg_bytes())
            try:
                ballast = torch.empty(bal_bytes, dtype=torch.uint8,
                                      device="cuda")
            except torch.cuda.OutOfMemoryError:
                oom = t
                if shrink_enabled:
                    break               # clause 1 fails; stop, report
                else:
                    break               # spoiler: expected
        if t == RELEASE_AT:
            del ballast
            ballast = None
            torch.cuda.empty_cache()
            pool.grow(pool.shrunk_segments - pool.grown_segments)
        budget = smooth_cap
        step_miss = 0
        t0 = time.perf_counter()
        gpu_by_seg = {}
        cold = []
        for lay in sorted(r["routed"], key=int):
            L = int(lay)
            ex = [pidx[(L, e)] for e in r["routed"][lay]]
            tag = StepTag("cuda")
            placed, need, skipped = pool.want(L, ex, tag, budget=budget)
            budget -= len(need)
            fill_by_seg = {}
            for key in need:
                si, slot = placed[key]
                fill_by_seg.setdefault(si, []).append((key, slot))
            with torch.cuda.stream(side):
                for si, items in fill_by_seg.items():
                    blocks, scales = pool.views(si, (N, K // 2), (N, K // 32), PB)
                    for (L_, e_), slot in items:
                        blocks[slot].copy_(pk[e_], non_blocking=True)
                        scales[slot].copy_(sc[e_], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(side)
            torch.cuda.current_stream().wait_event(ev)
            need_set = set(need)
            skip_set = set(skipped)
            for i, e_arena in enumerate(ex):
                key = (L, e_arena)
                if key in need_set or key in skip_set:
                    cold.append(e_arena)
                else:
                    si, slot = placed[key]
                    gpu_by_seg.setdefault(si, []).append((i, slot))
            step_miss += len(need) + len(skipped)
            tag.record()
        for si, items in gpu_by_seg.items():
            blocks, scales = pool.views(si, (N, K // 2), (N, K // 32), PB)
            rows = torch.tensor([i for i, _ in items], dtype=torch.long)
            slots = [s_ for _, s_ in items]
            gemm_mxfp4_grouped(a16[rows], blocks, scales,
                               [1] * len(items), slots)
        if cold:
            g1.cpu_exec(a32, pk, sc, cold, threads)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        caps.append(pool.rows_capacity())
        misses_s.append(step_miss)
    return {"walls": walls, "caps": caps, "misses": misses_s,
            "wall_nocache": wall_nocache, "shrink_wall_s": shrink_wall,
            "oom_at": oom, "segments": segments, "m": m,
            "stats": pool.stats()}


def clauses(res):
    walls = res["walls"]
    pre = statistics.median(walls[32:CONVERGE])
    c1 = res["oom_at"] is None and len(walls) == TOTAL
    c2 = (res["shrink_wall_s"] is not None
          and res["shrink_wall_s"] <= 2 * pre)
    c3 = all(w <= 1.10 * res["wall_nocache"] for w in walls)
    tail = walls[RELEASE_AT:TOTAL]
    rec_wall = None
    for i in range(16, len(tail) + 1):
        if statistics.fmean(tail[i - 16:i]) <= 1.10 * pre:
            rec_wall = RELEASE_AT + i - 1
            break
    c4 = (rec_wall is not None and rec_wall <= RELEASE_AT + 64
          and res["caps"][-1] == res["segments"] * SEG_ROWS)
    return {"steady_pre_s": pre, "c1_no_oom": c1, "c2_shrink_fast": c2,
            "c3_no_cliff": c3, "c4_recovered": c4,
            "recovered_at": rec_wall,
            "max_wall_over_nocache": max(w / res["wall_nocache"]
                                         for w in walls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    gnf4_native.load()
    torch.set_num_threads(1)
    cal = json.load(open(a.calib))
    if not (2 <= cal["nstar_direct"] <= 5):
        sys.exit("box fails the n* gate: %.2f" % cal["nstar_direct"])

    out = {"traces": {}, "calib_nstar": cal["nstar_direct"]}
    verdicts = []
    tdir = os.path.join(HERE, "rank-2026-08-22")
    for tr in ("gptoss_code.jsonl", "qwen_code.jsonl"):
        meta, recs = load(os.path.join(tdir, tr))
        pidx = pair_index(recs)
        print("%s: %d pairs, building arena..." % (tr, len(pidx)), flush=True)
        _N, _K, pk, sc, _ = g1.build(E=len(pidx))
        res = run_trace(meta, recs, pk, sc, pidx, a.threads)
        cl = clauses(res)
        # correctness spot-check: one resident row, GPU vs bf16-fed ref
        ok = g1.correctness(pk, sc, N, K, a.threads, sample=2)
        cl["correctness"] = ok
        void = not all(v for k_, v in ok.items() if isinstance(v, bool))
        cl["void"] = void
        print(tr, json.dumps({k_: v for k_, v in cl.items()
                              if k_ != "correctness"}, default=str))
        sp = run_trace(meta, recs, pk, sc, pidx, a.threads,
                       shrink_enabled=False)
        cl["spoiler_oom_at"] = sp["oom_at"]
        spoiler_ok = sp["oom_at"] is not None
        print("  spoiler (shrink disabled) OOM at:", sp["oom_at"])
        out["traces"][tr] = {"clauses": cl, "walls": res["walls"],
                             "caps": res["caps"], "misses": res["misses"],
                             "wall_nocache": res["wall_nocache"],
                             "stats": res["stats"]}
        verdicts.append((not void) and spoiler_ok
                        and cl["c1_no_oom"] and cl["c2_shrink_fast"]
                        and cl["c3_no_cliff"] and cl["c4_recovered"])
        del pk, sc
    spoilers_ok = all(out["traces"][t]["clauses"]["spoiler_oom_at"] is not None
                      for t in out["traces"])
    if not spoilers_ok:
        verdict = "UNINFORMATIVE"
    elif all(verdicts):
        verdict = "PASS"
    else:
        verdict = "REFUTED"
    out["verdict"] = verdict
    print("G3:", verdict)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
