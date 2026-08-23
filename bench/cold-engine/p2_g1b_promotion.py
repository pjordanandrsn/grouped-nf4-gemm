"""P2-G1b: the AMENDED promotion mechanics (spec #4, I9) against the amended
model. Registered in bench/cold-engine/PREREG-p2-g1b.md.

Reuses G1's arena, transient pool, CPU tier, scrub, and correctness gates
(p2_g1_promotion) verbatim -- the only new code is the dispatch under test:
one stream-ordered burst per step (single want(), one burst event covering
copies + id staging, pre-staged device ids, GPU work enqueued before the CPU
tier starts), plus the two registered spoilers.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

import p2_g1_promotion as g1
from p2_g1_promotion import (SHAPE, STEPS, M, ROWS_T, SCRUB_FLOATS,   # noqa: F401
                             Transient, build, cpu_exec, correctness)
from dev_row_cache import StepTag                           # noqa: E402
from mxfp4_grouped import gemm_mxfp4_grouped                # noqa: E402

P_SWEEP = (1, 2, 3, 4, 8)


def run_arms_burst(pk, sc, N, K, ids_stream, p, threads, scrub, mode="burst"):
    """One paired A/B pass. mode: 'burst' (the amended mechanics),
    'sync' (hide spoiler: blocking default-stream copies, burst kept),
    'serial' (dispatch spoiler: the un-amended G1 mechanics verbatim)."""
    a32 = torch.randn(M, K, dtype=torch.float32)
    a16 = a32[:max(p, 1)].to(torch.bfloat16).cuda()
    side = torch.cuda.Stream()
    tr = Transient(N, K)
    stage = torch.empty(max(p, 1), dtype=torch.int32).pin_memory()
    ids_dev = torch.empty(max(p, 1), dtype=torch.int32, device="cuda")
    walls_a, walls_b = [], []
    copies = 0
    tags = []
    for step in range(STEPS):
        ids = ids_stream[step]
        # ---- arm A: everything on CPU
        float(scrub.sum())
        t0 = time.perf_counter()
        cpu_exec(a32, pk, sc, ids, threads)
        walls_a.append(time.perf_counter() - t0)
        # ---- arm B
        promo, rest = ids[:p], ids[p:]
        float(scrub.sum())
        t0 = time.perf_counter()
        if p:
            tag = StepTag("cuda")
            if mode == "serial":
                # Un-amended G1 mechanics verbatim: per-row events, list ids,
                # launch only after the CPU tier returns.
                assign, evs, n = tr.promote(promo, pk, sc, side, tag)
                copies += n
                if rest:
                    cpu_exec(a32, pk, sc, rest, threads)
                for ev in evs.values():
                    torch.cuda.current_stream().wait_event(ev)
                slots = [assign[e] for e in promo]
                gemm_mxfp4_grouped(a16[:p], tr.blocks, tr.scales,
                                   [1] * p, slots)
                torch.cuda.synchronize()
                tag.record()
                tags.append(tag)
                walls_b.append(time.perf_counter() - t0)
                continue
            # Amended burst: single want; copies + id staging under ONE event.
            assign, need = tr.cache.want(0, promo, tag)
            if mode == "sync":
                for i, e in enumerate(promo):
                    s_ = assign[e]
                    tr.blocks[s_].copy_(pk[e], non_blocking=False)
                    tr.scales[s_].copy_(sc[e], non_blocking=False)
                    stage[i] = s_
                ids_dev[:p].copy_(stage[:p])
                torch.cuda.synchronize()          # the hide destroyed
            else:
                with torch.cuda.stream(side):
                    for i, e in enumerate(promo):
                        s_ = assign[e]
                        tr.blocks[s_].copy_(pk[e], non_blocking=True)
                        tr.scales[s_].copy_(sc[e], non_blocking=True)
                        stage[i] = s_
                    ids_dev[:p].copy_(stage[:p], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(side)
                torch.cuda.current_stream().wait_event(ev)
            tr.cache.note_filled(len(need))
            copies += len(need)
            # GPU work enqueued BEFORE the CPU tier starts (I9).
            gemm_mxfp4_grouped(a16[:p], tr.blocks, tr.scales,
                               [1] * p, ids_dev[:p])
        if rest:
            cpu_exec(a32, pk, sc, rest, threads)
        if p:
            torch.cuda.synchronize()
            tag.record()
            tags.append(tag)
        walls_b.append(time.perf_counter() - t0)
    return (statistics.median(walls_a), statistics.median(walls_b), copies,
            tr, ids_stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--calib", required=True,
                    help="elastic_e3 receipt (must carry dispatch_per_p_s)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    import gnf4_native
    gnf4_native.load()
    torch.set_num_threads(1)          # the scrub is a scrub (G1 lesson)
    cal = json.load(open(a.calib))
    if not (2 <= cal["nstar_direct"] <= 5):
        sys.exit("box fails the n* gate: %.2f" % cal["nstar_direct"])
    disp = cal.get("dispatch_per_p_s")
    if not disp:
        sys.exit("calibration receipt lacks dispatch_per_p_s -- rerun "
                 "elastic_e3.py at the amended commit")

    N, K, pk, sc, keep = build()
    scrub = torch.ones(SCRUB_FLOATS, dtype=torch.float32)
    rowbytes = N * (K // 2) + N * (K // 32)
    hide = cal["e3b"]["hidden_frac"]
    t_cpu_row = rowbytes / (cal["B_cpu_gbs"] * 1e9)
    t_gpu_row = rowbytes / (cal["B_gpu_gbs"] * 1e9)
    t_link_row = rowbytes / (cal["B_link_gbs"] * 1e9)
    delta = t_cpu_row - (1 - hide) * t_link_row - t_gpu_row
    link_cap = int(M * t_cpu_row / (t_cpu_row + t_link_row))
    p_min = None
    for p in P_SWEEP:
        c = disp.get(str(p))
        if c is not None and p * (t_cpu_row - t_gpu_row) > c:
            p_min = p
            break
    window = [p for p in P_SWEEP
              if p_min is not None and p_min <= p <= link_cap]
    print("t_cpu_row %.1f us  t_gpu_row %.1f us  t_link_row %.1f us  "
          "Delta %.1f us" % (t_cpu_row * 1e6, t_gpu_row * 1e6,
                             t_link_row * 1e6, delta * 1e6))
    print("C_disp:", {k: "%.1f us" % (v * 1e6) for k, v in disp.items()})
    print("feasible window: p_min=%s LINK_CAP=%d -> %s"
          % (p_min, link_cap, window))
    if not window:
        sys.exit("feasible window empty on this box -- registered box-gate "
                 "failure, no wall claimed")

    checks = correctness(pk, sc, N, K, a.threads)
    print("correctness:", json.dumps(checks))
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        sys.exit("correctness failed -- walls void")

    rng = np.random.default_rng(20260823)
    perm0 = rng.permutation(pk.shape[0])[:STEPS * M]
    stream0 = [perm0[s_ * M:(s_ + 1) * M].tolist() for s_ in range(STEPS)]
    wa0, wb0, c0, _, _ = run_arms_burst(pk, sc, N, K, stream0, 0,
                                        a.threads, scrub)
    gap = abs(wb0 - wa0) / wa0
    print("p=0 validation: wall_A %.3f ms  wall_B %.3f ms  (%.1f%% apart), "
          "copies=%d" % (wa0 * 1e3, wb0 * 1e3, gap * 100, c0))
    if gap > 0.10 or c0 != 0:
        sys.exit("p=0 validation failed")

    rows = []
    print("\n   p  wall_A ms  wall_B ms  save/step us   bar us  in-window"
          "        arm")
    for mode in ("burst", "sync", "serial"):
        for p in P_SWEEP:
            per = []
            for rep in range(a.repeats):
                perm = rng.permutation(pk.shape[0])[:STEPS * M]
                stream = [perm[s_ * M:(s_ + 1) * M].tolist()
                          for s_ in range(STEPS)]
                wa, wb, copies, tr, _ = run_arms_burst(
                    pk, sc, N, K, stream, p, a.threads, scrub, mode=mode)
                exp = p * STEPS
                if copies != exp:
                    sys.exit("counter accounting (%s p=%d): %d copies, "
                             "expected %d" % (mode, p, copies, exp))
                per.append((wa, wb))
            wa = statistics.median(x[0] for x in per)
            wb = statistics.median(x[1] for x in per)
            save = wa - wb
            c = disp.get(str(p), 0.0)
            bar = 0.70 * (p * delta - c)
            rows.append({"p": p, "mode": mode, "wall_a_s": wa,
                         "wall_b_s": wb, "save_step_s": save,
                         "bar_s": bar, "in_window": p in window,
                         "meets_bar": save >= bar})
            print("%4d %10.3f %10.3f %13.1f %8.1f %10s %10s" % (
                p, wa * 1e3, wb * 1e3, save * 1e6, bar * 1e6,
                "yes" if p in window else "no", mode))

    in_win = [r for r in rows if r["mode"] == "burst" and r["in_window"]]
    burst_pass = all(r["meets_bar"] for r in in_win)
    spoilers_fail = all(
        not r["meets_bar"] for r in rows
        if r["mode"] in ("sync", "serial") and r["in_window"])
    if not spoilers_fail:
        verdict = "UNINFORMATIVE"      # a spoiler cleared the bar
    elif burst_pass:
        verdict = "PASS"
    else:
        verdict = "REFUTED"
    print("\nspoilers fail everywhere in-window:", spoilers_fail)
    print("G1b:", verdict)
    rec = {"verdict": verdict, "rows": rows, "window": window,
           "p_min": p_min, "link_cap": link_cap, "delta_s": delta,
           "t_cpu_row_s": t_cpu_row, "t_gpu_row_s": t_gpu_row,
           "t_link_row_s": t_link_row, "shape": SHAPE, "m": M,
           "steps": STEPS, "threads": a.threads, "calib": cal,
           "correctness": checks,
           "gpu": torch.cuda.get_device_name(0)}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
