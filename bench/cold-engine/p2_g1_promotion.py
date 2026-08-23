"""P2-G1: do the promotion mechanics pay what the calibrated model predicts?

Registered in bench/cold-engine/PREREG-p2-g1.md. Real components end to end:
the same packed MXFP4 bytes consumed by the phase-2 CPU kernel from
(page-locked) host memory and by the oracle-adjudicated GPU kernel after
promotion; pinned H2D on a side stream; first GPU use gated on the copy
event; retention in a transient DevRowCache with protected = rows - k.

The transient pool IS a DevRowCache: its byte buffer is viewed with
as_strided as the [rows, N, K/2] blocks stack and the [rows, N, K/32] scales
stack the GPU kernel wants -- the kernel takes explicit expert strides, so no
gather copy exists to pollute the measurement.

Registered conservatisms, stated here as in the prereg:
  * the routing stream is NO-REUSE -- every promotion is a pure
    copy-plus-execute (the n* = 1 regime); retention is verified by counters,
    not wall;
  * predicted Delta uses the box's elastic_e3 calibration (NF4 CPU rate, bf16
    GPU proxy), both of which overstate Delta for this MXFP4 harness -- the
    bar is therefore strict, and the in-situ terms are reported beside it.
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "kernel"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import cpu_grouped as cg                                   # noqa: E402
import gnf4_native                                          # noqa: E402
from dev_row_cache import DevRowCache, StepTag              # noqa: E402
from mxfp4_grouped import gemm_mxfp4_grouped                # noqa: E402
import phase2_gemv_bench as p2                              # noqa: E402

SHAPE = "gptossish_gateup"          # N=5760, K=2880
STEPS = 32
M = 16                              # cold invocations per step
P_SWEEP = (1, 2, 4, 8)
ROWS_T = 300                        # >= max(P_SWEEP) * STEPS: no eviction
SCRUB_FLOATS = 256 * 1024 * 1024    # 1 GiB fp32 — > any x86 L3; read before
                                    # each timed arm so neither arm inherits
                                    # the other's cache state (review round 1)


def build(seed=20260823, E=640):
    N, K = p2.SHAPES[SHAPE]
    keep, packed, scales, arena_bytes = p2.build_arena("mxfp4", 1, E, N, K,
                                                       seed)
    # Page-locked host arena: BOTH tiers read these bytes -- the CPU kernel
    # directly, the promotion path by DMA at full link rate. One artifact.
    pk = torch.from_numpy(packed[0]).pin_memory()
    sc = torch.from_numpy(scales[0]).pin_memory()
    return N, K, pk, sc, keep


class Transient:
    """The transient pool: a DevRowCache whose buffer is viewed as the GPU
    kernel's two stacked tensors. protected = rows - k (spec I1)."""

    def __init__(self, N, K, rows=ROWS_T, k=8):
        self.pb = N * (K // 2)
        self.sb = N * (K // 32)
        self.rowbytes = self.pb + self.sb
        self.cache = DevRowCache(rows, self.rowbytes, device="cuda", routed=k)
        buf = self.cache.buf
        self.blocks = torch.as_strided(buf, (rows, N, K // 2),
                                       (self.rowbytes, K // 2, 1))
        self.scales = torch.as_strided(buf, (rows, N, K // 32),
                                       (self.rowbytes, K // 32, 1),
                                       storage_offset=self.pb)

    def promote(self, eids, pk, sc, stream, tag):
        """Assign slots and enqueue the copies on `stream`. Returns
        (slot per eid, per-row events)."""
        assign, need = self.cache.want(0, eids, tag)
        evs = {}
        with torch.cuda.stream(stream):
            for e in need:
                s = assign[e]
                self.blocks[s].copy_(pk[e], non_blocking=True)
                self.scales[s].copy_(sc[e], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(stream)
                evs[e] = ev
        self.cache.note_filled(len(need))
        return assign, evs, len(need)


def cpu_exec(a_cat, pk, sc, eids, threads):
    cg.gemv_mxfp4_grouped_cpu(a_cat[:len(eids)], pk, sc,
                              [1] * len(eids), list(eids), threads=threads)


def run_arms(pk, sc, N, K, ids_stream, p, threads, scrub, spoiler=False):
    """One paired A/B pass. Returns per-step walls and counters."""
    a32 = torch.randn(M, K, dtype=torch.float32)
    a16 = a32[:max(p, 1)].to(torch.bfloat16).cuda()
    side = torch.cuda.Stream()
    tr = Transient(N, K)
    walls_a, walls_b = [], []
    copies = 0
    tags = []
    for step in range(STEPS):
        ids = ids_stream[step]
        # ---- arm A: everything on CPU
        float(scrub.sum())          # untimed: equalize cache state
        t0 = time.perf_counter()
        cpu_exec(a32, pk, sc, ids, threads)
        walls_a.append(time.perf_counter() - t0)
        # ---- arm B: promote p, CPU does the rest concurrently
        promo, rest = ids[:p], ids[p:]
        float(scrub.sum())          # untimed: arm B must not read arm A's L3
        t0 = time.perf_counter()
        if p:
            tag = StepTag("cuda")
            if spoiler:
                # Registered spoiler: synchronous default-stream copies --
                # the regime phase 0 showed loses. Must FAIL the bar.
                assign, _ = tr.cache.want(0, promo, tag)
                for e in promo:
                    tr.blocks[assign[e]].copy_(pk[e], non_blocking=False)
                    tr.scales[assign[e]].copy_(sc[e], non_blocking=False)
                torch.cuda.synchronize()
                tr.cache.note_filled(len(promo))
                evs = {}
                copies += len(promo)
            else:
                assign, evs, n = tr.promote(promo, pk, sc, side, tag)
                copies += n
        if rest:
            cpu_exec(a32, pk, sc, rest, threads)
        if p:
            for ev in evs.values():
                torch.cuda.current_stream().wait_event(ev)
            slots = [assign[e] for e in promo]
            gemm_mxfp4_grouped(a16[:p], tr.blocks, tr.scales, [1] * p, slots)
            torch.cuda.synchronize()
            tag.record()
            tags.append(tag)
        walls_b.append(time.perf_counter() - t0)
    return (statistics.median(walls_a), statistics.median(walls_b), copies,
            tr, ids_stream)


def correctness(pk, sc, N, K, threads, sample=4):
    """Registered checks, each against its kernel's own committed contract:
    CPU bit-exact vs the numpy executable spec (numpy in, the committed
    tests' calling convention); GPU within its committed bound with the
    reference built from the SAME bf16-rounded activations the kernel sees
    (test_mxfp4_interp's procedure); promoted bytes identical."""
    rng = np.random.default_rng(1)
    eids = rng.choice(pk.shape[0], size=sample, replace=False).tolist()
    a32 = torch.randn(sample, K, dtype=torch.float32)
    pk_np, sc_np = pk.numpy(), sc.numpy()
    # CPU contract: fp32 activations; ref takes numpy, kernel takes torch.
    ref_cpu = cg.ref_gemv_grouped(a32.numpy(), pk_np, sc_np,
                                  [1] * sample, eids, fmt="mxfp4")
    got_cpu = cg.gemv_mxfp4_grouped_cpu(a32, pk, sc, [1] * sample, eids,
                                        threads=threads)
    cpu_exact = np.array_equal(got_cpu.numpy(), ref_cpu)
    tr = Transient(N, K)
    side = torch.cuda.Stream()
    tag = StepTag("cuda")
    assign, evs, _ = tr.promote(eids, pk, sc, side, tag)
    for ev in evs.values():
        torch.cuda.current_stream().wait_event(ev)
    byte_ok = all(
        torch.equal(tr.blocks[assign[e]].cpu(), pk[e]) and
        torch.equal(tr.scales[assign[e]].cpu(), sc[e]) for e in eids)
    # GPU contract: the committed bound is asserted against a reference fed
    # the same bf16-rounded activations the kernel consumes.
    a16 = a32.to(torch.bfloat16)
    ref_gpu = cg.ref_gemv_grouped(a16.float().numpy(), pk_np, sc_np,
                                  [1] * sample, eids, fmt="mxfp4")
    got_gpu = gemm_mxfp4_grouped(a16.cuda(), tr.blocks, tr.scales,
                                 [1] * sample, [assign[e] for e in eids])
    tag.record()
    diff = np.abs(got_gpu.float().cpu().numpy() - ref_gpu)
    rel = float(diff.max() / np.abs(ref_gpu).max())
    # retention: re-invoking produces hits and zero new H2D
    before = tr.cache.filled
    tag2 = StepTag("cuda")
    assign2, need2 = tr.cache.want(0, eids, tag2)
    tag2.record()
    retained = (len(need2) == 0 and tr.cache.filled == before
                and all(assign2[e] == assign[e] for e in eids))
    return {"cpu_bit_exact": bool(cpu_exact), "bytes_identical": bool(byte_ok),
            "gpu_rel_max": rel, "gpu_within_committed_2e-2": rel < 2e-2,
            "retention_zero_h2d": bool(retained)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--calib", required=True,
                    help="elastic_e3 receipt for this box (the registered "
                         "source of the predicted Delta)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # The 1 GiB scrub reads run on torch's intra-op pool; at its default width
    # (every core) the pool's spin-wait storms straight into the promote/launch
    # calls that follow and inflates arm B's host dispatch ~10x (measured 215 us
    # enqueue vs 20 us clean on the same box). One thread keeps the scrub a
    # scrub -- cache eviction, not a CPU contention generator the model never
    # charged for. The GEMV pool is native and unaffected.
    torch.set_num_threads(1)

    assert torch.cuda.is_available()
    gnf4_native.load()
    cal = json.load(open(a.calib))
    if not (2 <= cal["nstar_direct"] <= 5):
        sys.exit("box fails the registered calibration gate: n*=%.2f"
                 % cal["nstar_direct"])

    N, K, pk, sc, keep = build()
    scrub = torch.ones(SCRUB_FLOATS, dtype=torch.float32)
    rowbytes = N * (K // 2) + N * (K // 32)
    hide = cal["e3b"]["hidden_frac"]
    d_pred = (rowbytes / (cal["B_cpu_gbs"] * 1e9)
              - (1 - hide) * rowbytes / (cal["B_link_gbs"] * 1e9)
              - rowbytes / (cal["B_gpu_gbs"] * 1e9))

    print("row=%d B  predicted Delta=%.1f us/row  bar=%.1f us/row"
          % (rowbytes, d_pred * 1e6, 0.7 * d_pred * 1e6))
    checks = correctness(pk, sc, N, K, a.threads)
    print("correctness:", json.dumps(checks))
    if not (checks["cpu_bit_exact"] and checks["bytes_identical"]
            and checks["gpu_within_committed_2e-2"]
            and checks["retention_zero_h2d"]):
        sys.exit("CORRECTNESS FAILED -- wall numbers are void, per the "
                 "preregistration.")

    # Registered harness validation: p = 0 must reduce arm B to arm A --
    # walls within repeat spread, zero copies enqueued.
    rng0 = np.random.default_rng(3)
    perm = rng0.permutation(pk.shape[0])[:STEPS * M]
    stream0 = [perm[s * M:(s + 1) * M].tolist() for s in range(STEPS)]
    wa0, wb0, c0, _, _ = run_arms(pk, sc, N, K, stream0, 0, a.threads, scrub)
    spread = abs(wa0 - wb0) / wa0
    print("p=0 validation: wall_A %.3f ms  wall_B %.3f ms  (%.1f%% apart), "
          "copies=%d" % (wa0 * 1e3, wb0 * 1e3, 100 * spread, c0))
    if c0 != 0 or spread > 0.10:
        sys.exit("p=0 VALIDATION FAILED -- the harness is being scored, "
                 "not the mechanics.")

    rng = np.random.default_rng(7)
    rows = []
    print("\n%4s %10s %10s %12s %14s %10s" % (
        "p", "wall_A ms", "wall_B ms", "save/row us", "realized/pred", "arm"))
    for spoiler in (False, True):
        for p in P_SWEEP:
            per = []
            for rep in range(a.repeats):
                perm = rng.permutation(pk.shape[0])[:STEPS * M]
                stream = [perm[s * M:(s + 1) * M].tolist()
                          for s in range(STEPS)]
                wa, wb, copies, tr, _ = run_arms(pk, sc, N, K, stream, p,
                                                 a.threads, scrub,
                                                 spoiler=spoiler)
                exp = p * STEPS
                if not spoiler and copies != exp:
                    sys.exit("counter accounting: %d copies, expected %d"
                             % (copies, exp))
                per.append((wa, wb))
            wa = statistics.median(x[0] for x in per)
            wb = statistics.median(x[1] for x in per)
            save = (wa - wb) / p
            rows.append({"p": p, "spoiler": spoiler, "wall_a_s": wa,
                         "wall_b_s": wb, "save_per_row_s": save,
                         "realized_over_predicted": save / d_pred})
            print("%4d %10.3f %10.3f %12.1f %14.2f %10s" % (
                p, wa * 1e3, wb * 1e3, save * 1e6, save / d_pred,
                "SPOILER" if spoiler else "side-stream"))

    real = [r for r in rows if not r["spoiler"]]
    spoil = [r for r in rows if r["spoiler"]]
    g1 = all(r["realized_over_predicted"] >= 0.70 for r in real)
    falsified = all(r["realized_over_predicted"] < 0.70 for r in spoil)
    verdict = ("UNINFORMATIVE (spoiler passed the bar)" if not falsified
               else ("PASS" if g1 else "REFUTED"))
    print("\nspoiler fails the bar everywhere:", falsified)
    print("G1:", verdict)
    with open(a.out, "w") as f:
        json.dump({"shape": SHAPE, "rowbytes": rowbytes, "steps": STEPS,
                   "m": M, "delta_pred_s": d_pred, "hide_used": hide,
                   "calib": cal, "correctness": checks, "rows": rows,
                   "verdict": verdict,
                   "gpu": torch.cuda.get_device_name(0)}, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
