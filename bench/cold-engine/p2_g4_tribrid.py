"""P2-G4: the objective — overlap realised across NVMe, CPU, and GPU tiers.

Registered in bench/cold-engine/PREREG-p2-g4.md. Every component is the
shipped artifact: nvme_arena.bake (layout contract-tested against the
engine), ColdTier(pinned=True) as the DRAM hot set, ColdCpuView's copy path
materializing the kernel-shaped CONTIGUOUS stacks (the CPU kernel's
contract; stacks re-seated pinned pre-use so VRAM fills stay async),
SegmentedRowPool (#217) as the VRAM pool with the G1b burst copy path, and
the MXFP4 kernels on the same bytes end to end.

The prefetch thread is the ONLY ColdTier user between its spawn and the
next step's join (single-controller, like every allocator here): per step,
join previous prefetch -> ensure(t) on the main thread (warm hits) ->
spawn prefetch(t+1) -> compute overlaps the reads. ctypes and O_DIRECT
release the GIL, so the overlap is real.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kernel"))
sys.path.insert(0, os.path.join(HERE, "routing-trace"))
sys.path.insert(0, HERE)

import cpu_grouped as cg                                    # noqa: E402
import gnf4_native                                          # noqa: E402
from dev_row_cache import StepTag                           # noqa: E402
from mxfp4_grouped import gemm_mxfp4_grouped                # noqa: E402
from mxfp4_loader import (DOWN_BLOCKS, DOWN_SCALES,         # noqa: E402
                          GATE_UP_BLOCKS, GATE_UP_SCALES)
from nvme_arena import bake, load_index                     # noqa: E402
from nvme_residency import ColdTier                         # noqa: E402
from cold_cpu_view import ColdCpuView                       # noqa: E402
from segmented_pool import SegmentedRowPool                 # noqa: E402
from replay_dev_cache import load                           # noqa: E402
from test_nvme_arena import _st_bytes                       # noqa: E402

L_LAYERS, E_EXPERTS = 24, 32
N1, K1 = 5760, 2880                 # the G1 gate/up shape
N2, K2 = 2880, 2880                 # down projection: baked, not executed
HALF1, NB1 = K1 // 2, K1 // 32
HALF2, NB2 = K2 // 2, K2 // 32
PB = N1 * HALF1
ROWBYTES_VRAM = PB + N1 * NB1
SEG_ROWS = 64
HOT_ROWS = int(0.75 * L_LAYERS * E_EXPERTS)      # 576 of 768
STEADY_LO, STEADY_HI = 32, 64
TOTAL = 64


def build_snapshot(root, seed=20260823, sample=((0, 0), (0, 1), (5, 3))):
    """Full-size synthetic gpt-oss-shaped snapshot, one shard per layer.
    Returns {(layer, expert): {suffix: bytes}} for the sampled pairs — the
    live byte-identity check compares the view's rows against these."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    ground = {k: {} for k in sample}
    weight_map = {}
    shapes = {GATE_UP_BLOCKS: (N1, HALF1), GATE_UP_SCALES: (N1, NB1),
              DOWN_BLOCKS: (N2, HALF2), DOWN_SCALES: (N2, NB2)}
    for lay in range(L_LAYERS):
        shard = f"model-{lay:02d}.safetensors"
        tensors = {}
        for suf, es in shapes.items():
            name = f"model.layers.{lay}.{suf}"
            if suf.endswith("scales"):
                arr = rng.integers(100, 140, size=(E_EXPERTS, *es),
                                   dtype=np.uint8)
            else:
                arr = rng.integers(0, 256, size=(E_EXPERTS, *es),
                                   dtype=np.uint8)
            tensors[name] = ((E_EXPERTS, *es), arr.tobytes())
            weight_map[name] = shard
            for (sl, se) in ground:
                if sl == lay:
                    ground[(sl, se)][suf] = arr[se].tobytes()
        with open(os.path.join(root, shard), "wb") as f:
            f.write(_st_bytes(tensors))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    with open(os.path.join(root, "ground.json"), "w") as f:
        json.dump({f"{k[0]},{k[1]}": {suf: v.hex() for suf, v in d.items()}
                   for k, d in ground.items()}, f)
    return ground


def load_ground(root):
    with open(os.path.join(root, "ground.json")) as f:
        raw = json.load(f)
    return {tuple(int(x) for x in k.split(",")):
            {suf: bytes.fromhex(v) for suf, v in d.items()}
    for k, d in raw.items()}


def correctness_checks(view, tier, ground, recs, threads):
    """The prereg's three live checks, sampled."""
    checks = {}
    bs, ss = view.stacks[GATE_UP_BLOCKS], view.stacks[GATE_UP_SCALES]
    ok_bytes = True
    for (lay, e), segs in ground.items():
        slot = view.ensure(lay, [e])[0]
        got_b = bytes(bs[slot].contiguous().view(torch.uint8).numpy().tobytes())
        got_s = bytes(ss[slot].contiguous().view(torch.uint8).numpy().tobytes())
        ok_bytes &= got_b == segs[GATE_UP_BLOCKS]
        ok_bytes &= got_s == segs[GATE_UP_SCALES]
    checks["tier_rows_byte_identical"] = bool(ok_bytes)
    (lay, e) = next(iter(ground))
    slot = view.ensure(lay, [e])[0]
    a32 = torch.randn(2, K1, dtype=torch.float32)
    ref = cg.ref_gemv_grouped(a32.numpy(), bs.numpy(), ss.numpy(),
                              [1, 1], [slot, slot], fmt="mxfp4")
    got = cg.gemv_mxfp4_grouped_cpu(a32, bs, ss, [1, 1], [slot, slot],
                                    threads=threads)
    checks["cpu_bit_exact"] = bool(np.array_equal(got.numpy(), ref))
    pool = SegmentedRowPool(1, 8, ROWBYTES_VRAM, device="cuda", routed=4)
    tag = StepTag("cuda")
    placed, need, _ = pool.want(lay, [e], tag, budget=1)
    pb, ps = pool.views(0, (N1, HALF1), (N1, NB1), PB)
    _si, vslot = placed[(lay, e)]
    pb[vslot].copy_(bs[slot])
    ps[vslot].copy_(ss[slot])
    torch.cuda.synchronize()
    tag.record()
    a16 = a32.to(torch.bfloat16)
    ref_g = cg.ref_gemv_grouped(a16.float().numpy(), bs.numpy(), ss.numpy(),
                                [1, 1], [slot, slot], fmt="mxfp4")
    got_g = gemm_mxfp4_grouped(a16.cuda(), pb, ps, [1, 1], [vslot, vslot])
    rel = float(np.abs(got_g.float().cpu().numpy() - ref_g).max()
                / np.abs(ref_g).max())
    checks["gpu_within_committed_2e-2"] = bool(rel < 2e-2)
    return checks


def nvme_probe(path, gib=2):
    """O_DIRECT sequential read rate on a fresh file next to the arena."""
    fn = path + ".probe"
    blk = 8 * 2**20
    data = np.random.default_rng(1).integers(0, 256, size=blk, dtype=np.uint8)
    with open(fn, "wb") as f:
        for _ in range(gib * 2**30 // blk):
            f.write(data.tobytes())
        f.flush()
        os.fsync(f.fileno())
    if not hasattr(os, "O_DIRECT"):
        os.unlink(fn)
        return 0.0                      # no O_DIRECT: the gate fails cleanly
    fd = os.open(fn, os.O_RDONLY | os.O_DIRECT)
    buf = np.zeros(blk + 4096, dtype=np.uint8)
    off = (-buf.ctypes.data) % 4096
    mv = memoryview(buf)[off:off + blk]
    t0 = time.perf_counter()
    n = 0
    try:
        while n < gib * 2**30:
            got = os.preadv(fd, [mv], n)
            if got <= 0:
                break
            n += got
    finally:
        os.close(fd)
        os.unlink(fn)
    return n / (time.perf_counter() - t0)


def make_view(tier, index):
    """The CPU kernel's contiguous, kernel-shaped stacks: ColdCpuView's copy
    path (the strided-read shortcut violated gemv_mxfp4_grouped_cpu's
    contiguity contract — the shipped view exists precisely to satisfy it),
    with the stacks re-seated as PINNED tensors before any materialization:
    segment_into writes through self.stacks[suffix] at fill time, so the
    swap is transparent to the artifact, host memcpys land in pinned memory,
    and VRAM fills from these rows stay genuinely async."""
    view = ColdCpuView(tier, index, (GATE_UP_BLOCKS, GATE_UP_SCALES))
    if torch.cuda.is_available():
        for suf in view.segments:
            view.stacks[suf] = view.stacks[suf].pin_memory()
    return view


class Prefetcher:
    """Prefetch through the VIEW, not the bare tier: a tier-resident row
    whose materialization memcpys still run inside the step would move the
    landing cost back into the wall. The view is single-controller like the
    tier; join-before-next-use keeps it single-user."""

    def __init__(self, view):
        self.view = view
        self.th = None

    def spawn(self, routed):
        def work():
            for lay, ex in routed:
                self.view.ensure(lay, ex)
        self.th = threading.Thread(target=work)
        self.th.start()

    def join(self):
        if self.th is not None:
            self.th.join()
            self.th = None


def run_arm(recs, view, tier, threads, prefetch):
    k = 4
    m = L_LAYERS * k
    pairs = L_LAYERS * E_EXPERTS
    pool = SegmentedRowPool(-(-int(0.7 * pairs) // SEG_ROWS), SEG_ROWS,
                            ROWBYTES_VRAM, device="cuda", routed=k)
    side = torch.cuda.Stream()
    smooth_cap = -(-m // 4)
    a32 = torch.randn(m, K1, dtype=torch.float32)
    a16 = a32.to(torch.bfloat16).cuda()
    stage = torch.empty(SEG_ROWS, dtype=torch.int32).pin_memory()
    blocks_stack = view.stacks[GATE_UP_BLOCKS]
    scales_stack = view.stacks[GATE_UP_SCALES]

    def routed_of(r):
        return [(int(lay), [e for e in ex])
                for lay, ex in sorted(r["routed"].items(), key=lambda kv: int(kv[0]))]

    # full untimed dry step: NVMe read + fill + gemm + CPU (warm-up inventory)
    dry = routed_of(recs[0])
    fills_before = tier.reader.reads if hasattr(tier, "reader") else 0
    for lay, ex in dry:
        slots = view.ensure(lay, ex)
        tag = StepTag("cuda")
        placed, need, _ = pool.want(lay, ex, tag, budget=len(ex))
        with torch.cuda.stream(side):
            for key in need:
                si, slot = placed[key]
                b, s_ = pool.views(si, (N1, HALF1), (N1, NB1), PB)
                b[slot].copy_(blocks_stack[slots[ex.index(key[1])]],
                              non_blocking=True)
                s_[slot].copy_(scales_stack[slots[ex.index(key[1])]],
                               non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(side)
        torch.cuda.current_stream().wait_event(ev)
        if need:
            si0, sl0 = placed[need[0]]
            b, s_ = pool.views(si0, (N1, HALF1), (N1, NB1), PB)
            gemm_mxfp4_grouped(a16[:1], b, s_, [1], [sl0])
        cg.gemv_mxfp4_grouped_cpu(a32[:len(ex)], blocks_stack, scales_stack,
                                  [1] * len(ex), slots, threads=threads)
        torch.cuda.synchronize()
        tag.record()

    pf = Prefetcher(view)
    walls, cpu_rows_s, gpu_rows_s, nvme_bytes_s = [], [], [], []
    row_disk = tier.row_stride
    # per-step NVMe attribution blurs one step under prefetch (t's counter
    # window includes the tail of t+1's background reads); the gate uses the
    # 32-step median, where the blur cancels
    for t in range(TOTAL):
        r = routed_of(recs[t])
        nxt = routed_of(recs[t + 1]) if t + 1 < len(recs) else []
        pf.join()
        reads_before = tier.stats()["disk_reads"]
        t0 = time.perf_counter()
        slots_by_layer = {}
        for lay, ex in r:
            slots_by_layer[lay] = view.ensure(lay, ex)
        if prefetch and nxt:
            pf.spawn(nxt)
        gpu_by_seg = {}
        cpu_rows = 0
        budget = smooth_cap
        for lay, ex in r:
            slots = slots_by_layer[lay]
            tag = StepTag("cuda")
            placed, need, skipped = pool.want(lay, ex, tag, budget=budget)
            budget -= len(need)
            with torch.cuda.stream(side):
                for key in need:
                    si, slot = placed[key]
                    b, s_ = pool.views(si, (N1, HALF1), (N1, NB1), PB)
                    src = slots[ex.index(key[1])]
                    b[slot].copy_(blocks_stack[src], non_blocking=True)
                    s_[slot].copy_(scales_stack[src], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(side)
            torch.cuda.current_stream().wait_event(ev)
            need_or_skip = set(need) | set(skipped)
            cold_local = []
            for i, e in enumerate(ex):
                key = (lay, e)
                if key in need_or_skip:
                    cold_local.append(slots[i])
                else:
                    si, slot = placed[key]
                    gpu_by_seg.setdefault(si, []).append((lay, i, slot))
            if cold_local:
                cg.gemv_mxfp4_grouped_cpu(
                    a32[:len(cold_local)], blocks_stack, scales_stack,
                    [1] * len(cold_local), cold_local, threads=threads)
                cpu_rows += len(cold_local)
            tag.record()
        gpu_rows = 0
        for si, items in gpu_by_seg.items():
            b, s_ = pool.views(si, (N1, HALF1), (N1, NB1), PB)
            gemm_mxfp4_grouped(a16[:len(items)].contiguous(), b, s_,
                               [1] * len(items), [sl for _, _, sl in items])
            gpu_rows += len(items)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        cpu_rows_s.append(cpu_rows)
        gpu_rows_s.append(gpu_rows)
        nvme_bytes_s.append((tier.stats()["disk_reads"] - reads_before)
                            * row_disk)
    pf.join()
    return {"walls": walls, "cpu_rows": cpu_rows_s, "gpu_rows": gpu_rows_s,
            "nvme_bytes": nvme_bytes_s, "tier_stats": tier.stats()}


class KeepWarm:
    """Sustains boost clocks on lazy-ramp hosts (PREREG-p2-g4p): a tiny
    matmul on its own stream every ~2 ms from a daemon thread — ~0.1%
    occupancy, no privileges, defeats the down-ramp G4 measured (SM parked
    at 180 MHz of 3,090 for the decode launch pattern)."""

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.a = torch.randn(64, 64, device="cuda")
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            with torch.cuda.stream(self.stream):
                torch.matmul(self.a, self.a)
            time.sleep(0.002)

    def stop(self):
        self._stop.set()
        self.th.join()


def clock_ladder(mode):
    """Returns the mode that actually engaged: 'lock', 'keepwarm', 'none'.
    The keep-warm object (if any) is returned so it stays alive."""
    import subprocess
    if mode in ("auto", "lock"):
        try:
            mx = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.max.sm",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout.split()[0]
            r1 = subprocess.run(["nvidia-smi", "-pm", "1"],
                                capture_output=True, timeout=20)
            r2 = subprocess.run(["nvidia-smi", "-lgc", mx],
                                capture_output=True, timeout=20)
            if r2.returncode == 0:
                return "lock", None
        except Exception:
            pass
        if mode == "lock":
            sys.exit("clock lock requested but unavailable on this box")
    if mode in ("auto", "keepwarm"):
        return "keepwarm", KeepWarm()
    return "none", None


def burst_gate():
    """The G4-collapse pattern as a box gate — self-calibrating: the
    gap-pattern per-launch wall is compared against the SAME probe with no
    gaps, so sync-wake latency (which dominates a tiny launch after a 2 ms
    idle and varies by driver wait-mode) cancels out of the signal. A
    ramping host's ratio is ~1; the G4 lazy host's decode rate was ~13x its
    sustained rate. The first registration used an absolute 50 us bar and
    rejected a demonstrably healthy host at 50.5 (matmul20 188 ms) —
    a screen defect, corrected pre-measurement and disclosed."""
    a = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        a = (a @ a).clamp(-1, 1)
    torch.cuda.synchronize()
    mm = time.perf_counter() - t0
    b = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    torch.matmul(b, w)
    torch.cuda.synchronize()

    def probe(gap_s):
        ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            torch.matmul(b, w)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            if gap_s:
                time.sleep(gap_s)
        return statistics.median(ts)

    nogap = probe(0.0)
    gap = probe(0.002)
    ratio = gap / nogap
    print("burst gate: matmul20 %.0f ms  no-gap %.1f us  gap %.1f us  "
          "ratio %.2f" % (mm * 1e3, nogap * 1e6, gap * 1e6, ratio))
    return mm, ratio


def solo_rates(view, tier, threads):
    blocks_stack = view.stacks[GATE_UP_BLOCKS]
    scales_stack = view.stacks[GATE_UP_SCALES]
    a32 = torch.randn(64, K1, dtype=torch.float32)
    slots = list(range(64))
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        cg.gemv_mxfp4_grouped_cpu(a32, blocks_stack, scales_stack,
                                  [1] * 64, slots, threads=threads)
        ts.append((time.perf_counter() - t0) / 64)
    t_cpu_row = statistics.median(ts)
    pool = SegmentedRowPool(1, SEG_ROWS, ROWBYTES_VRAM, device="cuda",
                            routed=4)
    tag = StepTag("cuda")
    placed, need, _ = pool.want(0, list(range(48)), tag, budget=48)
    b, s_ = pool.views(0, (N1, HALF1), (N1, NB1), PB)
    for key in need:
        _si, slot = placed[key]
        b[slot].copy_(blocks_stack[slot % tier.hot_rows])
        s_[slot].copy_(scales_stack[slot % tier.hot_rows])
    torch.cuda.synchronize()
    tag.record()
    a16 = torch.randn(48, K1, dtype=torch.bfloat16, device="cuda")
    sl = [placed[(0, e)][1] for e in range(48)]
    for _ in range(5):
        gemm_mxfp4_grouped(a16, b, s_, [1] * 48, sl)
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        gemm_mxfp4_grouped(a16, b, s_, [1] * 48, sl)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / 48)
    return t_cpu_row, statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--workdir", default="/root/g4")
    ap.add_argument("--clock-mode", default="auto",
                    choices=("auto", "lock", "keepwarm", "none"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    gnf4_native.load()
    torch.set_num_threads(1)
    cal = json.load(open(a.calib))
    if not (2 <= cal["nstar_direct"] <= 5):
        sys.exit("box fails the n* gate: %.2f" % cal["nstar_direct"])

    clock_mode, warm = clock_ladder(a.clock_mode)
    print("clock mode:", clock_mode)
    mm, ratio = burst_gate()
    if mm > 0.220 or ratio > 3.0:
        if warm is not None:
            warm.stop()             # never sys.exit through a live CUDA
        sys.exit("box fails the burst-clock gate (matmul20 %.0f ms, "  # thread
                 "gap/no-gap ratio %.2f) — lazy-ramp host, reject" %
                 (mm * 1e3, ratio))

    os.makedirs(a.workdir, exist_ok=True)
    arena = os.path.join(a.workdir, "g4.arena")
    b_nvme = nvme_probe(arena)
    print("NVMe O_DIRECT probe: %.2f GB/s" % (b_nvme / 1e9))
    if b_nvme < 1e9:
        sys.exit("box fails the NVMe gate: %.2f GB/s < 1 GB/s" % (b_nvme / 1e9))

    snap = os.path.join(a.workdir, "snap")
    if not os.path.exists(arena):
        print("building snapshot + baking arena...", flush=True)
        build_snapshot(snap)
        bake(snap, arena, align=4096, log=lambda *x: None)
    ground = load_ground(snap)
    index = load_index(arena)
    tier = ColdTier(arena, hot_rows=HOT_ROWS, pinned=True, index=index)
    view = make_view(tier, index)

    meta, recs = load(os.path.join(HERE, "rank-2026-08-22",
                                   "gptoss_code.jsonl"))
    t_cpu_row, t_gpu_row = solo_rates(view, tier, a.threads)
    print("solo rates: t_cpu_row %.1f us  t_gpu_row %.1f us"
          % (t_cpu_row * 1e6, t_gpu_row * 1e6))

    out = {"b_nvme_solo": b_nvme, "t_cpu_row_solo": t_cpu_row,
           "t_gpu_row_solo": t_gpu_row, "hot_rows": HOT_ROWS,
           "clock_mode": clock_mode, "burst_gate_matmul_s": mm,
           "burst_gate_ratio": ratio}
    arms = {}
    for name, prefetch in (("overlap", True), ("sequential", False)):
        res = run_arm(recs, view, tier, a.threads, prefetch)
        arms[name] = res
        med = statistics.median(res["walls"][STEADY_LO:STEADY_HI])
        print("%s: steady median wall %.2f ms" % (name, med * 1e3))
    sl = slice(STEADY_LO, STEADY_HI)
    A, B = arms["overlap"], arms["sequential"]
    t_cpu_alone = statistics.median(
        [r * t_cpu_row for r in A["cpu_rows"][sl]])
    t_gpu_alone = statistics.median(
        [r * t_gpu_row for r in A["gpu_rows"][sl]])
    t_sto_alone = statistics.median(
        [by / b_nvme for by in A["nvme_bytes"][sl]])
    mx = max(t_cpu_alone, t_gpu_alone, t_sto_alone)
    medA = statistics.median(A["walls"][sl])
    medB = statistics.median(B["walls"][sl])
    budget_pred = statistics.median(
        [(A["cpu_rows"][i] * ROWBYTES_VRAM + A["nvme_bytes"][i])
         / (cal["B_cpu_gbs"] * 1e9)
         for i in range(STEADY_LO, STEADY_HI)])
    g4a = medA <= 1.15 * mx
    g4b = medA <= 0.80 * medB
    spoiler_fails = medB > 1.15 * mx
    checks = correctness_checks(view, tier, ground, recs, a.threads)
    void = not all(v for v in checks.values())
    verdict = ("UNINFORMATIVE" if not spoiler_fails else
               "VOID" if void else
               "PASS" if (g4a and g4b) else "REFUTED")
    out.update({"alones": {"t_cpu": t_cpu_alone, "t_gpu": t_gpu_alone,
                           "t_storage": t_sto_alone, "max": mx},
                "median_wall_overlap": medA, "median_wall_sequential": medB,
                "budget_pred_s": budget_pred,
                "g4a_objective": g4a, "g4b_gap": g4b,
                "spoiler_fails": spoiler_fails, "checks": checks,
                "arms": arms, "verdict": verdict})
    print(json.dumps({k: v for k, v in out.items() if k != "arms"},
                     indent=1, default=str))
    print("G4:", verdict)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("receipt ->", a.out)
    if warm is not None:
        warm.stop()                 # end the keep-warm before CUDA teardown


if __name__ == "__main__":
    main()
