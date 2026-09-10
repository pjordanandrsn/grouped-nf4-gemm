"""Find the true sk optimum as a function of R, so _plan can be fitted not guessed.

Usage: ``python sk_sweep.py <family> <proj>`` on a CUDA box, one shape per
invocation -- see the process-isolation note at the end of this docstring.
Writes ``rows/sk_<family>_<proj>.json``, which kernel/test_int4_b32.py reads
as the receipt behind SPLITK_TARGET_BLOCKS_PER_SM.

`_plan(N, K)` takes no R. Its docstring says sk exists to "fill the grid to
2+ waves" and the sk rule is `8 if cdiv(N,128)*8 >= 256 else 16` -- i.e. a
target of 256 blocks, which is 2 waves of a 128-SM sm_120 part. But the launch
grid is `(cdiv(N,BLOCK_N), R, sk)`, so the blocks actually resident are
tiles*R*sk. R is a factor of the very quantity the rule is trying to control,
and it is absent from the rule.

The first A/B compared only sk=1 against sk=plan, which cannot see an interior
optimum. It reported "sk=1 wins at every R>=16"; this sweep exists to check
that, and the interior turns out to matter (sk=4 beats both at several cells).
Hence: sweep every sk up to the cap, including the planned one.

Measurement follows int4_b32's own two load-bearing rules: graph replay, never
eager; and configs compared under that metric, because an eager sweep
anti-selects split-K (the reduce pays a launch the replay does not). Every cell
reports the FULL cost including the reduce, since sk=1 needs no reduce at all
and a kernel-only number would flatter the split arms.

One (family, proj) per PROCESS: the first attempt hit an illegal memory access
whose fatal traceback landed on the NEXT shape's manual_seed -- async faults
surface late, so a shared process makes one shape's fault look like another's
result.
"""
import json
import os
import pathlib
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "kernel"))
from int4_b32 import _gemv_int4_b32, _plan, reduce_partials  # noqa: E402
from int4_pack_ref import pack_int4_b32  # noqa: E402

DEV = "cuda"
FAMILIES = {
    "qwen3_moe": (2048, 768),
    "granitemoe": (1536, 512),
    "olmoe": (2048, 1024),
}
RS = [int(x) for x in os.environ.get("RS", "1,2,4,8,16,32,64,128").split(",")]
SKS = [int(x) for x in os.environ.get("SKS", "1,2,3,4,6,8,16").split(",")]
E = 128


def timed_replay(fn, iters=60, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    name, proj = sys.argv[1], sys.argv[2]
    H, I = FAMILIES[name]
    N, K = (2 * I, H) if proj == "gate_up" else (H, I)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    gpu = torch.cuda.get_device_name(0)

    torch.manual_seed(0)
    W = torch.randn(E, N, K, dtype=torch.bfloat16, device=DEV)
    ps = [pack_int4_b32(W[e]) for e in range(E)]
    packed = torch.stack([a for a, _ in ps]).contiguous()
    scales = torch.stack([b for _, b in ps]).contiguous()
    del W, ps
    bn, wp, sk_plan, ku = _plan(N, K)
    kb = K // 32
    sk_cap = max(1, kb // ku)
    tiles = triton.cdiv(N, bn)
    # the planned sk must be in the swept set or its cell reads NaN
    sks = sorted({s for s in SKS + [sk_plan] if s <= sk_cap})
    print(f"## {name} {proj}  N={N} K={K}  plan sk={sk_plan} (cap {sk_cap}), "
          f"tiles={tiles}, {gpu} {sms} SMs", flush=True)

    rows = []
    for R in RS:
        xq = torch.randint(-127, 127, (R, K), dtype=torch.int8, device=DEV)
        # per-(row, 32-block) activation scales, [R, K//32] -- what quant_x_rows produces and
        # what the kernel indexes (xs_ptr + e*KB + kb0 + ku). The first cut of this harness
        # passed [R, 1]: the kernel then read R*(KB-1) floats past the buffer -- harmless
        # garbage on small shapes, an illegal memory access on the largest, which the
        # first RESULTS write-up mis-attributed to graph pools. Timings are data-independent
        # (same loads, same MACs) and were re-measured with this shape to confirm.
        xs = torch.rand(R, K // 32, dtype=torch.float32, device=DEV) * 0.01
        eids = torch.randint(0, E, (R,), dtype=torch.int32, device=DEV)
        times = {}
        for sk in sks:
            part = torch.empty(sk * R, N, dtype=torch.float32, device=DEV)
            dst = torch.empty(R, N, dtype=torch.bfloat16, device=DEV)

            def call(sk=sk, part=part, dst=dst):
                _gemv_int4_b32[(tiles, R, sk)](
                    xq, xs, packed, scales, eids, part,
                    N, K=K, R=R, BLOCK_N=bn, SK=sk, KU=ku, num_warps=wp)
                if sk > 1:
                    reduce_partials(part, sk, R, N, out=dst)
            try:
                times[sk] = timed_replay(call)
            except Exception as e:                          # noqa: BLE001
                print(f"    sk={sk} R={R} FAILED {type(e).__name__}: {e}", flush=True)
                raise                                       # never keep timing a poisoned context
        best = min(times, key=times.get)
        pen = times[sk_plan] / times[best]
        cells = " ".join(f"sk{sk}={times[sk] * 1000:.1f}" for sk in sks)
        print(f"   R={R:4d} | {cells} us | BEST sk={best}"
              + ("  = plan" if best == sk_plan else f"  (plan sk={sk_plan} costs x{pen:.3f})"),
              flush=True)
        rows.append({"family": name, "proj": proj, "N": N, "K": K, "R": R,
                     "sk_plan": sk_plan, "sk_best": best, "sk_cap": sk_cap,
                     "tiles": tiles, "sms": sms, "gpu": gpu,
                     "ms": {str(k): v for k, v in times.items()},
                     "plan_penalty": pen})
    out = pathlib.Path(__file__).resolve().parent / "rows"
    out.mkdir(exist_ok=True)
    with (out / f"{os.environ.get('OUT_PREFIX', 'sk')}_{name}_{proj}.json").open("w") as f:
        json.dump({"gpu": gpu, "sms": sms, "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
