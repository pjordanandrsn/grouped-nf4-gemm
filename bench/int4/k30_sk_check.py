"""K30's correctness gate (kernel/PREREG-k30-splitk-r-term-l4.md), run on the card BEFORE any timing.

For every shape `sk_sweep.py` times, at R = 16 and R = 128, every sk the sweep will time must produce what sk = 1
produces, through the served two-launch path (`reduce_partials` at every sk, at sk = 1 as the cast), to within
split-K's fp32 reorder: max |out_sk - out_1| <= 1e-2 * max |out_1|. A
timing of a configuration that computes something else is not a timing of the plan. Exits 22 on any failure, so the
runner produces no perf number. Uses the installed int4_b32, exactly as the sweep does.

    python k30_sk_check.py
"""
import sys

import torch
import triton

from int4_b32 import _gemv_int4_b32, _plan, reduce_partials
from int4_pack_ref import pack_int4_b32

FAMILIES = {"qwen3_moe": (2048, 768), "granitemoe": (1536, 512), "olmoe": (2048, 1024)}
SKS = (1, 2, 3, 4, 6, 8, 16)
E, TOL = 16, 1e-2


def main() -> int:
    dev = "cuda"
    bad = 0
    for name, (hidden, inter) in FAMILIES.items():
        for proj in ("gate_up", "down"):
            N, K = (2 * inter, hidden) if proj == "gate_up" else (hidden, inter)
            torch.manual_seed(0)
            W = torch.randn(E, N, K, dtype=torch.bfloat16, device=dev)
            ps = [pack_int4_b32(W[e]) for e in range(E)]
            packed = torch.stack([a for a, _ in ps]).contiguous()
            scales = torch.stack([b for _, b in ps]).contiguous()
            del W, ps
            bn, wp, sk_plan, ku = _plan(N, K)
            cap = max(1, (K // 32) // ku)
            sks = sorted({s for s in (*SKS, sk_plan) if s <= cap})
            tiles = triton.cdiv(N, bn)
            for R in (16, 128):
                xq = torch.randint(-127, 127, (R, K), dtype=torch.int8, device=dev)
                xs = torch.rand(R, K // 32, dtype=torch.float32, device=dev) * 0.01
                eids = torch.randint(0, E, (R,), dtype=torch.int32, device=dev)
                outs = {}
                for sk in sks:
                    part = torch.empty(sk * R, N, dtype=torch.float32, device=dev)
                    dst = torch.empty(R, N, dtype=torch.bfloat16, device=dev)
                    _gemv_int4_b32[(tiles, R, sk)](xq, xs, packed, scales, eids, part, part, dst,
                                                   N, K=K, R=R, BLOCK_N=bn, SK=sk, KU=ku, FUSED_REDUCE=0, num_warps=wp)
                    reduce_partials(part, sk, R, N, out=dst)   # at sk = 1 too: the served two-launch path
                    outs[sk] = dst.float()
                torch.cuda.synchronize()
                ref = outs[1]
                scale = ref.abs().max().item()
                worst = max((outs[s] - ref).abs().max().item() / scale for s in sks)
                ok = scale > 0 and worst <= TOL and all(torch.isfinite(o).all() for o in outs.values())
                bad += not ok
                print(f"{'ok ' if ok else 'BAD'} {name} {proj} N={N} K={K} R={R} sks={sks} worst rel {worst:.2e}",
                      flush=True)
    print(f"K30 sk check: {'PASS' if not bad else f'FAIL ({bad} cells)'}")
    return 22 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
