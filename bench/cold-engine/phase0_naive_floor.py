# Phase-0 premise test 2: the naive CPU floor — pure-torch NF4 dequant (+ GEMV)
# over one expert-layer's worth of packed bytes, IN PLACE (arena layout, no
# repack — the design law). This is the A-stock of the cold lane; the Phase-2
# microkernel is graded as a fraction of the memcpy ceiling, and must beat this.
#
# Decode math = kernel/nf4_pack_ref.py semantics (LUT nibble decode x absmax);
# uses bnb's LUT ordering via the local pack_ref import when run from the repo
# kernel dir, else a self-contained copy of the NF4 table.
#
# Usage: python3 phase0_naive_floor.py [--experts 32] [--n 5760] [--k 2880]
#        [--threads 6] [--gemv]
import argparse
import json
import os
import time

import torch

# bnb NF4 codebook (bitsandbytes get_4bit_type("nf4"); nibble 2j = HIGH for bnb)
NF4_LUT = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
           -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
           0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
           0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
           0.7229568362236023, 1.0]
BLOCK = 64  # bnb blocksize used by the arena


def dequant_nf4_rows(packed_u8: torch.Tensor, absmax: torch.Tensor,
                     n: int, k: int, dtype=torch.bfloat16) -> torch.Tensor:
    """packed [n*k//2] u8 (+ absmax [n*k//64] fp32) -> [n, k] dtype. bnb order:
    element 2j = HIGH nibble."""
    lut = torch.tensor(NF4_LUT, dtype=dtype)
    hi = (packed_u8 >> 4).long()
    lo = (packed_u8 & 0xF).long()
    vals = torch.stack([lut[hi], lut[lo]], dim=-1).reshape(-1)      # [n*k]
    vals = vals.reshape(-1, BLOCK) * absmax[:, None].to(dtype)
    return vals.reshape(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--n", type=int, default=5760)
    ap.add_argument("--k", type=int, default=2880)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--gemv", action="store_true", help="also time dequant+GEMV")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(args.threads)

    E, n, k = args.experts, args.n, args.k
    packed = torch.randint(0, 256, (E, n * k // 2), dtype=torch.uint8)
    absmax = torch.rand(E, n * k // BLOCK, dtype=torch.float32) + 0.5
    x = torch.randn(1, k, dtype=torch.bfloat16)

    # warm one expert
    w = dequant_nf4_rows(packed[0], absmax[0], n, k)
    if args.gemv:
        _ = x @ w.T

    t0 = time.perf_counter()
    for e in range(E):
        w = dequant_nf4_rows(packed[e], absmax[e], n, k)
        if args.gemv:
            _ = x @ w.T
    dt = time.perf_counter() - t0

    packed_bytes = E * (n * k // 2 + absmax.shape[1] * 4)
    per_expert_ms = dt / E * 1e3
    eff_gbs = packed_bytes / dt / 1e9
    label = "dequant+gemv" if args.gemv else "dequant-only"
    print(f"{label}: {per_expert_ms:.2f} ms/expert ({E}x [{n},{k}]), "
          f"packed-bytes {eff_gbs:.2f} GB/s, threads={args.threads}")
    result = dict(mode=label, experts=E, n=n, k=k, threads=args.threads,
                  ms_per_expert=round(per_expert_ms, 3),
                  packed_gbs=round(eff_gbs, 3), torch=torch.__version__)
    print(json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
