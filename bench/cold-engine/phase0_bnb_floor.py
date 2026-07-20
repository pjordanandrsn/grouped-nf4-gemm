# Phase-0 premise test 2b: bitsandbytes CPU dequantize_4bit (AVX512 path,
# standard packed layout, IN PLACE — the free floor arm found in the 0.3
# scout). Same shapes/accounting as phase0_naive_floor.py.
import argparse
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--n", type=int, default=5760)
    ap.add_argument("--k", type=int, default=2880)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--gemv", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(args.threads)

    import bitsandbytes  # noqa: F401  (registers the CPU backend kernels)
    import bitsandbytes.functional as F_bnb

    E, n, k = args.experts, args.n, args.k
    # quantize a real stack once so absmax/layout are genuine bnb artifacts
    w = torch.randn(n, k, dtype=torch.bfloat16)
    packed_1, state = F_bnb.quantize_4bit(
        w, blocksize=64, compress_statistics=False, quant_type="nf4")
    packed = packed_1.reshape(-1)
    absmax = state.absmax.reshape(-1)
    x = torch.randn(1, k, dtype=torch.bfloat16)

    def dq():
        qs = F_bnb.QuantState(
            absmax=absmax, shape=torch.Size((n, k)),
            code=F_bnb.get_4bit_type("nf4", device="cpu"), blocksize=64,
            quant_type="nf4", dtype=torch.bfloat16)
        return F_bnb.dequantize_4bit(packed.reshape(-1, 1), quant_state=qs)

    wd = dq()
    if args.gemv:
        _ = x @ wd.T
    t0 = time.perf_counter()
    for _ in range(E):
        wd = dq()
        if args.gemv:
            _ = x @ wd.T
    dt = time.perf_counter() - t0

    packed_bytes = E * (packed.numel() + absmax.numel() * 4)
    label = "bnb-dequant+gemv" if args.gemv else "bnb-dequant-only"
    per = dt / E * 1e3
    gbs = packed_bytes / dt / 1e9
    print(f"{label}: {per:.2f} ms/expert ({E}x [{n},{k}]), packed-bytes {gbs:.2f} GB/s, "
          f"threads={args.threads}")
    result = dict(mode=label, experts=E, n=n, k=k, threads=args.threads,
                  ms_per_expert=round(per, 3), packed_gbs=round(gbs, 3),
                  bnb=__import__("bitsandbytes").__version__)
    print(json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
