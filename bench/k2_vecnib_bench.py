# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k2 on-box instrument: at the K1 winner configs, (1) assert
BITWISE equality legacy-vs-vectorized on real CUDA codegen (the hard
G-B gate -- interp CI only guards the algebra), (2) time both paths
with the registered chunked-median estimator. JSON out."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
import nf4_grouped  # noqa: E402

# the K1 winners (RESULTS-m1-decode-config): shape -> (bn, warps, sk)
CELLS = (("gate_up", 1536, 2048, 8, (64, 2), 16),
         ("down", 2048, 768, 8, (32, 2), 1))


def _mk(N, K, T, E=8, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8,
                           generator=g).to(device)
    absmax = (torch.rand(E, N, K // 64, generator=g) + 0.5).to(device)
    a = torch.randn(T, K, generator=g).to(device=device,
                                          dtype=torch.bfloat16)
    eids = torch.arange(E, dtype=torch.int32, device=device)[:T]
    return a, packed, absmax, [1] * T, eids


def _time(fn, iters=200, warmup=50, chunks=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = max(1, iters // chunks)
    spans = []
    for _ in range(chunks):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(per):
            fn()
        e1.record()
        e1.synchronize()
        spans.append(e0.elapsed_time(e1) / per)
    spans.sort()
    return spans[len(spans) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="k2_bench.json")
    args = ap.parse_args()
    assert nf4_grouped.HAS_TL_INTERLEAVE, "this triton lacks tl.interleave"
    rep = {"gpu": torch.cuda.get_device_name(0), "cells": {}}
    legacy_sum = vec_sum = 0.0
    bitwise_all = True
    for name, N, K, T, cfg, sk in CELLS:
        a, p, ax, sizes, eids = _mk(N, K, T)

        def call():
            return nf4_grouped.gemm_4bit_grouped(
                a, p, ax, sizes, eids, decode_config=cfg, split_k=sk)

        os.environ["GNF4_GEMV_SCALAR_LOADS"] = "1"
        ref = call()
        t_legacy = _time(call)
        del os.environ["GNF4_GEMV_SCALAR_LOADS"]
        vec = call()
        t_vec = _time(call)
        bitwise = bool(torch.equal(ref, vec))
        bitwise_all &= bitwise
        rep["cells"][name] = {"config": list(cfg) + [sk],
                              "legacy_ms": t_legacy, "vec_ms": t_vec,
                              "bitwise": bitwise}
        legacy_sum += t_legacy
        vec_sum += t_vec
    rep["summary"] = {"legacy_sum_ms": legacy_sum, "vec_sum_ms": vec_sum,
                      "ratio_vec_over_legacy": vec_sum / legacy_sum,
                      "bitwise_all": bitwise_all}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    s = rep["summary"]
    print(f"K2BENCH legacy={legacy_sum*1000:.1f}us vec={vec_sum*1000:.1f}us "
          f"ratio={s['ratio_vec_over_legacy']:.3f} "
          f"bitwise={'PASS' if bitwise_all else 'FAIL'}")


if __name__ == "__main__":
    main()
