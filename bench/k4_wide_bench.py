# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k4 on-box instrument: at given configs, (1) assert BITWISE
equality legacy-vs-WIDE on real CUDA codegen, (2) time both paths with
the registered chunked-median estimator. Configs default to the K1
winners; pass --configs to use the wide-sweep winners."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
import nf4_grouped  # noqa: E402

DEFAULT_CELLS = (("gate_up", 1536, 2048, 8, (64, 2), 16),
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
    ap.add_argument("--out", default="k4_bench.json")
    ap.add_argument("--configs", default=None,
                    help="name=bn,warps,sk;... overrides per cell")
    args = ap.parse_args()
    cells = {c[0]: c for c in DEFAULT_CELLS}
    if args.configs:
        for entry in args.configs.split(";"):
            name, _, cfg = entry.partition("=")
            bn, w, sk = (int(x) for x in cfg.split(","))
            n0, N, K, T, _, _ = cells[name]
            cells[name] = (n0, N, K, T, (bn, w), sk)
    rep = {"gpu": torch.cuda.get_device_name(0), "cells": {}}
    legacy_sum = wide_sum = 0.0
    bitwise_all = True
    for name, N, K, T, cfg, sk in cells.values():
        a, p, ax, sizes, eids = _mk(N, K, T)

        def call():
            return nf4_grouped.gemm_4bit_grouped(
                a, p, ax, sizes, eids, decode_config=cfg, split_k=sk)

        os.environ.pop("GNF4_GEMV_WIDE_LOADS", None)
        ref = call()
        t_legacy = _time(call)
        os.environ["GNF4_GEMV_WIDE_LOADS"] = "1"
        wide = call()
        t_wide = _time(call)
        os.environ.pop("GNF4_GEMV_WIDE_LOADS", None)
        bitwise = bool(torch.equal(ref, wide))
        bitwise_all &= bitwise
        rep["cells"][name] = {"config": list(cfg) + [sk],
                              "legacy_us": t_legacy * 1000.0,
                              "wide_us": t_wide * 1000.0,
                              "bitwise": bitwise}
        legacy_sum += t_legacy * 1000.0
        wide_sum += t_wide * 1000.0
    rep["summary"] = {"legacy_sum_us": legacy_sum,
                      "wide_sum_us": wide_sum,
                      "bitwise_all": bitwise_all}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"K4BENCH legacy={legacy_sum:.1f}us wide={wide_sum:.1f}us "
          f"bitwise={'PASS' if bitwise_all else 'FAIL'}")


if __name__ == "__main__":
    main()
