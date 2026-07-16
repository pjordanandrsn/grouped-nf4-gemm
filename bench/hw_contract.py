# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hardware contract check — bnb-free kernel correctness on ANY torch device.

The on-silicon twin of the CI interpreter suite: pack NF4 with the pure-torch
reference packer, run the REAL triton kernels on the device, compare against
dequant_ref matmuls. Answers "does the kernel run correctly on this backend?"
without requiring a bitsandbytes build (the ROCm/XPU long pole). The bnb
exactness pins stay the GPU suite's job where bnb exists.

Usage: python bench/hw_contract.py [--device cuda]  (exit 0 = all pass)
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kernel"))

import triton  # noqa: E402
from nf4_grouped import (  # noqa: E402
    BLOCKSIZE, _gemv_nf4_grouped, _gemm_nf4_grouped, _lut, build_group_tiles,
    dequant_ref, gemm_4bit_grouped)
from nf4_pack_ref import make_stack  # noqa: E402


def ref_out(B, A, eids, acts, N, K):
    outs = []
    for g, e in enumerate(eids):
        w = dequant_ref(B[e], A[e], N, K).float()
        outs.append(w @ acts[g].float())
    return torch.stack(outs)


def check(name, out, ref, tol=1e-2):
    # norm-relative error (the suite's b_rel metric) — per-element relative
    # error blows up on near-zero reference cells and mislabels a correct
    # kernel (bit us on first run: row-0 values matched to bf16 precision
    # while a near-zero cell read "88% error").
    err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: b_rel={err:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ok = True

    # decode gemv (several shapes incl. non-multiple-of-BLOCK_N N)
    for (E, N, K, G) in [(4, 128, 128, 3), (8, 384, 256, 8), (2, 100, 192, 2)]:
        B, A = make_stack(E, N, K, seed=E + N, device=dev)
        eids = torch.arange(G, dtype=torch.int32, device=dev) % E
        acts = torch.randn(G, K, dtype=torch.bfloat16, device=dev)
        out = gemm_4bit_grouped(acts, B, A, [1] * G, eids)
        ref = ref_out(B, A, eids.tolist(), acts, N, K)
        ok &= check(f"gemv E{E} N{N} K{K} G{G}", out, ref)

    # M-tile prefill path (multi-row groups) through the wrapper (rule config)
    for (E, N, K, sizes) in [(4, 128, 128, [5, 9]), (4, 256, 256, [64, 32, 16])]:
        B, A = make_stack(E, N, K, seed=N, device=dev)
        eids = list(range(len(sizes)))
        acts = torch.randn(sum(sizes), K, dtype=torch.bfloat16, device=dev)
        out = gemm_4bit_grouped(acts, B, A, sizes, eids)
        refs, i = [], 0
        for g, e in enumerate(eids):
            w = dequant_ref(B[e], A[e], N, K).float()
            refs.append(acts[i:i + sizes[g]].float() @ w.t()); i += sizes[g]
        ref = torch.cat(refs)
        ok &= check(f"mtile E{E} N{N} K{K} sizes{sizes}", out, ref)

    print("HW-CONTRACT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
