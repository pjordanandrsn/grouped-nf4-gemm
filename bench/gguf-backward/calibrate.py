# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Calibrate the GGUF dX bars BEFORE any path under test exists: the control's error (the floor a correct bf16 path
sits at) and each deliberate fault's error (what a wrong path scores), both against an independent fp64 oracle.

CPU only. dX = dY @ W, W the dequantized [N, K] frozen weight. Oracle: gguf-py's own dequantize, cast to fp64.
Path under test (here, the control and its mutants): gnf4's kquant_ref decode, bf16 W, fp32 accumulate, bf16 out.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import gguf.quants as gq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernel"))
import kquant_ref as kq  # noqa: E402

T = {"Q8_0": kq.GGML_Q8_0, "Q4_K": kq.GGML_Q4_K, "Q5_K": kq.GGML_Q5_K, "Q6_K": kq.GGML_Q6_K}
ORACLE = {kq.GGML_Q8_0: gq.Q8_0, kq.GGML_Q4_K: gq.Q4_K, kq.GGML_Q5_K: gq.Q5_K, kq.GGML_Q6_K: gq.Q6_K}
FP16_COLS = {kq.GGML_Q8_0: [(0, 2)], kq.GGML_Q4_K: [(0, 2), (2, 4)], kq.GGML_Q5_K: [(0, 2), (2, 4)],
             kq.GGML_Q6_K: [(208, 210)]}
SHAPES = [(17, 37, 256), (128, 256, 256), (3, 129, 2816), (64, 2048, 512)]   # (M, N, K); the 2nd has N == K


def blocks_for(gtype, n_rows, k, seed):
    """Random packed blocks with finite fp16 scale fields (the test_kquant_ref pattern), scaled so |W| ~ 1/sqrt(K)."""
    elems, bbytes, _ = kq.GGML_DEQUANT[gtype]
    n = n_rows * (k // elems)
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(n, bbytes), dtype=np.uint8)
    mag = 1.0 / np.sqrt(k) / (16.0 if gtype != kq.GGML_Q8_0 else 64.0)
    finite = (rng.uniform(0.5, 1.5, size=(n,)) * mag).astype(np.float16)
    for lo, hi in FP16_COLS[gtype]:
        b[:, lo:hi] = finite.view(np.uint8).reshape(n, 2)
        finite = finite * np.float16(0.5)
    return b


def oracle_w(gtype, b, n_rows, k):
    return torch.from_numpy(ORACLE[gtype].dequantize_blocks(b).astype(np.float64)).reshape(n_rows, k)


def decode(gtype, b, n_rows, k):
    return kq.dequantize_ggml(gtype, b.tobytes(), (n_rows, k))


def dx_bf16(dy, w32):
    return (dy.float() @ w32.to(torch.bfloat16).float()).to(torch.bfloat16).double()


def rel(a, ref):
    return float(torch.linalg.norm(a - ref) / torch.linalg.norm(ref))


# -- the deliberate faults: each changes ONLY the path under test, never the oracle ------------------------------
def m1_scale_spill(gtype, b, n_rows, k):                       # Q4_K/Q5_K: high 2 bits of sc/m[4..7] dropped
    orig = kq._get_scale_min_k4

    def broken(scales):
        s = scales.reshape(-1, 3, 4)
        d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
        return torch.cat([d & 0x3F, m_d & 0x0F], -1), torch.cat([m & 0x3F, m_d >> 4], -1)
    kq._get_scale_min_k4 = broken
    try:
        return decode(gtype, b, n_rows, k)
    finally:
        kq._get_scale_min_k4 = orig


def m5_dmin_zero(gtype, b, n_rows, k):                         # Q4_K/Q5_K: the min term dropped
    b2 = b.copy()
    b2[:, 2:4] = 0
    return decode(gtype, b2, n_rows, k)


def m6_signed_as_unsigned(gtype, b, n_rows, k):                # Q8_0 qs / Q6_K scales read as uint8, not int8
    elems, bbytes, _ = kq.GGML_DEQUANT[gtype]
    if gtype == kq.GGML_Q8_0:
        d = torch.from_numpy(b[:, 0:2].copy().view(np.float16).astype(np.float32))
        q = torch.from_numpy(b[:, 2:34].astype(np.float32))      # uint8, wrong
        return (d * q).reshape(n_rows, k)
    b2 = b.copy()                                               # Q6_K: clamp int8 scales to their absolute value
    sc = b2[:, 192:208].view(np.int8)
    b2[:, 192:208] = np.abs(sc.astype(np.int16)).clip(0, 127).astype(np.uint8)
    return decode(gtype, b2, n_rows, k)


def main():
    out = []
    for name, gtype in T.items():
        for si, (M, N, K) in enumerate(SHAPES):
            for seed in range(3):
                s = gtype * 1000 + si * 10 + seed
                b = blocks_for(gtype, N, K, s)
                w64 = oracle_w(gtype, b, N, K)
                w32 = decode(gtype, b, N, K)
                assert torch.equal(w32.double(), w64), f"{name}: decoder disagrees with gguf-py (the bit-exact link)"
                g = torch.Generator().manual_seed(s + 7)
                dy = torch.randn(M, N, generator=g).to(torch.bfloat16)
                ref = dy.double() @ w64
                row = {"type": name, "M": M, "N": N, "K": K, "seed": seed,
                       "adjoint_rel": None, "control": rel(dx_bf16(dy, w32), ref),
                       "control_fp32_out": rel((dy.float() @ w32.to(torch.bfloat16).float()).double(), ref),
                       "scales_bf16_calibration": None}
                x = torch.randn(M, K, generator=g).to(torch.bfloat16).double()
                lhs, rhs = float((dy.double() * (x @ w64.T)).sum()), float(((dy.double() @ w64) * x).sum())
                row["adjoint_rel"] = abs(lhs - rhs) / max(abs(lhs), 1e-300)
                faults = {"zero_dX": torch.zeros_like(ref),
                          "independent_W": dx_bf16(dy, decode(gtype, blocks_for(gtype, N, K, s + 99991), N, K)),
                          "m4_row_shift": dx_bf16(dy, torch.roll(w32, 1, 0))}
                if N == K:
                    faults["m3_transpose"] = dx_bf16(dy, w32.T.contiguous())
                if gtype in (kq.GGML_Q4_K, kq.GGML_Q5_K):
                    faults["m1_scale_spill"] = dx_bf16(dy, m1_scale_spill(gtype, b, N, K))
                    faults["m5_dmin_zero"] = dx_bf16(dy, m5_dmin_zero(gtype, b, N, K))
                if gtype in (kq.GGML_Q8_0, kq.GGML_Q6_K):
                    faults["m6_signed_as_unsigned"] = dx_bf16(dy, m6_signed_as_unsigned(gtype, b, N, K))
                row["faults"] = {k_: rel(v, ref) for k_, v in faults.items()}
                out.append(row)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    # summary: per type, the control's max and each fault's min over shapes and seeds
    for name in T:
        rows = [r for r in out if r["type"] == name]
        fnames = sorted({f for r in rows for f in r["faults"]})
        print(name, "control max %.2e (fp32-out %.2e) | adjoint max %.1e |" % (
            max(r["control"] for r in rows), max(r["control_fp32_out"] for r in rows),
            max(r["adjoint_rel"] for r in rows)),
            " ".join("%s min %.2e" % (f, min(r["faults"][f] for r in rows if f in r["faults"])) for f in fnames))


if __name__ == "__main__":
    main()
