# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Backward correctness through a frozen GGUF k-quant weight: dX = dY @ W against an independent fp64 oracle.

The design is bench/gguf-backward/DESIGN.md; the bars were calibrated BEFORE this path was scored
(bench/gguf-backward/CALIBRATION.md): the correct bf16 path sits at 1.7-2.5e-3 relative error against fp64 (the
floor), and every deliberate fault scores at least 130x the realized bound on released bytes.

  oracle   W64 from gguf-py's own dequantize (independent of kquant_ref), dX_ref = dY64 @ W64
  control  e_ctrl, the error of the bf16 reference path (decode, bf16 W, fp32 accumulate, bf16 out)
  bars     e_test <= min(2 * e_ctrl, 1e-2); the control live (0 < e_ctrl < 1e-2); the adjoint identity to 1e-12
  probe    dY = I returns W itself: the path's decode must equal the reference decode bit for bit
  armed    every fault below must FAIL a bar or the probe with AssertionError (a crash would not count)

The real-byte arm runs on sha-pinned tensors from released GGUF files (scripts/fetch_gguf_fixtures.py): set
GNF4_GGUF_FIXTURES to one or more fixture dirs (os.pathsep-separated).
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

gguf = pytest.importorskip("gguf")
import gguf.quants as gq  # noqa: E402
import numpy as np  # noqa: E402

import kquant_ref as kq  # noqa: E402
from kquant_ref import (  # noqa: E402
    GGML_DEQUANT, GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_TYPE_NAMES,
    dequantize_ggml, kquant_linear,
)

B_REL, B_ABS, ADJOINT = 2.0, 1e-2, 1e-12
ORACLE = {GGML_Q8_0: gq.Q8_0, GGML_Q2_K: gq.Q2_K, GGML_Q3_K: gq.Q3_K, GGML_Q4_K: gq.Q4_K, GGML_Q5_K: gq.Q5_K,
          GGML_Q6_K: gq.Q6_K}
FP16_COLS = {GGML_Q8_0: [(0, 2)], GGML_Q2_K: [(80, 82), (82, 84)], GGML_Q3_K: [(108, 110)],
             GGML_Q4_K: [(0, 2), (2, 4)], GGML_Q5_K: [(0, 2), (2, 4)], GGML_Q6_K: [(208, 210)]}
DMIN_TYPES = (GGML_Q2_K, GGML_Q4_K, GGML_Q5_K)
TYPES = sorted(ORACLE)
SHAPES = [(1, 7, 256), (17, 37, 256), (3, 129, 2816), (16, 256, 256), (128, 1, 512)]   # (M, N, K); one N == K


# -- fixtures --------------------------------------------------------------------------------------------------
def _blocks(gtype, n_rows, k, seed, dmin_ratio=8.0):
    """Random packed blocks (the test_kquant_ref pattern), scale fields finite and sized so |W| ~ 1/sqrt(K).
    dmin is `dmin_ratio` x d, the profile released files ship (medians ~8 for Q4_K, ~17 for Q5_K)."""
    elems, bbytes, _ = GGML_DEQUANT[gtype]
    n = n_rows * (k // elems)
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(n, bbytes), dtype=np.uint8)
    mag = 1.0 / np.sqrt(k) / (64.0 if gtype == GGML_Q8_0 else 16.0)
    d = (rng.uniform(0.5, 1.5, size=(n,)) * mag).astype(np.float16)
    cols = FP16_COLS[gtype]
    b[:, cols[0][0]:cols[0][1]] = d.view(np.uint8).reshape(n, 2)
    if len(cols) > 1:
        b[:, cols[1][0]:cols[1][1]] = (d * np.float16(dmin_ratio / 16.0)).view(np.uint8).reshape(n, 2)
    return b


def _oracle_w(gtype, b, n_rows, k):
    return torch.from_numpy(ORACLE[gtype].dequantize_blocks(b).astype(np.float64)).reshape(n_rows, k)


def _rel(a, ref):
    return float(torch.linalg.norm(a.double() - ref) / torch.linalg.norm(ref))


def _dx_path(gtype, data, shape, dy):
    """The path under test: KQuantLinearRef's backward, through autograd."""
    x = torch.zeros(dy.shape[0], shape[1], requires_grad=True)
    kquant_linear(x, data, gtype, shape).backward(dy)
    return x.grad


def _dx_from_w(w32, dy):
    """The bf16 control (and any mutant): bf16 W, fp32 accumulate, bf16 out."""
    return (dy.to(torch.bfloat16).float() @ w32.to(torch.bfloat16).float()).to(torch.bfloat16)


def check_bars(e_test, e_ctrl):
    """The registered bars. Raises AssertionError, the exception every armed fault must produce."""
    assert 0.0 < e_ctrl < B_ABS, f"control not live: e_ctrl={e_ctrl:.3e}"
    bound = min(B_REL * e_ctrl, B_ABS)
    assert e_test <= bound, f"e_test={e_test:.3e} exceeds the bound {bound:.3e} (e_ctrl={e_ctrl:.3e})"


def _case(gtype, M, N, K, seed):
    b = _blocks(gtype, N, K, seed)
    g = torch.Generator().manual_seed(seed)
    dy = torch.randn(M, N, generator=g).to(torch.bfloat16)
    w64 = _oracle_w(gtype, b, N, K)
    w32 = dequantize_ggml(gtype, b.tobytes(), (N, K))
    return b, dy, w64, w32, dy.double() @ w64


def _seed(gtype, i, s):
    return 7919 * gtype + 101 * i + s


# -- the path under test passes ------------------------------------------------------------------------------
@pytest.mark.parametrize("gtype", TYPES, ids=[GGML_TYPE_NAMES[t] for t in TYPES])
@pytest.mark.parametrize("si", range(len(SHAPES)))
def test_dx_meets_the_bars_and_equals_the_reference_bitwise(gtype, si):
    M, N, K = SHAPES[si]
    for s in range(3):
        b, dy, w64, w32, ref = _case(gtype, M, N, K, _seed(gtype, si, s))
        dx = _dx_path(gtype, b.tobytes(), (N, K), dy)
        ctrl = _dx_from_w(w32, dy)
        assert torch.equal(dx.to(torch.bfloat16), ctrl), "the Function's dX must equal the bf16 reference bit for bit"
        check_bars(_rel(dx, ref), _rel(ctrl, ref))


@pytest.mark.parametrize("gtype", TYPES, ids=[GGML_TYPE_NAMES[t] for t in TYPES])
def test_adjoint_identity_holds_in_fp64(gtype):
    """<dY, X W^T> == <dY W, X> on the fp64 oracle: catches transpose, layout and row-order bugs in the harness."""
    M, N, K = 17, 37, 512
    b, dy, w64, _, _ = _case(gtype, M, N, K, _seed(gtype, 9, 0))
    x = torch.randn(M, K, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    lhs = float((dy.double() * (x @ w64.T)).sum())
    rhs = float(((dy.double() @ w64) * x).sum())
    assert abs(lhs - rhs) <= ADJOINT * abs(lhs)


def test_dx_responds_to_dy():
    b, dy, _, _, _ = _case(GGML_Q4_K, 4, 37, 256, 11)
    dy2 = dy.clone()
    dy2[1] = -dy2[1]
    assert not torch.equal(_dx_path(GGML_Q4_K, b.tobytes(), (37, 256), dy),
                           _dx_path(GGML_Q4_K, b.tobytes(), (37, 256), dy2))


def test_frozen_bytes_take_no_gradient_and_adapters_match_the_dense_path():
    """LoRA over the frozen k-quant weight: only the adapters learn, and their gradients match the same adapters over
    the dense bf16 weight within 1e-2 relative."""
    N, K, r = 37, 512, 4
    b = _blocks(GGML_Q5_K, N, K, 3)
    data = torch.from_numpy(b.reshape(-1).copy())
    w_dense = dequantize_ggml(GGML_Q5_K, b.tobytes(), (N, K)).to(torch.bfloat16)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(9, K, generator=g)
    grads = []
    for packed in (True, False):
        A = (torch.randn(r, K, generator=torch.Generator().manual_seed(1)) * 0.02).requires_grad_()
        B = (torch.randn(N, r, generator=torch.Generator().manual_seed(2)) * 0.02).requires_grad_()
        base = kquant_linear(x, data, GGML_Q5_K, (N, K)) if packed else \
            (x.to(torch.bfloat16).float() @ w_dense.float().T).to(torch.bfloat16)
        y = base.float() + (x @ A.T) @ B.T
        y.square().mean().backward()
        grads.append((A.grad.clone(), B.grad.clone()))
    assert data.grad is None and not data.requires_grad
    for gp, gd in zip(grads[0], grads[1]):
        assert gp.abs().sum() > 0
        assert _rel(gp, gd.double()) <= 1e-2


# -- armed: every fault must FAIL a bar -----------------------------------------------------------------------
def _m1_scale_spill(gtype, b, N, K):
    orig = kq._get_scale_min_k4

    def broken(scales):
        s = scales.reshape(-1, 3, 4)
        d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
        return torch.cat([d & 0x3F, m_d & 0x0F], -1), torch.cat([m & 0x3F, m_d >> 4], -1)
    kq._get_scale_min_k4 = broken
    try:
        return dequantize_ggml(gtype, b.tobytes(), (N, K))
    finally:
        kq._get_scale_min_k4 = orig


def _m2_q3k_hmask_not_inverted(gtype, b, N, K):
    b2 = b.copy()
    b2[:, :32] = ~b2[:, :32]                     # equivalent to dropping the documented inversion of the high bit
    return dequantize_ggml(gtype, b2.tobytes(), (N, K))


def _m5_dmin_zero(gtype, b, N, K):
    b2 = b.copy()
    lo, hi = FP16_COLS[gtype][1]
    b2[:, lo:hi] = 0
    return dequantize_ggml(gtype, b2.tobytes(), (N, K))


def _m6_signed_as_unsigned(gtype, b, N, K):
    if gtype == GGML_Q8_0:
        d = torch.from_numpy(b[:, 0:2].copy().view(np.float16).astype(np.float32))
        return (d * torch.from_numpy(b[:, 2:34].astype(np.float32))).reshape(N, K)
    b2 = b.copy()
    b2[:, 192:208] = np.abs(b2[:, 192:208].view(np.int8).astype(np.int16)).clip(0, 127).astype(np.uint8)
    return dequantize_ggml(gtype, b2.tobytes(), (N, K))


def scales_to_bf16(gtype, b):
    """Every fp16 super-block scale (d, dmin) rounded to bf16 precision: the stated KERNEL CONSTRAINT's violation."""
    b2 = b.copy()
    for lo, hi in FP16_COLS[gtype]:
        f = torch.from_numpy(b2[:, lo:hi].copy().view(np.float16).astype(np.float32)).to(torch.bfloat16).float()
        b2[:, lo:hi] = f.numpy().astype(np.float16).view(np.uint8).reshape(-1, 2)
    return b2


FAULTS = {
    "m1_scale_spill": ((GGML_Q4_K, GGML_Q5_K), _m1_scale_spill),
    "m2_q3k_hmask": ((GGML_Q3_K,), _m2_q3k_hmask_not_inverted),
    "m4_row_shift": (tuple(TYPES), lambda g, b, N, K: torch.roll(dequantize_ggml(g, b.tobytes(), (N, K)), 1, 0)),
    "m5_dmin_zero": (DMIN_TYPES, _m5_dmin_zero),
    "m6_signed_as_unsigned": ((GGML_Q8_0, GGML_Q6_K), _m6_signed_as_unsigned),
}


@pytest.mark.parametrize("name", sorted(FAULTS))
def test_every_fault_fails_a_bar(name):
    types, fault = FAULTS[name]
    for gtype in types:
        for si, (M, N, K) in enumerate(SHAPES):
            if name == "m4_row_shift" and N == 1:
                continue                          # rolling a single row is the identity
            b, dy, w64, w32, ref = _case(gtype, M, N, K, _seed(gtype, si, 0))
            e_ctrl = _rel(_dx_from_w(w32, dy), ref)
            with pytest.raises(AssertionError, match="exceeds the bound"):
                check_bars(_rel(_dx_from_w(fault(gtype, b, N, K), dy), ref), e_ctrl)


@pytest.mark.parametrize("gtype", TYPES, ids=[GGML_TYPE_NAMES[t] for t in TYPES])
def test_m3_transpose_at_n_equals_k_fails_a_bar(gtype):
    """Wt in place of W: at N == K the shapes still agree and the answer is silently wrong."""
    b, dy, w64, w32, ref = _case(gtype, 16, 256, 256, _seed(gtype, 3, 0))
    with pytest.raises(AssertionError, match="exceeds the bound"):
        check_bars(_rel(_dx_from_w(w32.T.contiguous(), dy), ref), _rel(_dx_from_w(w32, dy), ref))


def check_decode_probe(dx_probe, w_bf16):
    """The decode probe: with dY = I_N the backward returns W itself (bf16), so the path's decode must equal the
    reference decode bit for bit. Raises AssertionError on any difference."""
    diff = int((dx_probe.to(torch.bfloat16) != w_bf16).sum())
    assert diff == 0, f"decode probe: {diff} of {w_bf16.numel()} weights differ from the reference decode"


@pytest.mark.parametrize("gtype", TYPES, ids=[GGML_TYPE_NAMES[t] for t in TYPES])
def test_decode_probe_passes_the_path_and_catches_bf16_superblock_scales(gtype):
    """KERNEL CONSTRAINT: d and dmin stay fp16/fp32, as dequantize_ggml applies them. Rounded to bf16 they cost a
    median 2.1-2.9x the floor on released Q2_K/Q4_K/Q5_K bytes, but as little as 1.3x on some draws, so the per-case
    error bar alone catches them only most of the time (CALIBRATION.md). The decode probe catches them every time,
    on every type, and needs no real bytes."""
    N, K = 37, 512
    b = _blocks(gtype, N, K, _seed(gtype, 8, 0))
    eye = torch.eye(N, dtype=torch.bfloat16)
    w_ref = dequantize_ggml(gtype, b.tobytes(), (N, K)).to(torch.bfloat16)
    check_decode_probe(_dx_path(gtype, b.tobytes(), (N, K), eye), w_ref)
    w_bf16_scales = dequantize_ggml(gtype, scales_to_bf16(gtype, b).tobytes(), (N, K))
    with pytest.raises(AssertionError, match="decode probe"):
        check_decode_probe(_dx_from_w(w_bf16_scales, eye), w_ref)


# -- real released bytes (env-gated) ---------------------------------------------------------------------------
def _fixture_entries():
    dirs = [Path(p) for p in os.environ.get("GNF4_GGUF_FIXTURES", "").split(os.pathsep) if p]
    dirs = [d for d in dirs if (d / "manifest.json").is_file()]
    if not dirs:
        pytest.skip("GNF4_GGUF_FIXTURES not set (run scripts/fetch_gguf_fixtures.py)")
    for d in dirs:
        for e in json.loads((d / "manifest.json").read_text())["entries"]:
            if e["ggml_type"] in ORACLE:
                raw = (d / e["file"]).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == e["sha256"], f"{e['file']} does not match its pinned sha256"
                yield e, raw


def test_real_released_bytes_meet_the_bars_and_arm_the_faults():
    n = 0
    for e, raw in _fixture_entries():
        gtype, (N, K) = e["ggml_type"], e["shape"]
        _, bbytes, _ = GGML_DEQUANT[gtype]
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bbytes).copy()
        w64 = _oracle_w(gtype, b, N, K)
        w32 = dequantize_ggml(gtype, raw, (N, K))
        dy = torch.randn(17, N, generator=torch.Generator().manual_seed(gtype)).to(torch.bfloat16)
        ref = dy.double() @ w64
        e_ctrl = _rel(_dx_from_w(w32, dy), ref)
        check_bars(_rel(_dx_path(gtype, raw, (N, K), dy), ref), e_ctrl)
        for name, (types, fault) in FAULTS.items():
            if gtype in types and not (name == "m4_row_shift" and N == 1):
                with pytest.raises(AssertionError, match="exceeds the bound"):
                    check_bars(_rel(_dx_from_w(fault(gtype, b, N, K), dy), ref), e_ctrl)
        eye = torch.eye(N, dtype=torch.bfloat16)
        check_decode_probe(_dx_path(gtype, raw, (N, K), eye), w32.to(torch.bfloat16))
        w_bf16_scales = dequantize_ggml(gtype, scales_to_bf16(gtype, b).tobytes(), (N, K)).to(torch.bfloat16)
        if not torch.equal(w_bf16_scales, w32.to(torch.bfloat16)):    # some released tensors' scales are bf16-exact
            with pytest.raises(AssertionError, match="decode probe"):
                check_decode_probe(_dx_from_w(w_bf16_scales, eye), w32.to(torch.bfloat16))
        n += 1
    assert n, "the fixture manifests held no k-quant tensor"
