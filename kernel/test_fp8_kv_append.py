# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Bitwise gate for the fused T=1 append (e4b F1 Stage B, arm B2).

The fused kernel must reproduce, byte for byte, what the eager path
writes: ``quantize_kv_fp8`` for the values and the paged row layout for
the placement (payload at ``fill*H*D``, fp32 scales at
``pay + fill*H*groups*4``). GPU-only — the quantize's e4m3 cast goes
through hardware ``cvt.rn.satfinite`` and an interp run would certify
the wrong instruction (the ports-drop-hardening lesson: interp-mode
masks exactly the class of bug this test exists to catch). The same
check re-runs on-box before any timed B2 arm.
"""

import pytest
import torch

import fp8_kv
from fp8_kv import E4M3_MAX, FP8_DTYPE, fp8_kv_append_t1, quantize_kv_fp8

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="bitwise e4m3 gate is hardware-cast "
                                       "specific; interp would certify the "
                                       "wrong instruction")


def _reference_row(x, bt, H, D, groups, fill):
    """The eager path's bytes for one block row holding this token."""
    q, s = quantize_kv_fp8(x.unsqueeze(0), group=None if groups == 1
                           else D // groups)
    pay = bt * H * D
    srow = H * groups * 4
    row = torch.zeros(pay + bt * srow, dtype=torch.uint8, device=x.device)
    row.narrow(0, fill * H * D, H * D).copy_(
        q.view(torch.uint8).reshape(-1))
    row.narrow(0, pay + fill * srow, srow).copy_(
        s.float().reshape(-1).view(torch.uint8))
    return row


@needs_cuda
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("fill", [0, 1, 7])
def test_bitwise_against_eager_path(groups, fill):
    torch.manual_seed(1234 + groups * 10 + fill)
    H, D, bt, blocks = 4, 128, 8, 3
    pay = bt * H * D
    row_bytes = pay + bt * H * groups * 4
    dev = "cuda"
    for trial in range(8):
        x = torch.randn(H, D, device=dev, dtype=torch.bfloat16) * (
            10.0 ** torch.randint(-2, 3, (1,), device=dev).item())
        if trial == 5:
            x[0, :D // groups] = 0.0          # all-zero group -> scale 1.0
        if trial == 6:
            x[1] = 448.0                       # at the e4m3 saturation edge
        blk = torch.randint(0, blocks, (1,)).item()
        rows = torch.randperm(16)[:blocks].to(dev, torch.int32)
        pool = torch.randint(0, 256, (16 * row_bytes,), dtype=torch.uint8,
                             device=dev)
        expect = pool.clone()
        target = int(rows[blk])
        ref = _reference_row(x, bt, H, D, groups, fill)
        pr = expect.narrow(0, target * row_bytes, row_bytes)
        pr.narrow(0, fill * H * D, H * D).copy_(
            ref.narrow(0, fill * H * D, H * D))
        srow = H * groups * 4
        pr.narrow(0, pay + fill * srow, srow).copy_(
            ref.narrow(0, pay + fill * srow, srow))

        lens = torch.tensor([blk * bt + fill], dtype=torch.int32,
                            device=dev)
        fp8_kv_append_t1(x, pool, rows, lens, row_bytes, pay, bt, groups)
        torch.cuda.synchronize()
        assert torch.equal(pool, expect), (
            f"fused bytes differ from the eager path "
            f"(groups={groups} fill={fill} trial={trial})")
        assert int(lens) == blk * bt + fill, "kernel must not touch lens"


@needs_cuda
def test_untouched_bytes_stay_untouched():
    """The kernel writes exactly one token's payload+scales; every other
    byte of the arena — other rows, other fills — must be bit-stable."""
    torch.manual_seed(7)
    H, D, bt, groups = 4, 128, 8, 2
    pay = bt * H * D
    row_bytes = pay + bt * H * groups * 4
    pool = torch.randint(0, 256, (8 * row_bytes,), dtype=torch.uint8,
                         device="cuda")
    before = pool.clone()
    rows = torch.arange(8, dtype=torch.int32, device="cuda")
    lens = torch.tensor([bt * 2 + 3], dtype=torch.int32, device="cuda")
    x = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)
    fp8_kv_append_t1(x, pool, rows, lens, row_bytes, pay, bt, groups)
    torch.cuda.synchronize()
    touched = torch.zeros_like(pool, dtype=torch.bool)
    base = 2 * row_bytes
    touched[base + 3 * H * D: base + 4 * H * D] = True
    srow = H * groups * 4
    touched[base + pay + 3 * srow: base + pay + 4 * srow] = True
    assert torch.equal(pool[~touched], before[~touched]), \
        "bytes outside the target token changed"
    assert not torch.equal(pool[touched], before[touched]), \
        "the target token's bytes never changed -- vacuous run"


def test_alignment_and_shape_refusals():
    x = torch.zeros(4, 128)
    pool = torch.zeros(1024, dtype=torch.uint8)
    tbl = torch.zeros(2, dtype=torch.int32)
    lens = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="4-byte"):
        fp8_kv_append_t1(x, pool, tbl, lens, 1023, 511, 8, 1)
    with pytest.raises(ValueError, match="divide"):
        fp8_kv_append_t1(x, pool, tbl, lens, 1024, 512, 8, 3)
    with pytest.raises(ValueError, match="uint8"):
        fp8_kv_append_t1(x, pool.view(torch.int32), tbl, lens,
                         1024, 512, 8, 1)


def test_fp8_dtype_is_e4m3fn():
    """The kernel bitcasts float8e4nv; the reference stores
    float8_e4m3fn. These are the same bit layout — this pins the
    reference side so a future dtype change cannot silently diverge."""
    assert FP8_DTYPE == torch.float8_e4m3fn
    assert E4M3_MAX == 448.0


def test_no_triton_is_a_clear_refusal(monkeypatch):
    """Without triton the wrapper must refuse loudly, not NameError --
    the module's own contract is that its torch surface stays importable
    (and callable-with-clear-errors) on platforms with no triton."""
    monkeypatch.setattr(fp8_kv, "HAS_TRITON", False)
    with pytest.raises(RuntimeError, match="needs triton"):
        fp8_kv_append_t1(torch.zeros(4, 128),
                         torch.zeros(1024, dtype=torch.uint8),
                         torch.zeros(2, dtype=torch.int32),
                         torch.zeros(1, dtype=torch.int32),
                         1024, 512, 8, 1)


def test_cpu_tensors_refuse_cleanly():
    """triton installs on CPU-only hosts, where a launch dies inside
    triton's driver ("0 active drivers") -- an error that names
    neither this function nor the fix. Valid-shaped CPU tensors must
    get the clean availability refusal instead (e4b#251). Runs
    everywhere: on GPU boxes the cpu tensors still make the call
    illegal for the same reason."""
    x = torch.zeros(2, 64)
    pool = torch.zeros(4096, dtype=torch.uint8)
    row = torch.zeros(4, dtype=torch.int32)
    lens = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="CUDA-resident"):
        fp8_kv_append_t1(x, pool, row, lens, 2048, 1024, 16, 4)
