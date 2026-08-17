# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""FP8 KV format contract (Phase 7): the scale is derived so the cast can
never overflow, zeros survive exactly, the pack/unpack round trip is
byte-faithful, and the error floor is the format's own — measured, not
asserted against a number someone picked."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(__file__))

from fp8_kv import (  # noqa: E402
    E4M3_MAX,
    dequant_kv_fp8_ref,
    kv_block_bytes,
    kv_roundtrip_error,
    pack_kv_block,
    quantize_kv_fp8,
    unpack_kv_block,
)

T, H, D = 16, 4, 64
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVS)
def dev(request):
    return request.param


def test_roundtrip_is_finite_and_close(dev):
    x = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16) * 3.0
    q, s = quantize_kv_fp8(x)
    assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
    assert s.shape == (T, H) and s.dtype == torch.float32
    back = dequant_kv_fp8_ref(q, s)
    assert back.isfinite().all(), "dequant produced inf/NaN"
    # e4m3 carries 3 mantissa bits: ~6% worst-case relative step. Assert the
    # FLOOR is respected, not a tighter number that would be luck.
    rel = ((back.float() - x.float()).norm() / x.float().norm()).item()
    assert rel < 0.05, f"round trip worse than the format's floor: {rel}"


def test_extreme_magnitudes_never_overflow(dev):
    """The scale is amax-derived, so the cast cannot reach e4m3's inf/NaN
    even when the input is far outside e4m3's own range — the property
    that makes clamping unnecessary rather than merely unused."""
    for mag in (1e-8, 1e-3, 1.0, 1e4, 3e38):
        x = torch.full((2, 1, D), mag, device=dev, dtype=torch.float32)
        x[0, 0, 0] = -mag
        q, s = quantize_kv_fp8(x)
        assert q.to(torch.float32).isfinite().all(), f"mag {mag} overflowed"
        back = dequant_kv_fp8_ref(q, s, dtype=torch.float32)
        assert back.isfinite().all()
        assert torch.allclose(back, x, rtol=0.07), f"mag {mag} lost value"


def test_all_zero_rows_roundtrip_exactly(dev):
    x = torch.zeros(3, H, D, device=dev, dtype=torch.bfloat16)
    x[1, 0, 5] = 2.0                       # one live row among dead ones
    q, s = quantize_kv_fp8(x)
    back = dequant_kv_fp8_ref(q, s)
    assert torch.equal(back[0], x[0]), "zero row must survive exactly"
    assert torch.equal(back[2], x[2])
    assert (s[0] == 1.0).all(), "zero row's scale must be 1, never 0"
    assert back[1, 0, 5].item() == pytest.approx(2.0, rel=0.07)


def test_scale_is_per_token_per_head(dev):
    """Rows differing by orders of magnitude must each keep their own
    precision — the property per-tensor scaling would destroy."""
    x = torch.zeros(2, 2, D, device=dev, dtype=torch.float32)
    x[0, 0] = torch.randn(D, device=dev) * 1e-4
    x[1, 1] = torch.randn(D, device=dev) * 1e4
    q, s = quantize_kv_fp8(x)
    assert s[0, 0] < s[1, 1] / 1e6, "scales did not track their own rows"
    back = dequant_kv_fp8_ref(q, s, dtype=torch.float32)
    for t, h in ((0, 0), (1, 1)):
        rel = ((back[t, h] - x[t, h]).norm() / x[t, h].norm()).item()
        assert rel < 0.05, f"row ({t},{h}) lost precision: {rel}"


def test_pack_unpack_is_byte_faithful(dev):
    x = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    q, s = quantize_kv_fp8(x)
    row = torch.zeros(kv_block_bytes(T, H, D), dtype=torch.uint8, device=dev)
    pack_kv_block(q, s, row)
    q2, s2 = unpack_kv_block(row, T, H, D)
    assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(s2, s)
    assert torch.equal(dequant_kv_fp8_ref(q2, s2), dequant_kv_fp8_ref(q, s))


@pytest.mark.parametrize("hd", [64, 128, 256])
def test_block_bytes_counts_the_scale_tail(hd):
    """The compression ratio against bf16 is 2*D/(D+4), never a flat 2x —
    one fp32 scale rides every head_dim values. At D=64 that is 1.88x and
    at D=128 it is 1.94x, so a caller sizing a pool from the payload alone
    under-budgets by the tail."""
    payload = T * H * hd
    assert kv_block_bytes(T, H, hd) == payload + T * H * 4
    ratio = (payload * 2) / kv_block_bytes(T, H, hd)
    assert ratio == pytest.approx(2 * hd / (hd + 4), rel=1e-9)
    assert ratio < 2.0


def test_a_too_small_row_is_refused(dev):
    x = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    q, s = quantize_kv_fp8(x)
    row = torch.zeros(T * H * D, dtype=torch.uint8, device=dev)  # no tail
    with pytest.raises(ValueError, match="too small"):
        pack_kv_block(q, s, row)


def test_error_report_is_measured_not_claimed(dev):
    x = torch.randn(64, H, D, device=dev, dtype=torch.float32)
    max_abs, rel = kv_roundtrip_error(x)
    assert max_abs > 0 and rel > 0, "a lossy format reporting zero error " \
                                    "means the round trip was not exercised"
    assert rel < 0.05
    assert E4M3_MAX == 448.0
