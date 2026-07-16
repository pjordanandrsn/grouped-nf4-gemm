# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R1 anchor gate: the projection model must reproduce the already-published
flagship numbers before any new projected row is trusted. Red anchor => stop.

Anchors (from bench/phase3/flagship/ committed receipts):
  Phase A  Qwen3-235B-A22B  link 44.3 GB/s  7.98 GB/tok  -> waterfall 5.55
  671B-class scaling point   link 55.5 GB/s  12.09 GB/tok -> waterfall 4.594
  gen4-desktop projection    ~24 GB/s        7.98 GB/tok  -> ~3.0 (published band)
"""
from model import (BF16_BYTES_PER_PARAM, NF4_BYTES_PER_PARAM, gb_per_token,
                   streaming_waterfall_toks, unified_waterfall_toks)


def approx(a, b, tol=0.02):
    return abs(a - b) <= tol * b


def test_phaseA_235b_streaming_anchor():
    assert approx(streaming_waterfall_toks(44.3, 7.98), 5.55)


def test_671b_streaming_anchor():
    assert approx(streaming_waterfall_toks(55.5, 12.09), 4.594)


def test_gen4_desktop_projection_band():
    v = streaming_waterfall_toks(24.0, 7.98)
    assert 2.7 <= v <= 3.3, v


def test_geometry_reproduces_measured_235b_bytes():
    # H=4096 I=1536 L=94 k=8 -> the measured 7.98 GB/token, exactly
    assert approx(gb_per_token(94, 8, 4096, 1536), 7.98, tol=0.01)


def test_geometry_reproduces_measured_671b_bytes():
    # DeepSeek-V3-class: H=7168 I=2048 L=61 k=8 -> the measured 12.09 GB/token
    assert approx(gb_per_token(61, 8, 7168, 2048), 12.09, tol=0.03)


def test_nf4_vs_bf16_reduction_is_3p56x_not_4x():
    # The honest number: NF4 carries fp32 absmax (0.5+0.0625 vs bf16 2.0), so
    # the byte reduction on the binding resource is ~3.56x, NOT a round 4x.
    # Same ratio in both regimes (absmax-inclusive).
    g_nf4 = gb_per_token(94, 8, 4096, 1536, NF4_BYTES_PER_PARAM)
    g_bf16 = gb_per_token(94, 8, 4096, 1536, BF16_BYTES_PER_PARAM)
    assert approx(g_bf16 / g_nf4, 3.556, tol=0.01), g_bf16 / g_nf4
    u_nf4 = unified_waterfall_toks(256.0, 22.0, NF4_BYTES_PER_PARAM)
    u_bf16 = unified_waterfall_toks(256.0, 22.0, BF16_BYTES_PER_PARAM)
    assert approx(u_nf4 / u_bf16, 3.556, tol=0.01), u_nf4 / u_bf16


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fail = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fail += 1; print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-fail}/{len(fns)} anchors green")
    sys.exit(1 if fail else 0)
