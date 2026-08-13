# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bf16 absmax storage: lossless where claimed, refused where not.

absmax is 11.1% of a Qwen3-30B arena row and ships as fp32. It is the max of
``|w|`` over a block, so for a bf16 checkpoint it IS one of the source
magnitudes and is exactly representable in bf16 — measured on the real model,
80/80 expert tensors bitwise identical after a round-trip. Storing it bf16 takes
5.6% off every row read at zero accuracy cost.

The whole change rests on that exactness, so the tests are built around what
would happen if it were ever false:

  * ``cast_absmax`` REFUSES an inexact cast rather than rounding quietly.
  * ``auto`` decides from the SOURCE dtype (a proof) and not from sampling.
  * the end-to-end arena comparison asserts BITWISE equality of the staged
    fp32 values, not closeness — "close" is what a silent precision loss looks
    like.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import load_index  # noqa: E402
from nvme_bake_nf4 import (  # noqa: E402
    ABSMAX_DTYPES,
    bake_nf4,
    cast_absmax,
    resolve_absmax_dtype,
)
from nvme_residency import (  # noqa: E402
    ColdTier,
    segment_geometry,
    segment_into,
    widening_casts,
)


# ---------------------------------------------------------------- cast_absmax
def test_bf16_roundtrip_is_exact_for_bf16_derived_absmax():
    """The claim, stated as a property: absmax of a bf16 tensor is bf16-exact."""
    w = torch.randn(64, 512, dtype=torch.bfloat16).float()
    am = w.reshape(64, 8, 64).abs().amax(-1)
    out = cast_absmax(am, "bf16")
    assert out.dtype is torch.bfloat16
    assert torch.equal(out.float(), am)


def test_cast_refuses_inexact_bf16():
    """An fp32-derived absmax needs more than 8 mantissa bits. Refuse, loudly.

    This is the control for the test above: if it ever passes, the exactness
    check has stopped checking anything.
    """
    am = torch.tensor([[1.0 + 2 ** -12, 3.0]], dtype=torch.float32)
    assert not torch.equal(am.to(torch.bfloat16).float(), am), "bad fixture"
    with pytest.raises(ValueError, match="not exact"):
        cast_absmax(am, "bf16")


def test_cast_f32_always_works():
    am = torch.tensor([[1.0 + 2 ** -12]], dtype=torch.float32)
    assert torch.equal(cast_absmax(am, "f32"), am)


def test_cast_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unknown absmax storage dtype"):
        cast_absmax(torch.ones(1, 1), "int8")


# ------------------------------------------------------------ resolve / auto
@pytest.mark.parametrize("source,expect", [
    ("bf16", "bf16"),
    # fp16 has 10 mantissa bits to bf16's 7, so an fp16 magnitude is not
    # generally bf16-representable and the exactness proof does not carry.
    # `auto` must NOT pick a mode cast_absmax would then refuse.
    ("fp16", "f32"), ("f16", "f32"),
    ("fp32", "f32"), ("mxfp4", "f32"), ("fp8", "f32"), ("weird", "f32"),
])
def test_auto_decides_from_source_dtype(source, expect):
    assert resolve_absmax_dtype("auto", source) == expect


def test_auto_never_picks_a_mode_the_cast_would_refuse():
    """The two halves must agree: whatever `auto` selects for a source, a real
    absmax from that source must survive `cast_absmax`. fp16 is the case that
    made this worth asserting rather than assuming."""
    w16 = torch.randn(8, 128, dtype=torch.float16).float()
    am16 = w16.reshape(8, 2, 64).abs().amax(-1)
    chosen = resolve_absmax_dtype("auto", "fp16")
    cast_absmax(am16, chosen)          # must not raise
    assert chosen == "f32"


def test_explicit_beats_auto():
    assert resolve_absmax_dtype("f32", "bf16") == "f32"
    assert resolve_absmax_dtype("bf16", "fp32") == "bf16"   # caller's call; cast still guards


def test_unknown_mode_refused():
    with pytest.raises(ValueError, match="absmax_dtype must be one of"):
        resolve_absmax_dtype("int8", "bf16")


def test_default_is_f32_so_old_consumers_keep_working():
    """The index is self-describing but older READERS are not. Flipping this
    default would break them on a library upgrade alone."""
    import inspect
    assert inspect.signature(bake_nf4).parameters["absmax_dtype"].default == "f32"
    assert "f32" in ABSMAX_DTYPES and "bf16" in ABSMAX_DTYPES


# ------------------------------------------------------------ widening table
def test_widening_is_whitelisted_and_one_directional():
    w = widening_casts()
    assert (torch.bfloat16, torch.float32) in w
    assert (torch.float16, torch.float32) in w
    # Narrowing must never appear: it would round on a path that promises the
    # bytes it serves are the bytes that were baked.
    assert (torch.float32, torch.bfloat16) not in w
    assert (torch.float32, torch.float16) not in w


# ------------------------------------------------------------------ bake e2e
# Reuse the bake suite's own snapshot builder. It hand-writes the safetensors
# container rather than importing `safetensors` -- which is the reason those
# tests RUN in CI, where that package is not installed. A prettier fixture built
# on save_file would skip here and be green about nothing.
from test_nvme_bake_nf4 import E, make_snapshot  # noqa: E402


def _fake_quantizer():
    """Deterministic stand-in for bnb: real blockwise absmax, fake nibbles.

    The absmax must be real -- the exactness property is about it. The packed
    nibbles only need to be stable bytes.
    """
    def q(w):
        N, K = w.shape
        am = w.float().reshape(N, K // 64, 64).abs().amax(-1)
        packed = (torch.arange(N * (K // 2)) % 251).to(torch.uint8).reshape(N, K // 2)
        return packed, am
    return q


@pytest.fixture()
def snapshot(tmp_path):
    d = str(tmp_path / "snap")
    make_snapshot(d)
    return d


def _bake(snapshot, out, dtype):
    bake_nf4(snapshot, out, quantize_fn=_fake_quantizer(), log=lambda *a, **k: None,
             absmax_dtype=dtype)
    return load_index(out)


def test_bf16_arena_is_smaller_by_the_predicted_amount(snapshot, tmp_path):
    i32 = _bake(snapshot, str(tmp_path / "a32.arena"), "f32")
    i16 = _bake(snapshot, str(tmp_path / "a16.arena"), "bf16")
    am32 = sum(s["length"] for s in i32["segments"] if "absmax" in s["suffix"])
    am16 = sum(s["length"] for s in i16["segments"] if "absmax" in s["suffix"])
    assert am16 * 2 == am32
    assert i16["row_bytes"] == i32["row_bytes"] - am32 // 2
    for s in i16["segments"]:
        assert s["dtype"] == ("BF16" if "absmax" in s["suffix"] else "U8")


def test_bf16_arena_stages_bitwise_identical_fp32(snapshot, tmp_path):
    """The end-to-end claim: a bf16 arena hands the kernel EXACTLY the fp32
    absmax an f32 arena would. Not close — equal."""
    p32, p16 = str(tmp_path / "b32.arena"), str(tmp_path / "b16.arena")
    i32, i16 = _bake(snapshot, p32, "f32"), _bake(snapshot, p16, "bf16")

    t32 = ColdTier(p32, hot_rows=E, pinned=False)
    t16 = ColdTier(p16, hot_rows=E, pinned=False)
    ids = list(range(E))
    for suffix in ("nf4.gate_up_absmax", "nf4.down_absmax"):
        _dt, shape, _o, _l = segment_geometry(i32, suffix)
        out32 = torch.zeros(E, *shape, dtype=torch.float32)
        out16 = torch.zeros(E, *shape, dtype=torch.float32)   # SAME fp32 dest
        segment_into(t32, i32, 0, ids, suffix, out32)
        segment_into(t16, i16, 0, ids, suffix, out16)
        assert torch.equal(out32, out16), f"{suffix} diverged"
        assert out16.dtype is torch.float32


def test_widening_refused_when_not_whitelisted(snapshot, tmp_path):
    """A bf16 destination for a bf16 segment is fine; a u8 segment into fp32 is
    not, and must still raise rather than silently reinterpret bytes."""
    p16 = str(tmp_path / "c16.arena")
    i16 = _bake(snapshot, p16, "bf16")
    t16 = ColdTier(p16, hot_rows=E, pinned=False)
    _dt, shape, _o, _l = segment_geometry(i16, "nf4.gate_up_blocks")
    bad = torch.zeros(E, *shape, dtype=torch.float32)
    with pytest.raises(TypeError, match="Widening is allowed only"):
        segment_into(t16, i16, 0, list(range(E)), "nf4.gate_up_blocks", bad)


def test_f32_arena_still_takes_the_memcpy_path(snapshot, tmp_path):
    """The default path must be untouched: same dtype in and out, no widening."""
    p32 = str(tmp_path / "d32.arena")
    i32 = _bake(snapshot, p32, "f32")
    t32 = ColdTier(p32, hot_rows=E, pinned=False)
    dt, shape, _o, _l = segment_geometry(i32, "nf4.down_absmax")
    assert dt is torch.float32
    out = torch.zeros(E, *shape, dtype=torch.float32)
    segment_into(t32, i32, 0, list(range(E)), "nf4.down_absmax", out)
    assert out.abs().sum() > 0


def test_manifest_records_the_choice(snapshot, tmp_path):
    """A receipt has to say which of the two it is; 'absmax' alone is ambiguous
    once there are two storages for it."""
    import json
    out = str(tmp_path / "e16.arena")
    _bake(snapshot, out, "bf16")
    man = json.loads(open(out + ".manifest.json").read())
    q = man.get("quantizer") or man
    blob = json.dumps(q)
    assert "bf16" in blob
