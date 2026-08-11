# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Oracle adjudication for the k-quant reference — gguf-py (the llama.cpp
project's numpy implementation) is ground truth; disagreement is STOP, not
tolerance (the test_mxfp4_oracle pattern).

Two arms:
  1. Synthetic: random block bytes per type (with the fp16 scale fields
     constrained to finite values so both sides see identical arithmetic —
     random exponent bytes would manufacture inf/NaN edges no released file
     contains), compared BIT-exact via int32 views.
  2. Real bytes (env-gated): tensors range-fetched from the released Glimmer
     GGUFs by scripts/fetch_gguf_fixtures.py, sha256-pinned in the manifest.
     Set GNF4_GGUF_FIXTURES to the fixture dir to enable.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

gguf = pytest.importorskip("gguf")
import numpy as np  # noqa: E402  (gguf-py guarantees numpy)
import gguf.quants as gq  # noqa: E402

from kquant_ref import (  # noqa: E402
    GGML_DEQUANT, GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_Q8_0, GGML_TYPE_NAMES, dequantize_ggml,
)

_ORACLE = {
    GGML_Q8_0: gq.Q8_0, GGML_Q2_K: gq.Q2_K, GGML_Q3_K: gq.Q3_K,
    GGML_Q4_K: gq.Q4_K, GGML_Q5_K: gq.Q5_K, GGML_Q6_K: gq.Q6_K,
}
# Byte columns holding fp16 scales (d / dmin) per type — constrained finite.
_FP16_COLS = {
    GGML_Q8_0: [(0, 2)],
    GGML_Q2_K: [(80, 82), (82, 84)],
    GGML_Q3_K: [(108, 110)],
    GGML_Q4_K: [(0, 2), (2, 4)],
    GGML_Q5_K: [(0, 2), (2, 4)],
    GGML_Q6_K: [(208, 210)],
}


def _random_blocks(gtype: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, bbytes, _ = GGML_DEQUANT[gtype]
    blocks = rng.integers(0, 256, size=(n, bbytes), dtype=np.uint8)
    finite = rng.uniform(-8.0, 8.0, size=(n,)).astype(np.float16)
    for lo, hi in _FP16_COLS[gtype]:
        blocks[:, lo:hi] = finite.view(np.uint8).reshape(n, 2)
        finite = (finite * np.float16(0.5))   # distinct d vs dmin
    return blocks


@pytest.mark.parametrize("gtype", sorted(_ORACLE))
def test_synthetic_bit_exact(gtype):
    n = 37
    blocks = _random_blocks(gtype, n, seed=0xC0FFEE + gtype)
    truth = _ORACLE[gtype].dequantize_blocks(blocks).astype(np.float32)
    elems = GGML_DEQUANT[gtype][0]
    ours = dequantize_ggml(gtype, blocks.tobytes(), (n, elems)).numpy()
    assert truth.shape == (n, elems) and ours.shape == (n, elems)
    assert np.array_equal(ours.view(np.int32), truth.view(np.int32)), \
        f"{GGML_TYPE_NAMES[gtype]}: reference disagrees with gguf-py — STOP"


def test_unknown_type_refuses():
    with pytest.raises(ValueError, match="not in the k-quant lane"):
        dequantize_ggml(23, b"\x00" * 136, (1, 256))   # IQ4_XS


def _fixture_dir():
    d = os.environ.get("GNF4_GGUF_FIXTURES", "")
    if not d or not Path(d, "manifest.json").is_file():
        pytest.skip("GNF4_GGUF_FIXTURES not set (run scripts/fetch_gguf_fixtures.py)")
    return Path(d)


def test_real_released_bytes_decode_and_match_oracle():
    d = _fixture_dir()
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["entries"], "empty fixture manifest"
    for e in manifest["entries"]:
        raw = (d / e["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"], \
            f"fixture {e['file']} does not match its pinned sha256"
        gtype = e["ggml_type"]
        shape = tuple(e["shape"])
        ours = dequantize_ggml(gtype, raw, shape)
        if gtype in _ORACLE:
            _, bbytes, _ = GGML_DEQUANT[gtype]
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bbytes)
            truth = _ORACLE[gtype].dequantize_blocks(blocks).astype(np.float32)
            assert np.array_equal(ours.numpy().reshape(truth.shape).view(np.int32),
                                  truth.view(np.int32)), \
                f"{e['tensor']} ({e['type_name']}): real-byte mismatch — STOP"
        assert torch.isfinite(ours).all(), \
            f"{e['tensor']}: released bytes decoded to non-finite values"


# --- malformed-input guards -------------------------------------------------
# These decode bytes that came off disk: a truncated download, a wrong tensor
# offset, a mis-parsed header. The guards were bare `assert`s, which (a) print
# nothing useful and (b) VANISH under `python -O`. They are ValueErrors now, so
# they survive optimization and name what disagrees.

@pytest.mark.parametrize("data, shape, expect", [
    (bytes(288), (256,), "288"),          # 2 blocks of data, shape claims 1
    (bytes(143), (256,), "143"),          # one byte short of a block
    (bytes(288), (2, 128), "not a multiple"),   # row ends mid-block
])
def test_malformed_kquant_buffers_raise_a_useful_valueerror(data, shape, expect):
    import kquant_ref as K
    with pytest.raises(ValueError) as ei:
        K.dequantize_ggml(12, data, shape)      # 12 = Q4_K
    msg = str(ei.value)
    assert "Q4_K" in msg, "the error must name the quant type"
    assert expect in msg, f"expected {expect!r} in: {msg}"


def test_unquantized_types_also_validate_their_length():
    import kquant_ref as K
    with pytest.raises(ValueError, match="disagree"):
        K.dequantize_ggml(0, bytes(16), (8,))   # 0 = F32: 16B = 4 values, not 8
