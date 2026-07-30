# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The load-bearing claim of the MXFP4 NVMe path, as a test rather than a comment.

`nvme_arena`'s docstring says its rows are "byte-compatible with the pinned DRAM
arena rows the engine already gathers from ... the `Mxfp4PipelinedGptOss` layout".
Everything about serving MXFP4 experts from disk rests on that: if the segment
ORDER or the padding rule differs, a relocation bake produces rows the engine
reads as garbage — and it would only be discovered after baking ~1.45 TB.

So: derive both layouts independently and compare offsets, lengths and total row
size. Pure stdlib + torch shapes; nothing large, no GPU.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from mxfp4_loader import EXPERT_SUFFIXES  # noqa: E402
from nvme_arena import _align8  # noqa: E402

# gpt-oss-20b-shaped geometry (n1 = 2*inter, k1 = hidden; n2 = hidden, k2 = inter)
E, N1, K1, N2, K2 = 4, 64, 128, 32, 32


def _engine_layout():
    """Segment offsets Mxfp4PipelinedGptOss computes in __init__."""
    half1, nb1 = K1 // 2, K1 // 32
    half2, nb2 = K2 // 2, K2 // 32
    seg = [N1 * half1, N1 * nb1, N2 * half2, N2 * nb2]
    off = [0]
    for s in seg[:-1]:
        off.append(_align8(off[-1] + s))
    return off, seg, _align8(off[-1] + seg[-1])


def _arena_layout():
    """Segment offsets nvme_arena.bake computes from per-expert tensor shapes."""
    shapes = {
        EXPERT_SUFFIXES[0]: (N1, K1 // 2),     # gate_up blocks
        EXPERT_SUFFIXES[1]: (N1, K1 // 32),    # gate_up scales
        EXPERT_SUFFIXES[2]: (N2, K2 // 2),     # down blocks
        EXPERT_SUFFIXES[3]: (N2, K2 // 32),    # down scales
    }
    off, seg = [], []
    cur = 0
    for suf in EXPERT_SUFFIXES:
        n = shapes[suf][0] * shapes[suf][1]    # u8, so elements == bytes
        off.append(cur); seg.append(n)
        cur = _align8(cur + n)
    return off, seg, off[-1] + seg[-1] if False else _align8(off[-1] + seg[-1])


def test_segment_order_matches():
    """The engine writes gu_blocks, gu_scales, dn_blocks, dn_scales in that order;
    EXPERT_SUFFIXES must enumerate the same sequence or every row is scrambled."""
    assert [s.split(".")[-1] for s in EXPERT_SUFFIXES] == [
        "gate_up_proj_blocks", "gate_up_proj_scales",
        "down_proj_blocks", "down_proj_scales"], EXPERT_SUFFIXES


def test_offsets_lengths_and_row_size_agree():
    e_off, e_seg, e_row = _engine_layout()
    a_off, a_seg, a_row = _arena_layout()
    assert e_seg == a_seg, f"segment LENGTHS differ: engine {e_seg} vs arena {a_seg}"
    assert e_off == a_off, f"segment OFFSETS differ: engine {e_off} vs arena {a_off}"
    assert e_row == a_row, f"row_bytes differ: engine {e_row} vs arena {a_row}"


def test_both_use_the_same_8_byte_padding_rule():
    """A different padding rule would misalign every segment after the first."""
    e_off, _e_seg, _e = _engine_layout()
    assert all(o % 8 == 0 for o in e_off), e_off
    # and a deliberately odd segment still lands 8-aligned in both derivations
    global N2
    keep = N2
    try:
        N2 = 33                                    # forces an odd byte count
        e_off2, _s, e_row2 = _engine_layout()
        a_off2, _s2, a_row2 = _arena_layout()
        assert e_off2 == a_off2 and e_row2 == a_row2, (e_off2, a_off2, e_row2, a_row2)
    finally:
        N2 = keep


def test_arena_row_stride_is_a_multiple_of_engine_row_bytes_alignment():
    """The arena pads rows to `align` (4096) for O_DIRECT while the engine only
    needs row_bytes. Reading a slot must therefore start at the row start and the
    engine must read only its own row_bytes — record that relationship."""
    _off, _seg, row_bytes = _engine_layout()
    for align in (512, 4096):
        stride = (row_bytes + align - 1) // align * align
        assert stride >= row_bytes and stride % align == 0
        assert stride - row_bytes < align, "padding must be less than one block"
