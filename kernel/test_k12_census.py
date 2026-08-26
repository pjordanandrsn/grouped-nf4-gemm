# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""k12_census must not lose counts to name truncation.

The fixture below is a REAL excerpt of a committed SV2 replay census,
not an invented table. That matters twice over: an invented fixture
would not have reproduced the ellipsis truncation these tests exist
to handle, and pinning k12's fixtures to real row names is a
correction this cycle already had to make once.
"""
import pytest

import k12_census
import k12_verdict

# Real rows from certrun/sv2/sv2_replay_kernels.txt, 8 replay steps.
# The two elementwise_kernel<128, 4, ...> rows are DISTINCT kernels
# whose displayed names are byte-identical after truncation.
REAL = """profiled replay steps: 8 (active window: 8/8)
-------------------------------------------------------  ------------  ------------
                                                   Name    Self CUDA    # of Calls
-------------------------------------------------------  ------------  ------------
                                       _gemv_nf4_dotpad      19.750ms           768
void at::native::unrolled_elementwise_kernel<at::nat...       1.178ms           784
void at::native::(anonymous namespace)::indexSelectS...     997.767us          1160
void at::native::reduce_kernel<128, 4, at::native::R...     876.233us           400
void at::native::elementwise_kernel<128, 4, at::nati...     708.674us           384
void at::native::elementwise_kernel<128, 4, at::nati...     656.406us           384
void at::native::unrolled_elementwise_kernel<at::nat...     632.809us           960
void at::native::elementwise_kernel<128, 2, at::nati...     412.100us           392
void at::native::vectorized_elementwise_kernel<4, at...     311.000us           680
"""


def _fam(d, row):
    """the same family sum k12_verdict does"""
    return sum(v for k, v in d.items() if row in k)


def test_truncation_collision_accumulates_instead_of_overwriting():
    """The bug this module exists to not have.

    Two distinct kernels print the same truncated name. A dict
    assignment keeps the last and silently drops 384 of 768 calls --
    half a family -- and the attribution gate then passes or fails on
    a number nobody chose.
    """
    d = k12_census.parse(REAL)
    key = [k for k in d if "elementwise_kernel<128, 4" in k]
    assert len(key) == 1, "the two rows must collapse to one key"
    assert d[key[0]] == pytest.approx(96.0), (
        f"expected (384+384)/8 = 96 calls/step, got {d[key[0]]} -- a "
        "value of 48 means one of the colliding rows was dropped")


def test_counts_are_per_step():
    d = k12_census.parse(REAL)
    assert _fam(d, "_gemv_nf4_dotpad") == pytest.approx(96.0)   # 768/8
    raw = k12_census.parse(REAL, per_step=False)
    assert _fam(raw, "_gemv_nf4_dotpad") == pytest.approx(768.0)


def test_tracked_matchers_are_disjoint_on_real_truncated_names():
    """The gnf4#285 finding, re-checked at the PARSE layer.

    `elementwise_kernel` as a bare substring also matches the
    `unrolled_` and `vectorized_` families. The `::` prefix separates
    them -- and these are the real truncated strings, where it has to
    hold.
    """
    d = k12_census.parse(REAL)
    for name in d:
        hits = [t for t in k12_verdict.TRACKED if t in name]
        assert len(hits) <= 1, (name, hits)
    assert _fam(d, "::elementwise_kernel") == pytest.approx(96.0 + 49.0)
    assert _fam(d, "unrolled_elementwise_kernel") == pytest.approx(218.0)
    assert _fam(d, "vectorized_elementwise_kernel") == pytest.approx(85.0)


def test_a_bare_matcher_would_have_conflated_the_families():
    """Show the disjointness assertion above could have failed."""
    d = k12_census.parse(REAL)
    bare = sum(v for k, v in d.items() if "elementwise_kernel" in k)
    prefixed = _fam(d, "::elementwise_kernel")
    assert bare > prefixed, (
        "if these are equal the fixture no longer contains the "
        "decorated families and the disjointness test proves nothing")


def test_header_absent_falls_back_to_raw_counts():
    d = k12_census.parse(REAL.split("\n", 1)[1])
    assert _fam(d, "_gemv_nf4_dotpad") == pytest.approx(768.0)


def test_zero_replay_steps_is_refused_not_divided_by():
    with pytest.raises(ValueError):
        k12_census.parse("profiled replay steps: 0\n")


def test_separator_and_header_lines_are_not_rows():
    d = k12_census.parse(REAL)
    assert not any(k.startswith("-") or "# of Calls" in k for k in d)
    # 9 data lines, TWO colliding pairs (both elementwise_kernel
    # <128, 4, ...> rows, and both unrolled_elementwise_kernel rows)
    # -> 7 unique keys. Getting this wrong is how the accumulate bug
    # hides: 218 calls/step for `unrolled` is only correct because
    # (784 + 960) / 8 accumulated rather than kept the last.
    assert len(d) == 7, sorted(d)


def test_census_block_has_the_shape_the_verdict_consumes():
    c = k12_census.census(REAL, REAL)
    assert set(c) == {"before", "after"}
    assert all(isinstance(v, dict) and v for v in c.values())
