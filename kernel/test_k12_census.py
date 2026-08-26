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
# Verbatim rows from certrun/sv2/sv2_replay_kernels.txt (8 replay
# steps), FULL WIDTH. An earlier version of this fixture trimmed
# the columns for readability -- which made it stop being a real
# excerpt in exactly the dimension the coverage check reads, and
# the check then saw 0% coverage on a complete table. A fixture
# is only "real" in the dimensions you did not edit.
#
# It contains both hazards on purpose: two DISTINCT
# elementwise_kernel<128, 4, ...> rows whose truncated names are
# byte-identical, and a cudaDeviceSynchronize runtime row.
REAL = """profiled replay steps: 8 (active window: 8/8)
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       _gemv_nf4_dotpad         0.00%       0.000us         0.00%       0.000us       0.000us      19.750ms        38.11%      19.750ms      25.716us           768  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.178ms         2.27%       1.178ms       1.503us           784  
void at::native::(anonymous namespace)::indexSelectS...         0.00%       0.000us         0.00%       0.000us       0.000us     997.767us         1.93%     997.767us       2.545us           392  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     876.233us         1.69%     876.233us       2.282us           384  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     708.674us         1.37%     708.674us       1.846us           384  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     656.406us         1.27%     656.406us       1.709us           384  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     632.809us         1.22%     632.809us       0.824us           768  
void at::native::(anonymous namespace)::indexSelectS...         0.00%       0.000us         0.00%       0.000us       0.000us     479.128us         0.92%     479.128us       1.248us           384  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     458.688us         0.89%     458.688us       1.170us           392  
void at::native::(anonymous namespace)::indexSelectS...         0.00%       0.000us         0.00%       0.000us       0.000us     394.843us         0.76%     394.843us       1.028us           384  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     308.075us         0.59%     308.075us       0.755us           408  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     156.290us         0.30%     156.290us       0.814us           192  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      84.222us         0.16%      84.222us       0.877us            96  
void at::native::vectorized_elementwise_kernel<2, at...         0.00%       0.000us         0.00%       0.000us       0.000us      83.683us         0.16%      83.683us       0.872us            96  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      38.014us         0.07%      38.014us       4.752us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      13.792us         0.03%      13.792us       0.862us            16  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      12.608us         0.02%      12.608us       1.576us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.943us         0.02%      10.943us       1.368us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.848us         0.02%      10.848us       1.356us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       8.128us         0.02%       8.128us       1.016us             8  
void at::native::vectorized_elementwise_kernel<2, at...         0.00%       0.000us         0.00%       0.000us       0.000us       7.585us         0.01%       7.585us       0.948us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.720us         0.01%       6.720us       0.840us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.657us         0.01%       6.657us       0.832us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.273us         0.01%       6.273us       0.784us             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       5.537us         0.01%       5.537us       0.692us             8  
                                  cudaDeviceSynchronize        83.45%      45.728ms        83.45%      45.728ms      22.864ms       0.000us         0.00%       0.000us       0.000us             2  
-----
Self CUDA time total: 26.892ms
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
    assert len(d) == 9, sorted(d)


def test_census_block_has_the_shape_the_verdict_consumes():
    c = k12_census.census(REAL, REAL)
    assert set(c) == {"before", "after"}
    assert all(isinstance(v, dict) and v for v in c.values())


# ---- row-limit / coverage -------------------------------------------

FOOTER = "\nSelf CUDA time total: 25.000ms\n"


def test_runtime_rows_are_not_kernels():
    """cudaDeviceSynchronize carries a call count and zero self-CUDA.

    It cannot corrupt a TRACKED family, but a parser that returns it
    invites a caller to sum it as work. A real replay census contains
    both it and `Activity Buffer Request`.
    """
    extra = (REAL
             + "                     cudaDeviceSynchronize        83.45%"
               "      45.728ms        83.45%      45.728ms      22.864ms"
               "       0.000us         0.00%       0.000us       0.000us"
               "             2  \n")
    d = k12_census.parse(extra, check_coverage=False)
    assert not any("cudaDeviceSynchronize" in k for k in d)


def test_a_row_limited_table_is_REFUSED_not_silently_partial():
    """The failure the guard exists for.

    `--replay-profile-out` writes table(row_limit=120) sorted by CUDA
    time, so a long census drops its SMALLEST rows -- which is where
    the tracked raw-ATen families live. K12's gate asks "did these
    rows fall in count?", and a row the profiler stopped printing is
    indistinguishable from one that fused away.
    """
    # Simulate the real mechanism: keep the header and the LARGEST
    # rows, drop the tail, keep the ORIGINAL footer -- which is
    # exactly what table(row_limit=N, sort_by="cuda_time_total") does.
    # Inflating the footer instead would test a case that cannot
    # happen.
    lines = REAL.splitlines()
    head = [l for l in lines[:5]]
    data = [l for l in lines[5:] if "Self CUDA time total" not in l
            and not l.startswith("-")]
    footer = [l for l in lines if "Self CUDA time total" in l]
    truncated = "\n".join(head + data[:2] + footer) + "\n"
    with pytest.raises(ValueError, match="row-limited"):
        k12_census.parse(truncated)


def test_a_complete_table_passes_coverage():
    """The fixture carries its own true footer, so it must pass."""
    d = k12_census.parse(REAL)
    assert d, "a table covering its own footer must parse"


def test_no_footer_means_no_coverage_claim():
    """Absence of a footer must not be read as full coverage OR as
    failure -- it is simply unknown, and the caller is told nothing
    either way rather than being given a false assurance."""
    d = k12_census.parse(REAL)
    assert d


def test_coverage_uses_self_cuda_not_cuda_total():
    """Column 7, not column 9.

    Summing CUDA total read 23.6 ms/step against a 12.6 ms truth in
    the F1 budget parser, because dispatcher rows inherit a CUDA-total
    from their children with zero self. Here the two columns hold the
    same value in most rows, so the fixture makes them differ.
    """
    row = ("void at::native::reduce_kernel<128, 4, at::native::R...  "
           "     0.00%       0.000us         0.00%       0.000us    "
           "   0.000us       1.000ms         5.00%      99.000ms    "
           "   2.282us           400  \n")
    text = ("profiled replay steps: 8 (active window: 8/8)\n" + row
            + "\nSelf CUDA time total: 1.000ms\n")
    d = k12_census.parse(text)          # passes iff it summed 1.000ms
    assert sum(v for k, v in d.items() if "reduce_kernel" in k) == 50.0


def test_over_coverage_is_refused_too():
    """Guarding only the LOW side is not a guard.

    A wrong-column sum (CUDA total instead of Self CUDA) makes the
    census cover 9900% of the footer, which a low-side-only check
    reads as "plenty" -- this module's own mutation test walked
    straight past it. Over-coverage cannot happen in a correct
    kernel-view sum, so it is refused.
    """
    row = ("void at::native::reduce_kernel<128, 4, at::native::R...  "
           "     0.00%       0.000us         0.00%       0.000us    "
           "   0.000us      99.000ms         5.00%      99.000ms    "
           "   2.282us           400  \n")
    text = ("profiled replay steps: 8 (active window: 8/8)\n" + row
            + "\nSelf CUDA time total: 1.000ms\n")
    with pytest.raises(ValueError, match="more than the device ran"):
        k12_census.parse(text)
