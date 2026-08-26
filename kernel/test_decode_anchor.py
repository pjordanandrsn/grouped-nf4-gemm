# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The committed anchor must agree with the RESULTS that certified it.

The defect this guards: a constant living only in a harness, drifting
from the document, gating rentals nobody re-checked.
"""
import pathlib
import re

import decode_anchor as A

_ROOT = pathlib.Path(__file__).resolve().parent
_RESULTS = _ROOT / "RESULTS-m2-anchor-recert.md"


def test_anchor_matches_its_results_document():
    text = _RESULTS.read_text()
    m = re.search(r"M2 ANCHOR CERTIFIED:\s*([\d.]+)\s*ms", text)
    assert m, "RESULTS-m2 has no machine-readable certified line"
    assert abs(A.ANCHOR_MS - float(m.group(1))) < 1e-9


def test_gate_bounds_match_the_document():
    """The module docstring CLAIMS this file and the RESULTS cannot
    disagree. That claim was false: the sync test locked ANCHOR_MS
    only, leaving the gate bounds -- the constants that actually
    decide rentals -- free to drift (Bugbot, gnf4#278). A comment
    asserting a property is not an enforcement of it."""
    text = _RESULTS.read_text()
    m = re.search(r"GATE\s*=\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*ms",
                  text)
    assert m, "RESULTS-m2 has no machine-readable GATE line"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert abs(A.GATE_LO_MS - lo) < 1e-9, (A.GATE_LO_MS, lo)
    assert abs(A.GATE_HI_MS - hi) < 1e-9, (A.GATE_HI_MS, hi)
    assert A.window() == (lo, hi)


def test_document_records_the_population_the_gate_covers():
    """The bounds are only defensible as 'population + margin', so the
    population must be in the document too, not just in this file."""
    text = _RESULTS.read_text()
    m = re.search(r"population\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]",
                  text)
    assert m, "RESULTS-m2 does not record the population the gate covers"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert A.GATE_LO_MS < lo and A.GATE_HI_MS > hi, \
        "the gate must strictly contain the population it was built from"


def test_gate_contains_every_box_it_was_built_from():
    """The whole point. The registered median-centred window excluded
    box 1 at 7.751 -- publishing it would have destroyed a normal box
    for being normal, which is the failure this cycle diagnosed."""
    for measured in (7.147, 7.369, 7.751):
        assert A.compliant(measured), measured
    lo, hi = A.window()
    assert lo < 7.147 and hi > 7.751


def test_retired_window_is_named_and_would_have_failed():
    """Kept recognisable: a harness still using +/-4.2% about the
    median rejects a box this cycle measured."""
    r_lo = A.ANCHOR_MS * (1 - A._RETIRED_WINDOW)
    r_hi = A.ANCHOR_MS * (1 + A._RETIRED_WINDOW)
    assert not (r_lo <= 7.751 <= r_hi)


def test_compliance_bounds():
    lo, hi = A.window()
    assert lo < A.ANCHOR_MS < hi
    assert A.compliant(A.ANCHOR_MS)
    assert A.compliant(lo) and A.compliant(hi)
    assert not A.compliant(lo * 0.99)
    assert not A.compliant(hi * 1.01)
    # a grossly broken box must still be excluded -- the gate is an
    # outlier excluder, and it has to actually exclude
    assert not A.compliant(9.0)
    assert not A.compliant(5.0)


def test_superseded_constants_are_recorded():
    """7.39 gated this campaign and was never certified. Keeping it
    named makes a stale harness recognisable."""
    assert A._SUPERSEDED["uncertified_harness_constant"] == 7.39
    assert A._SUPERSEDED["prior_certified"] == 7.35


def test_basis_is_recorded():
    """A probe taken with other arguments is not comparable."""
    assert A.ANCHOR_BASIS["gen_tokens"] == 128
    assert A.ANCHOR_BASIS["n_steps"] == 127
    assert A.ANCHOR_BASIS["knob"] == "off"
