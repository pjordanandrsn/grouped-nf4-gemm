# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The certified decode anchor, in the repository rather than a script.

An uncertified constant gated every box rental in this campaign: the
hunt harness screened on 7.39 ms, a number that appears in no RESULTS
document, carried forward in a scratchpad script
([[harness-defaults-are-values]]). This module is the committed
source; harnesses read it instead of holding a literal.

The values come from RESULTS-m2-anchor-recert.md (three RTX 5090
boxes, A/A pairs, no anchor gate, receipts in receipts-m2/), and
`test_decode_anchor.py` fails if this file and that document disagree.

READ THE WINDOW BEFORE USING IT. M2 measured **8.5% inter-box
dispersion** while every box repeated itself to 0.16%. The window is
therefore an OUTLIER EXCLUDER, not a certification of a class: a box
inside it is not thereby equivalent to any other box inside it. What
protects a verdict is each cycle's own same-box denominator, never
this constant.
"""

#: Median of three per-box A/A medians, ms/step. Knob-OFF b1 graph
#: decode, prompt_len 512, gen_tokens 128 (n_steps 127) -- the same
#: window the prior 7.35 was certified on.
ANCHOR_MS = 7.369

#: Gate bounds, ms, INCLUSIVE. Deliberately NOT a centre+/-window.
#:
#: M2's registered rule said +/-(spread/2) about the median. Applied
#: here that is [7.060, 7.678] -- which EXCLUDES box 1 at 7.751, one
#: of the three boxes the anchor was computed from. A half-spread
#: window centred on the median only spans the population when the
#: median is also the midrange, and it is not. Publishing that window
#: would have repeated the exact failure this cycle diagnosed: a
#: normal box destroyed for being normal (Bugbot, gnf4#278).
#:
#: These bounds instead cover the MEASURED population [7.147, 7.751]
#: with a 2% margin each side. They are wide -- 12.6% -- and that
#: width IS the finding: with 8.5% inter-box dispersion a gate can
#: only exclude gross outliers.
GATE_LO_MS = 7.004
GATE_HI_MS = 7.906

#: Retired: the median-centred window the registered rule produced.
#: Kept named so a harness still using it is recognisable.
_RETIRED_WINDOW = 0.042

#: What the anchor was measured ON. A probe taken with different
#: arguments is not comparable to ANCHOR_MS.
ANCHOR_BASIS = {"knob": "off", "loop": "b1d graph", "batch": 1,
                "prompt_len": 512, "gen_tokens": 128, "n_steps": 127}

#: Superseded, kept so a stale harness is recognisable rather than
#: mysterious. 7.39 was never certified; 7.35 was, and M2 found it
#: correct to 0.26%.
_SUPERSEDED = {"uncertified_harness_constant": 7.39,
               "prior_certified": 7.35}


def window() -> tuple[float, float]:
    """Inclusive (low, high) ms bounds a probe must fall within."""
    return (GATE_LO_MS, GATE_HI_MS)


def compliant(step_ms: float) -> bool:
    """True if a probe is inside the window. See the module docstring:
    this EXCLUDES OUTLIERS, it does not certify equivalence."""
    lo, hi = window()
    return lo <= step_ms <= hi
