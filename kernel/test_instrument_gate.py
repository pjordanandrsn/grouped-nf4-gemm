# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The gate must tell a noisy box from a shrunken effect.

Its first version asked one question -- effect > 3 * spread -- and refused
otherwise, which fires identically for a box that cannot measure anything and
for a clean box measuring an effect that has genuinely gone away. It refused
the demote-heap run on the second case: per-arm IQR 0.56% / 1.44%, among the
tightest measured, effect collapsed from ~8% to 1.58%. The collapse was the
result.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench",
                                "cold-engine", "routing-trace"))

from instrument_gate import judge  # noqa: E402


def _point(hard_ms, soft_ms, reads_pct=1.36):
    hard = [int(v * 1e6) for v in hard_ms]
    soft = [int(v * 1e6) for v in soft_ms]
    hm = sorted(hard)[len(hard) // 2]
    sm = sorted(soft)[len(soft) // 2]
    return {"rows": 256, "protected": 248,
            "hard_wall_ns": hard, "soft_wall_ns": soft,
            "delta_wall_pct": (sm - hm) / hm * 100,
            "delta_reads_pct": reads_pct}


def test_noisy_box_is_unusable_whatever_the_effect():
    """Spread alone condemns the box -- a property of the host, not the effect."""
    hard = [100, 130, 95, 140, 90, 135, 92]        # ~40% swings
    soft = [160, 120, 175, 118, 168, 122, 170]
    v = judge(_point(hard, soft))
    assert v["verdict"] == "UNUSABLE", v
    assert v["spread_pct"] > 2.0


def test_clean_box_with_a_large_effect_resolves():
    hard = [219.0, 219.4, 218.8, 219.2, 219.1, 219.3, 218.9]
    soft = [237.0, 237.4, 236.8, 237.2, 237.1, 237.3, 236.9]   # ~+8%
    v = judge(_point(hard, soft))
    assert v["verdict"] == "RESOLVED", v
    assert v["residual_pts"] > 5


# The real leading control from the demote-heap run -- the measurement the old
# gate refused. Synthetic arms were tried first and were too clean to reproduce
# it: at 0.18% IQR a 1.55% effect legitimately RESOLVES, so the fixture proved
# nothing. Real data or no test.
_REAL_HARD = [228965.3, 218674.6, 219103.3, 219012.8, 218626.4,
              219358.1, 218527.6, 220322.7, 218604.7]
_REAL_SOFT = [228073.9, 222485.5, 227330.1, 222421.6, 222348.0,
              222468.8, 222186.9, 222248.3, 223659.3]


def test_clean_box_with_a_shrunken_effect_is_bounded_not_refused():
    """The case the old gate got wrong: tight arms, small effect."""
    v = judge(_point(_REAL_HARD, _REAL_SOFT))
    assert v["verdict"] == "BELOW-RES", v
    assert v["spread_pct"] < 2.0, "this box is clean; it must not be UNUSABLE"
    lo, hi = v["residual_ci95_pts"]
    assert hi < 5.0, f"CI must exclude the pre-optimisation 5.61 pts, got {hi}"


def test_bootstrap_ci_is_tighter_than_the_single_run_resolution():
    """Per-run IQR is the wrong error bar for a 9-repeat median, and using it
    put a 0.22 pt residual behind a 2.96 pt bound."""
    v = judge(_point(_REAL_HARD, _REAL_SOFT))
    lo, hi = v["residual_ci95_pts"]
    assert (hi - lo) < v["resolution_pct"], (
        f"CI width {hi - lo:.2f} should beat single-run resolution "
        f"{v['resolution_pct']:.2f}")
