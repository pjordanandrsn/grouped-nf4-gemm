"""Tests for the PREREG-routed-residual verdict logic.

`evaluate` encodes **R5** — whether the expert-major coalescer gets built. That
decision is registered ahead of the data precisely so it cannot be chosen after
seeing it, which only works if the function that applies it is correct. A
decision function that has only ever run on a rented pod is a decision function
nobody has tested, so these run on the laptop with no torch and no GPU.

    python -m pytest bench/context/test_routed_residual_verdicts.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routed_residual_verdicts import evaluate  # noqa: E402

IDS = [10, 20, 30]


def rec(arm, s, *, ids=None, host=94, device=0, gbps=None):
    if arm == "C" and host == 94 and device == 0:      # control runs the device path
        host, device = 0, 94
    return {"arm": arm, "rep": 0, "s_per_token": s, "greedy_ids": ids or list(IDS),
            "counts": {"host": host, "device": device}, "routed_gbps": gbps}


def _clean(ratio=0.97, gbps=10.0):
    """A well-formed run: gates pass, T1 in band, R4 holds."""
    return [rec("C", 1.00, gbps=gbps), rec("C", 1.00, gbps=gbps),
            rec("T1", ratio, host=94, device=0), rec("T1", ratio, host=94, device=0)]


def test_clean_run_passes_everything():
    v = evaluate(_clean(), ceiling_gbps=22.21)
    assert v["gates_passed"]
    assert v["registered"]["R1_bit_identity"]["pass"]
    assert v["registered"]["R2_engagement"]["pass"]
    assert v["registered"]["R6_t1_magnitude"]["pass"]


def test_missing_registered_arm_is_an_error_not_a_verdict():
    v = evaluate([rec("C", 1.0), rec("C", 1.0)], ceiling_gbps=22.21)
    assert "error" in v and "T1" in v["error"]


# --- R1 -------------------------------------------------------------------
def test_divergent_greedy_ids_fail_the_gate():
    r = _clean()
    r[2]["greedy_ids"] = [10, 20, 31]
    v = evaluate(r, ceiling_gbps=22.21)
    assert not v["registered"]["R1_bit_identity"]["pass"]
    assert not v["gates_passed"]
    assert "STOP" in v["registered"]["R1_bit_identity"]["detail"]


# --- R2 -------------------------------------------------------------------
def test_fast_path_never_engaged_fails_even_though_device_is_zero():
    """The failure the counter exists to catch: nothing ran, so device==0 trivially."""
    r = _clean()
    for x in r:
        if x["arm"] == "T1":
            x["counts"] = {"host": 0, "device": 0}
    v = evaluate(r, ceiling_gbps=22.21)
    assert not v["registered"]["R2_engagement"]["pass"]
    assert not v["gates_passed"]


def test_control_that_did_not_flip_fails_r2():
    """If C also took the host path the switch did not flip and the pair is void."""
    r = _clean()
    for x in r:
        if x["arm"] == "C":
            x["counts"] = {"host": 94, "device": 0}
    v = evaluate(r, ceiling_gbps=22.21)
    assert not v["registered"]["R2_engagement"]["pass"]


def test_treatment_leaking_onto_the_device_path_fails_r2():
    r = _clean()
    r[2]["counts"] = {"host": 93, "device": 1}
    assert not evaluate(r, ceiling_gbps=22.21)["registered"]["R2_engagement"]["pass"]


# --- R4 / R5: the decision, both branches ---------------------------------
def test_r4_holds_means_build_the_coalescer():
    v = evaluate(_clean(gbps=10.0), ceiling_gbps=22.21)     # 0.45x ceiling
    r4, r5 = v["registered"]["R4_decomposition"], v["registered"]["R5_decision"]
    assert r4["pass"] and r4["fraction_of_ceiling"] < 0.70
    assert r5["build_expert_major_coalescer"] is True
    assert "BUILD" in r5["detail"]


def test_r4_falsified_means_do_not_build_it():
    v = evaluate(_clean(gbps=20.0), ceiling_gbps=22.21)     # 0.90x ceiling
    r4, r5 = v["registered"]["R4_decomposition"], v["registered"]["R5_decision"]
    assert not r4["pass"] and r4["fraction_of_ceiling"] > 0.70
    assert r5["build_expert_major_coalescer"] is False
    assert "DO NOT build" in r5["detail"]


def test_r4_bar_is_inclusive_at_exactly_0_70():
    v = evaluate(_clean(gbps=0.70 * 22.21), ceiling_gbps=22.21)
    assert v["registered"]["R4_decomposition"]["pass"] is True


def test_missing_stats_does_not_silently_decide():
    v = evaluate(_clean(gbps=None), ceiling_gbps=22.21)
    assert v["registered"]["R4_decomposition"]["pass"] is None
    assert v["registered"]["R5_decision"]["build_expert_major_coalescer"] is None


# --- R3 / R6 --------------------------------------------------------------
def test_regression_fails_r3_and_r6():
    v = evaluate(_clean(ratio=1.08), ceiling_gbps=22.21)
    assert not v["registered"]["R3_no_regression"]["pass"]
    assert v["registered"]["R6_t1_magnitude"]["verdict"] == "REGRESSION"


def test_below_band_is_a_miss_to_explain_not_a_win_to_claim():
    v = evaluate(_clean(ratio=0.60), ceiling_gbps=22.21)
    r6 = v["registered"]["R6_t1_magnitude"]
    assert not r6["pass"], "0.60 is outside the registered band and must not report as a pass"
    assert "BELOW BAND" in r6["verdict"] and "explain" in r6["verdict"]


@pytest.mark.parametrize("ratio,expected", [(0.95, True), (0.97, True), (1.00, True),
                                            (0.949, False), (1.001, False)])
def test_band_edges(ratio, expected):
    assert evaluate(_clean(ratio=ratio), ceiling_gbps=22.21)["registered"]["R6_t1_magnitude"]["pass"] is expected


def test_self_pair_spread_widens_the_no_regression_tolerance():
    """A noisy box must not turn ordinary jitter into a registered regression."""
    noisy = [rec("C", 1.00, gbps=10.0), rec("C", 1.10, gbps=10.0),
             rec("T1", 1.04, host=94), rec("T1", 1.04, host=94)]
    v = evaluate(noisy, ceiling_gbps=22.21)
    assert v["registered"]["R3_no_regression"]["self_pair_spread"] > 0.09
    assert v["registered"]["R3_no_regression"]["pass"]


# --- exploratory arms stay exploratory -------------------------------------
def test_exploratory_arms_are_labelled_and_kept_out_of_registered():
    r = _clean() + [rec("T1s", 0.99, host=94), rec("T1c", 0.98, host=0, device=94)]
    v = evaluate(r, ceiling_gbps=22.21)
    assert set(v["exploratory"]) == {"T1s", "T1c"}
    assert all("NOT REGISTERED" in x["note"] for x in v["exploratory"].values())
    assert not any(k.startswith("T1s") or k.startswith("T1c") for k in v["registered"])


# --- arm fidelity (harness gate, not a registered prediction) ---------------
def _with_row_plan(recs, c_plan=("dict",), t_plan=("flat",)):
    for x in recs:
        p = c_plan[0] if x["arm"] == "C" else t_plan[0]
        x["row_plan"] = {"flat": 94 if p == "flat" else 0, "dict": 94 if p == "dict" else 0}
    return recs


def test_arm_fidelity_passes_when_each_arm_ran_its_own_copy_loop():
    v = evaluate(_with_row_plan(_clean()), ceiling_gbps=22.21)
    assert v["arm_fidelity"]["pass"]
    assert v["gates_passed"]


def test_arm_leak_fails_the_run_even_though_bytes_are_identical():
    """Control silently running the treatment's copy loop. Nothing else detects it."""
    v = evaluate(_with_row_plan(_clean(), c_plan=("flat",)), ceiling_gbps=22.21)
    assert v["arm_fidelity"]["pass"] is False
    assert "ARM LEAK" in v["arm_fidelity"]["detail"]
    assert not v["gates_passed"]


def test_absent_row_plan_counts_do_not_fail_the_run():
    v = evaluate(_clean(), ceiling_gbps=22.21)
    assert v["arm_fidelity"]["pass"] is None
    assert v["gates_passed"]
