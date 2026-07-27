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
    r = {"arm": arm, "rep": 0, "s_per_token": s, "greedy_ids": ids or list(IDS),
         "counts": {"host": host, "device": device}, "routed_gbps": gbps}
    if gbps is not None:
        # sound-numerator fields: bytes over a 1.0 s decode wall == gbps GB/s.
        # routed_gbps stays as the stats-reported diagnostic.
        r["by_policy"] = {"routed": {"bytes": gbps * 1e9, "stages": 94}}
        r["decode_wall_s"] = 1.0
    return r


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


# --- interleave / position balance ------------------------------------------
from routed_residual_verdicts import interleave, position_balance  # noqa: E402


def _sums(order):
    s = {}
    for i, a in enumerate(order):
        s[a] = s.get(a, 0) + i
    return s


@pytest.mark.parametrize("n_arms", [2, 3, 4, 5])
@pytest.mark.parametrize("reps", [2, 4, 6])
def test_even_reps_balance_positions_exactly(n_arms, reps):
    arms = [f"a{i}" for i in range(n_arms)]
    assert len(set(_sums(interleave(arms, reps)).values())) == 1


def test_abba_is_the_actual_order():
    assert interleave(["C", "T1", "T1s", "T1c"], 2) == \
        ["C", "T1", "T1s", "T1c", "T1c", "T1s", "T1", "C"]


def test_plain_repetition_would_not_balance():
    """The bug this replaces: every arm pinned to one position every rep."""
    plain = ["C", "T1", "T1s", "T1c"] * 2
    assert len(set(_sums(plain).values())) > 1
    assert _sums(plain)["C"] < _sums(plain)["T1c"]      # C systematically earliest


def test_every_arm_still_runs_the_requested_number_of_times():
    order = interleave(["C", "T1", "T1s"], 4)
    assert all(order.count(a) == 4 for a in ("C", "T1", "T1s"))
    assert len(order) == 12


def test_odd_reps_are_reported_unbalanced_not_silently_accepted():
    order = interleave(["C", "T1"], 3)
    recs = [{"arm": a, "position": i} for i, a in enumerate(order)]
    pb = position_balance(recs)
    assert pb["balanced"] is False and "UNBALANCED" in pb["detail"]


def test_balance_flows_into_the_receipt_and_caveats_r6():
    unbal = _clean()
    for i, r in enumerate(unbal):
        r["position"] = i                     # C at 0,1 and T1 at 2,3 -> unbalanced
    v = evaluate(unbal, ceiling_gbps=22.21)
    assert v["position_balance"]["balanced"] is False
    assert "CAVEAT" in v["registered"]["R6_t1_magnitude"]["verdict"]


def test_balanced_positions_add_no_caveat():
    bal = _clean()
    for r, p in zip(bal, [0, 3, 1, 2]):       # C at 0,3 / T1 at 1,2 -> both sum 3
        r["position"] = p
    v = evaluate(bal, ceiling_gbps=22.21)
    assert v["position_balance"]["balanced"] is True
    assert "CAVEAT" not in v["registered"]["R6_t1_magnitude"]["verdict"]


# --- power: a noisy run must not print a verdict it cannot support ----------
def test_r6_reports_underpowered_when_noise_exceeds_the_band():
    noisy = [rec("C", 1.00, gbps=10.0), rec("C", 1.08, gbps=10.0),
             rec("T1", 1.04, host=94), rec("T1", 1.04, host=94)]
    v = evaluate(noisy, ceiling_gbps=22.21)
    r6 = v["registered"]["R6_t1_magnitude"]
    assert r6.get("underpowered") is True
    assert r6["pass"] is None, "an unresolvable band must not report pass OR fail"
    assert "UNDERPOWERED" in r6["verdict"]


def test_a_quiet_run_still_gets_a_real_verdict():
    v = evaluate(_clean(ratio=0.97), ceiling_gbps=22.21)      # spread 0.0
    r6 = v["registered"]["R6_t1_magnitude"]
    assert not r6.get("underpowered")
    assert r6["pass"] is True and "IN BAND" in r6["verdict"]


def test_underpowered_does_not_mask_a_real_regression_signal():
    """Nominal ratio is still reported so it can be inspected, just not as a verdict."""
    noisy = [rec("C", 1.00, gbps=10.0), rec("C", 1.09, gbps=10.0),
             rec("T1", 1.30, host=94), rec("T1", 1.30, host=94)]
    r6 = evaluate(noisy, ceiling_gbps=22.21)["registered"]["R6_t1_magnitude"]
    assert r6["ratio"] > 1.2 and r6["pass"] is None


# --- R4 plausibility gate --------------------------------------------------
def test_achieving_more_than_the_link_ceiling_is_invalid_not_a_failure():
    """The 2026-07-27 235B run: 22.83 GB/s against a probed 14.68 ceiling.

    Un-gated, this emitted R4=fail and R5='DO NOT build the coalescer' -- a
    confident registered negative produced by a contended ceiling probe.
    """
    v = evaluate(_clean(gbps=22.83), ceiling_gbps=14.68)
    r4, r5 = v["registered"]["R4_decomposition"], v["registered"]["R5_decision"]
    assert r4["pass"] is None, "must be None (cannot tell), never False (prediction failed)"
    assert "MEASUREMENT INVALID" in r4["detail"]
    assert r5["build_expert_major_coalescer"] is None
    assert not v["gates_passed"]


def test_a_plausible_fraction_still_adjudicates_normally():
    """With the real ~26 GB/s ceiling the same numerator is fine and R4 decides."""
    v = evaluate(_clean(gbps=22.83), ceiling_gbps=26.0)
    r4 = v["registered"]["R4_decomposition"]
    assert r4["pass"] is False and 0.87 < r4["fraction_of_ceiling"] < 0.89
    assert v["registered"]["R5_decision"]["build_expert_major_coalescer"] is False


def test_exactly_at_the_ceiling_is_allowed():
    v = evaluate(_clean(gbps=22.0), ceiling_gbps=22.0)
    assert v["registered"]["R4_decomposition"]["pass"] is False   # 1.0 <= 1.0, adjudicates


# --- prefill separation -----------------------------------------------------
def test_missing_prefill_block_is_flagged_as_unconfirmable():
    """Old receipts cannot prove prefill was reset away, so R4 may be contaminated."""
    v = evaluate(_clean(), ceiling_gbps=26.0)
    assert v["prefill_separated"]["pass"] is None
    assert "contaminated" in v["prefill_separated"]["detail"]


def test_prefill_block_present_on_every_record_passes():
    r = _clean()
    for x in r:
        x["prefill"] = {"by_policy": {"routed": {"gbps": 24.0, "stages": 94}}}
    v = evaluate(r, ceiling_gbps=26.0)
    assert v["prefill_separated"]["pass"] is True
    assert v["gates_passed"]


def test_partial_prefill_coverage_fails_the_run():
    r = _clean()
    r[0]["prefill"] = {"by_policy": {}}
    v = evaluate(r, ceiling_gbps=26.0)
    assert v["prefill_separated"]["pass"] is False
    assert not v["gates_passed"]



# --- sound numerator (PREREG-2 amendment-1) ---------------------------------
def test_r4_uses_the_sound_numerator_not_the_stats_field():
    """The stats-reported gbps must never adjudicate: build-dependent divisor."""
    r = _clean(gbps=10.0)
    for x in r:
        if x["arm"] == "C":
            x["routed_gbps"] = 99.0            # poisoned stats field
    v = evaluate(r, ceiling_gbps=22.21)
    r4 = v["registered"]["R4_decomposition"]
    assert abs(r4["routed_gbps_sound"] - 10.0) < 1e-9
    assert r4["pass"] is True                  # 10/22.21 = 0.45 <= 0.70
    assert r4["stats_reported_gbps"] == 99.0   # carried, as diagnostic only


def test_campaign_1_shaped_receipt_refuses_to_adjudicate():
    """routed_gbps present, sound fields absent — exactly the 2026-07-27 receipt."""
    r = _clean(gbps=8.8)
    for x in r:
        x.pop("by_policy", None); x.pop("decode_wall_s", None)
    v = evaluate(r, ceiling_gbps=14.68)
    r4 = v["registered"]["R4_decomposition"]
    assert r4["pass"] is None
    assert "build-dependent" in r4["detail"]
    assert v["registered"]["R5_decision"]["build_expert_major_coalescer"] is None


def test_sound_wall_divisor_is_the_wall_not_the_step_median():
    """bytes/decode_wall, exactly — a wrong divisor shifts the value measurably."""
    r = _clean(gbps=10.0)
    for x in r:
        if x["arm"] == "C":
            x["decode_wall_s"] = 2.0           # same bytes, doubled wall
    v = evaluate(r, ceiling_gbps=22.21)
    assert abs(v["registered"]["R4_decomposition"]["routed_gbps_sound"] - 5.0) < 1e-9


# --- ceiling-range robustness ------------------------------------------------
def test_verdict_robust_across_range_adjudicates():
    v = evaluate(_clean(gbps=8.8), ceiling_gbps=18.46, ceiling_range=(12.65, 18.46))
    r4 = v["registered"]["R4_decomposition"]
    assert r4["pass"] is True                  # 8.8/12.65=0.696 and 8.8/18.46=0.48 both hold
    assert r4["robust_across_ceiling_range"] is True


def test_verdict_flipping_across_range_yields_no_verdict():
    v = evaluate(_clean(gbps=8.8), ceiling_gbps=18.0, ceiling_range=(11.0, 18.0))
    r4 = v["registered"]["R4_decomposition"]   # 8.8/11=0.80 fails, 8.8/18=0.49 holds
    assert r4["pass"] is None
    assert "NOT ROBUST" in r4["detail"]
    assert not v["gates_passed"]
    assert v["registered"]["R5_decision"]["build_expert_major_coalescer"] is None


def test_no_range_given_still_adjudicates():
    v = evaluate(_clean(gbps=8.8), ceiling_gbps=14.68)
    assert v["registered"]["R4_decomposition"]["pass"] is True   # 0.60
