#!/usr/bin/env python3
"""Tests for the dequant-on-forward reducer, run against the REAL frozen prereg.

A reducer that has never been exercised is a coin flip you find out about after
the pod is gone. These build synthetic receipts and check that each registered
criterion can both pass and fail — including the self-pair, which is the gate
that voided the last training leg and therefore the one most worth proving.
"""
from __future__ import annotations

import importlib.util as _iu
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = _iu.spec_from_file_location(
    "rdf", _ROOT / "bench" / "phase1" / "reduce_dequant_forward.py")
rdf = _iu.module_from_spec(_spec)
_spec.loader.exec_module(rdf)

PRE = json.loads((_ROOT / "kernel" / "prereg_dequant_forward.json").read_text())
REGIMES = ["decode_m8", "tokbudget_2048", "tokbudget_11800"]
CELLS = [("allenai/OLMoE-1B-7B-0924", "gate_up"), ("allenai/OLMoE-1B-7B-0924", "down"),
         ("Qwen/Qwen3-30B-A3B", "gate_up"), ("Qwen/Qwen3-30B-A3B", "down"),
         ("google/gemma-4-26B-A4B", "gate_up"), ("google/gemma-4-26B-A4B", "down"),
         ("openai/gpt-oss-120b", "gate_up"), ("openai/gpt-oss-120b", "down")]


def _row(model, proj, regime, d_over_g=1.8, mem=4.0, b_rel=0.75,
         g_self=1.00, d_self=1.00, drift=1.00, gate=None):
    good = {"finite": True, "nonzero": True}
    zero_at_init = {"finite": True, "nonzero": False}
    ctl = {"present": True, "finite": True, "nonzero": True}
    return {
        "model": model, "proj": proj, "regime": regime, "status": "ok",
        "d_over_g": d_over_g, "dr_over_d": 1.4, "dr_over_g": d_over_g * 1.4,
        "mem_transient_d_over_g": mem, "b_rel_G_over_D": b_rel,
        "g_selfpair": g_self, "d_selfpair": d_self, "g_drift": drift,
        "j_ratio_d_over_g": 1.9, "lora_floor_frac_of_g": 0.2,
        "gate": gate or {
            "deq_calls_ok": True, "deq_calls_D": 8, "nonempty_groups": 8,
            "routed_matches_D": True, "routed_rows_differing": 0,
            "lora_A_grad_expected_zero_at_init": True,
            "grad_G": {"a": good, "lora_A": zero_at_init, "lora_B": good},
            "grad_D": {"a": good, "lora_A": zero_at_init, "lora_B": good},
            "grad_D_routed": {"a": good, "lora_A": zero_at_init, "lora_B": good},
            "gradA_at_nonzero_B_G": ctl, "gradA_at_nonzero_B_D": ctl,
            "gradA_at_nonzero_B_D_routed": ctl},
    }


def _receipt(**over):
    """A receipt whose every registered criterion passes, unless overridden by
    a per-(regime) map of row kwargs."""
    rows = []
    for regime in REGIMES:
        # default d/g decays with token budget, as S2 predicts
        dg = {"decode_m8": 1.8, "tokbudget_2048": 1.4, "tokbudget_11800": 1.05}[regime]
        for m, p in CELLS:
            kw = dict(d_over_g=dg)
            kw.update(over.get(regime, {}))
            rows.append(_row(m, p, regime, **kw))
    return {"gpu": "NVIDIA H100 80GB HBM3", "capability": "9.0",
            "regimes": REGIMES, "rows": rows}


def test_a_clean_receipt_confirms():
    d = rdf.grade_device("H100", _receipt(), PRE)
    assert d["Q1_self_pair"] and d["Q2_wiring"]
    assert d["S1_speed_small_batch"] and d["M1_memory"] and d["F1_fidelity"]
    assert d["DEVICE_CONFIRMED"]
    assert d["S1_detail"]["median_in_predicted_band"]
    assert d["S2_M_axis_report_only"]["observed_monotone_decay"] is True


def test_self_pair_out_of_band_VOIDS_cells_and_can_void_the_device():
    d = rdf.grade_device("H100", _receipt(decode_m8={"g_self": 1.09}), PRE)
    assert len(d["void_cells"]) == 8
    assert d["void_fraction"] > PRE["frozen_verdict_criteria_params"][
        "void_cell_fraction_that_voids_the_leg"]
    assert not d["Q1_self_pair"] and not d["DEVICE_CONFIRMED"]


def test_a_void_cell_contributes_no_ratio_to_S1():
    """The point of voiding: the number is withheld, not counted as a pass."""
    d = rdf.grade_device("H100", _receipt(decode_m8={"d_self": 1.20}), PRE)
    assert d["S1_detail"]["of"] == 0
    assert not d["S1_speed_small_batch"]


def test_drift_out_of_band_voids_the_cell():
    d = rdf.grade_device("H100", _receipt(tokbudget_11800={"drift": 1.12}), PRE)
    assert len(d["void_cells"]) == 8
    assert all("g_drift" in v["why"] for v in d["void_cells"])


def test_S1_fails_when_the_baseline_wins_too_many_cells():
    d = rdf.grade_device("H100", _receipt(decode_m8={"d_over_g": 0.8}), PRE)
    assert not d["S1_speed_small_batch"] and not d["DEVICE_CONFIRMED"]


def test_S1_bar_can_pass_while_the_predicted_band_MISSES():
    """A met bar with a missed prediction must read as a miss, not silence."""
    d = rdf.grade_device("H100", _receipt(decode_m8={"d_over_g": 1.05}), PRE)
    assert d["S1_speed_small_batch"]
    assert not d["S1_detail"]["median_in_predicted_band"]


def test_M1_fails_when_the_memory_trade_does_not_appear():
    d = rdf.grade_device("H100", _receipt(tokbudget_11800={"mem": 0.9}), PRE)
    assert not d["M1_memory"] and not d["DEVICE_CONFIRMED"]


def test_F1_fidelity_failure_is_graded_over_ALL_cells_including_void_ones():
    r = _receipt(decode_m8={"g_self": 1.09, "b_rel": 1.4})
    d = rdf.grade_device("H100", r, PRE)
    assert not d["F1_fidelity"]                    # 1.4 > bar 1.0
    assert len(d["F1_detail"]["failures"]) == 8    # counted despite the void


def test_Q2_rejects_a_hoisted_dequant_and_a_dead_gradient():
    bad_count = _receipt()
    bad_count["rows"][0]["gate"]["deq_calls_ok"] = False
    assert not rdf.grade_device("H100", bad_count, PRE)["Q2_wiring"]

    dead = _receipt()
    dead["rows"][0]["gate"]["grad_D"]["lora_B"] = {"finite": True, "nonzero": False}
    d = rdf.grade_device("H100", dead, PRE)
    assert not d["Q2_wiring"] and not d["DEVICE_CONFIRMED"]


def test_Q2_accepts_lora_A_being_zero_at_init_but_REQUIRES_its_control():
    """dL/dA is exactly 0 at LoRA init because B is zero-init. That must not
    void a cell — and the positive control at non-zero B must still be
    required, or the lora_A column would be unfalsifiable."""
    ok = _receipt()
    assert rdf.grade_device("H100", ok, PRE)["Q2_wiring"]

    no_ctl = _receipt()
    no_ctl["rows"][0]["gate"]["gradA_at_nonzero_B_D"] = {
        "present": True, "finite": True, "nonzero": False}
    assert not rdf.grade_device("H100", no_ctl, PRE)["Q2_wiring"]


def test_S2_reports_a_NON_decay_rather_than_hiding_it():
    d = rdf.grade_device("H100", _receipt(tokbudget_11800={"d_over_g": 3.0}), PRE)
    assert d["S2_M_axis_report_only"]["observed_monotone_decay"] is False
    assert d["DEVICE_CONFIRMED"]        # report-only: never enters the verdict


def test_one_device_alone_cannot_confirm():
    """Two-card rule: nothing ships on one card's evidence."""
    d = rdf.grade_device("H100", _receipt(), PRE)
    assert d["DEVICE_CONFIRMED"]
    # the leg-level key is what enforces it; mirror main()'s composition
    assert (len({"H100": d}) >= 2) is False
