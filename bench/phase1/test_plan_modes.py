"""The e2e driver's data-mode schedule, pinned as a pure function.

Ten consecutive runs across three hosts drifted the FIRST data mode's reference
self-pair below G1 (0.859–0.924) while the second mode stayed clean — a driver
property (it persisted with zero CPU neighbours; wall-clock warm-up made it
worse; the drift spans the whole first mode, so a short burn-in is falsified).
The registered remedy is a discarded first pass of the first mode, and this
test pins the schedule that implements it so a refactor cannot silently drop
the burn-in or discard the wrong pass.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
spec = importlib.util.spec_from_file_location("e2e", _HERE / "e2e_train_arms.py")
e2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2e)


def test_both_with_discard_burns_first_mode_once():
    assert e2e.plan_modes("both", True) == [
        ("text", True), ("text", False), ("random", False)]


def test_single_modes_with_discard():
    assert e2e.plan_modes("text", True) == [("text", True), ("text", False)]
    assert e2e.plan_modes("random", True) == [("random", True), ("random", False)]


def test_discard_off_restores_single_pass():
    assert e2e.plan_modes("both", False) == [("text", False), ("random", False)]
    assert e2e.plan_modes("text", False) == [("text", False)]


def test_every_mode_still_gets_a_graded_pass():
    for data in ("both", "text", "random"):
        plan = e2e.plan_modes(data, True)
        graded = [m for m, burn in plan if not burn]
        expect = ["text", "random"] if data == "both" else [data]
        assert graded == expect, (data, plan)
        assert sum(1 for _, b in plan if b) == 1, "exactly one discarded pass"
        assert plan[0][1] is True, "the discarded pass must run FIRST"
