#!/usr/bin/env python3
"""Tests for the routing-faithful fixture.

The fixture's whole job is to stop lying about occupancy and skew, so the tests
check it against the operator's measured datum and against the histogram it was
built from — not merely that it runs.

CPU only; no torch device needed for the sizing logic.
"""
from __future__ import annotations

import importlib.util as _iu
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = _iu.spec_from_file_location(
    "rf", _ROOT / "bench" / "phase1" / "routing_fixture.py")
rf = _iu.module_from_spec(_spec)
_spec.loader.exec_module(rf)

RESULTS = _ROOT / "bench" / "phase1" / "results"


def _olmoe():
    d = json.loads((RESULTS / "routing_olmoe.json").read_text())
    return d["per_layer_counts"], d["E"], d["k"]


def test_matches_the_operator_datum_at_training_shape():
    """Operator: a single OLMoE forward at training shape touches a median 57 of
    64 experts (89%), max 63 (98%). Independent work put that at T ~ 30 tokens.
    The fixture must land there, not at 8 and not at 64."""
    per_layer, E, k = _olmoe()
    occ = []
    for li, counts in enumerate(per_layer):
        s = rf.sample_group_sizes(counts, tokens=30, top_k=k, seed=li)
        occ.append(len(s) / E)
    occ.sort()
    med = occ[len(occ) // 2]
    assert 0.82 <= med <= 0.97, f"median occupancy {med:.3f} at T=30"
    assert max(occ) <= 1.0


def test_total_rows_is_exactly_tokens_times_top_k():
    """The correction must not change the amount of WORK, only its
    distribution — otherwise it is a different problem, not a fairer fixture."""
    per_layer, E, k = _olmoe()
    for T in (8, 32, 128, 2048):
        s = rf.sample_group_sizes(per_layer[0], tokens=T, top_k=k, seed=1)
        assert sum(s.values()) == T * k, (T, sum(s.values()))


def test_reproduces_the_measured_skew_not_a_uniform_one():
    """Uniform counts are the fiction being removed. At the histogram's own
    token count the drawn spread must look like the measured one (cv ~0.5), not
    like uniform (cv ~0)."""
    per_layer, E, k = _olmoe()
    counts = per_layer[len(per_layer) // 2]
    s = rf.sample_group_sizes(counts, tokens=2048, top_k=k, seed=3)
    rows = list(s.values())
    mean = sum(rows) / len(rows)
    cv = (sum((x - mean) ** 2 for x in rows) / len(rows)) ** 0.5 / mean
    measured_mean = sum(counts) / len(counts)
    measured_cv = (sum((c - measured_mean) ** 2 for c in counts)
                   / len(counts)) ** 0.5 / measured_mean
    assert cv > 0.25, f"drawn cv {cv:.3f} is too uniform"
    assert abs(cv - measured_cv) < 0.25, f"drawn {cv:.3f} vs measured {measured_cv:.3f}"


def test_occupancy_saturates_at_large_token_counts():
    """At T >= 512 the measured occupancy is 1.000, which is why the
    token-budget regimes were right about occupancy and wrong only about skew.
    The fixture must agree, or it would 'correct' something that was fine."""
    per_layer, E, k = _olmoe()
    s = rf.sample_group_sizes(per_layer[0], tokens=2048, top_k=k, seed=5)
    assert len(s) == E


def test_seeded_and_reproducible():
    per_layer, E, k = _olmoe()
    a = rf.sample_group_sizes(per_layer[0], 64, k, seed=11)
    b = rf.sample_group_sizes(per_layer[0], 64, k, seed=11)
    c = rf.sample_group_sizes(per_layer[0], 64, k, seed=12)
    assert a == b and a != c


def test_models_without_measured_routing_RAISE_rather_than_invent():
    """Gemma-4 was gated and GPT-OSS needed an 80 GB card, so neither has
    measured routing. The fixture must refuse, so a caller records NOT-RUN
    instead of quietly substituting another model's routing."""
    assert rf.routing_for("google/gemma-4-26B-A4B", RESULTS) is None
    assert rf.routing_for("openai/gpt-oss-120b", RESULTS) is None
    assert rf.routing_for("allenai/OLMoE-1B-7B-0924", RESULTS) is not None
    assert rf.routing_for("Qwen/Qwen3-30B-A3B", RESULTS) is not None


def test_the_fixture_changes_the_dequant_call_count_it_is_meant_to():
    """The point of the exercise: at T=32 the baseline goes from top_k dequant
    calls to ~58, on the same total rows."""
    per_layer, E, k = _olmoe()
    s = rf.sample_group_sizes(per_layer[len(per_layer) // 2], 32, k, seed=0)
    assert len(s) > 5 * k, f"{len(s)} hit experts vs top_k={k}"
    assert sum(s.values()) == 32 * k


if __name__ == "__main__":  # pragma: no cover
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
