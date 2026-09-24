# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The deadline cost model: does it produce gate 2's receipt at all?

The gate asks for a case where the GPU is intrinsically faster for an expert
yet the CPU delivers its contribution sooner because the GPU is already
committed. A model that cannot express that is a threshold with extra
arithmetic, so the load-bearing test here is that the flip EXISTS and is
driven by backlog rather than by shape.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from cold_deadline import Costs, Decision, choose, cpu_us, gpu_us  # noqa: E402

# OLMoE-1B-7B-ish: 3.54 MB per expert; constants in the band the serving
# playbook ships (cpu_us_fixed=55, cpu_us_per_row=2) and the Zen5+5090
# calibration measured.
C = Costs(cpu_us_fixed=55.0, cpu_us_per_row=2.0, b_dram_gbs=380.1,
          b_vram_gbs=1574.2, b_link_gbs=28.47, bytes_per_expert=3538944)


# ------------------------------------------------------- the gate-2 shape --

def test_backlog_can_flip_a_destination_the_isolated_comparison_would_not():
    """Gate 2's receipt, in the model: the GPU is faster for this shape in
    isolation, yet the CPU delivers sooner because the GPU is committed.
    Without this the model cannot express what the gate exists to find."""
    rows, uniq = 256, 2                      # many rows per expert: GPU's regime
    assert gpu_us(rows, uniq, C) < cpu_us(rows, uniq, C), (
        "this shape must favour the GPU in isolation")
    d = choose(rows, uniq, C, gpu_backlog_us=5000.0)
    assert d.dest == "cpu"
    assert d.flipped_by_backlog is True


def test_without_backlog_the_choice_is_isolated_speed():
    d = choose(256, 2, C)
    assert d.dest == "gpu" and d.flipped_by_backlog is False


def test_cpu_backlog_pushes_work_to_the_gpu_too():
    """The flip has to work in both directions or it is a bias, not a
    scheduler."""
    rows, uniq = 8, 4                        # few rows per expert: CPU's regime
    assert cpu_us(rows, uniq, C) < gpu_us(rows, uniq, C), "CPU wins solo here"
    d = choose(rows, uniq, C, cpu_backlog_us=50_000.0)
    assert d.dest == "gpu" and d.flipped_by_backlog is True


# ------------------------------------------------------------ the shapes --

def test_cpu_scales_with_rows_and_gpu_does_not():
    """The crossover the whole design turns on: the CPU term carries a
    per-row slope, the GPU term is one H2D per unique expert and flat in
    rows. If this ever stops holding, the destination question changes
    character."""
    g1, g2 = gpu_us(1, 4, C), gpu_us(64, 4, C)
    assert g1 == g2, "GPU cost must not depend on row count at fixed uniq"
    assert cpu_us(64, 4, C) > cpu_us(1, 4, C)
    # the slope is the per-row constant, exactly
    assert cpu_us(64, 4, C) - cpu_us(1, 4, C) == pytest.approx(
        63 * C.cpu_us_per_row)


def test_the_cpu_floor_dominates_a_singleton_call():
    """The fixbox finding: at decode shapes the DRAM tier is per-call-floor
    bound, not bandwidth bound."""
    t = cpu_us(1, 1, C)
    assert t > C.cpu_us_fixed
    assert (C.cpu_us_fixed / t) > 0.5, "the floor should dominate at rows=1"


def test_crossover_moves_with_rows_per_expert():
    """Prediction 4, and it lands the way the SHIPPED heuristic already
    says: few rows per expert favours the CPU (its per-call floor is 55 us
    against a 124 us H2D for one 3.54 MB expert), many rows per expert
    favours the GPU (whose cost is flat in rows).

    That the deadline model reproduces `offload_rows`' direction from
    measured constants alone — without being told about it — is a check on
    the model, not a coincidence to skip past.
    """
    assert choose(2, 2, C).dest == "cpu"
    assert choose(256, 2, C).dest == "gpu"


# ---------------------------------------------------------- the contract --

def test_empty_group_costs_nothing_on_either_engine():
    assert cpu_us(0, 0, C) == 0.0 and gpu_us(0, 0, C) == 0.0


def test_ties_go_to_the_gpu_so_a_balanced_prediction_changes_nothing():
    """GPU is the pre-Stage-3 destination; an exactly-balanced estimate
    must not churn."""
    d = choose(4, 2, C, cpu_backlog_us=gpu_us(4, 2, C) - cpu_us(4, 2, C))
    assert d.cpu_join_us == pytest.approx(d.gpu_join_us)
    assert d.dest == "gpu"


def test_negative_backlog_is_a_named_error():
    with pytest.raises(ValueError, match="backlog"):
        choose(4, 2, C, cpu_backlog_us=-1.0)


def test_the_decision_records_both_predictions_not_just_the_winner():
    """A scheduler that logs only its choice cannot be scored. This is the
    prereg's falsifiability hook."""
    r = choose(8, 4, C, gpu_backlog_us=5000.0).record()
    assert set(r) == {"dest", "cpu_join_us", "gpu_join_us", "margin_us",
                      "flipped_by_backlog"}
    assert r["cpu_join_us"] > 0 and r["gpu_join_us"] > 0


# ------------------------------------------------------- the blob reader --

def _blob(single=18.2):
    link = {"h2d_64mb": {"gbs": 28.47}}
    if single is not None:
        link["h2d_64mb_single"] = {"gbs": single}
    return {"gpu_bench": {"devices": [
                {"b_vram_triad_gbs": 1574.2, "b_link": link}]},
            "cpu_bench": {"triad_best": {"gbs": 380.1}}}


def test_costs_read_the_blobs_own_field_names():
    c = Costs.from_blob(_blob(), cpu_us_fixed=55.0, cpu_us_per_row=2.0,
                        bytes_per_expert=3538944)
    assert c.b_vram_gbs == 1574.2 and c.b_link_gbs == 28.47
    assert c.b_dram_gbs == 380.1
    assert c.link_eff == pytest.approx(18.2 / 28.47) and c.gpu_us_fixed == 0.0


def test_an_old_blob_without_the_single_copy_probe_raises_unless_link_eff_is_passed():
    """#400: a blob from before gnf4-hybrid-calib/2 carries no measured
    efficiency. Defaulting it to 1.0 would silently restore the model P66
    refuted, so the caller must say so."""
    with pytest.raises(KeyError, match="h2d_64mb_single"):
        Costs.from_blob(_blob(single=None), cpu_us_fixed=55.0,
                        cpu_us_per_row=2.0, bytes_per_expert=1)
    c = Costs.from_blob(_blob(single=None), cpu_us_fixed=55.0,
                        cpu_us_per_row=2.0, bytes_per_expert=1, link_eff=1.0)
    assert c.link_eff == 1.0


def test_link_eff_is_capped_at_one_and_validated():
    c = Costs.from_blob(_blob(single=40.0), cpu_us_fixed=55.0,
                        cpu_us_per_row=2.0, bytes_per_expert=1)
    assert c.link_eff == 1.0                       # a single copy cannot beat the DMA loop
    with pytest.raises(ValueError, match="link_eff"):
        Costs.from_blob(_blob(single=None), cpu_us_fixed=55.0,
                        cpu_us_per_row=2.0, bytes_per_expert=1, link_eff=0.0)
    with pytest.raises(ValueError, match="gpu_us_fixed"):
        Costs.from_blob(_blob(), cpu_us_fixed=55.0, cpu_us_per_row=2.0,
                        bytes_per_expert=1, gpu_us_fixed=-1.0)


def test_gpu_us_scales_the_link_by_its_efficiency_and_adds_the_fixed_term_once():
    """#400, from lane P66: the link term is bytes over b_link * link_eff;
    the fixed term is per call, never per row or per expert."""
    base = C._replace(link_eff=1.0, gpu_us_fixed=0.0)
    half = C._replace(link_eff=0.5, gpu_us_fixed=0.0)
    fixed = C._replace(link_eff=1.0, gpu_us_fixed=24.7)
    gb = 4 * C.bytes_per_expert / 1e9
    link_us = gb / C.b_link_gbs * 1e6
    assert gpu_us(1, 4, half) - gpu_us(1, 4, base) == pytest.approx(link_us)   # the link term doubled
    assert gpu_us(1, 4, fixed) - gpu_us(1, 4, base) == pytest.approx(24.7)
    assert gpu_us(16, 4, fixed) - gpu_us(1, 4, fixed) == pytest.approx(0.0)    # still flat in rows
    assert gpu_us(1, 8, fixed) - gpu_us(1, 4, fixed) == pytest.approx(
        gpu_us(1, 8, base) - gpu_us(1, 4, base))                             # fixed does not scale with uniq
    assert gpu_us(0, 4, fixed) == 0.0                                          # an empty group is free


def test_the_defaults_reproduce_the_bytes_only_model():
    """A Costs built without the #400 fields is the pre-#400 model to the
    bit, so every consumer that constructs Costs by keyword keeps its
    numbers until it measures its own."""
    assert C.link_eff == 1.0 and C.gpu_us_fixed == 0.0
    gb = 4 * C.bytes_per_expert / 1e9
    assert gpu_us(1, 4, C) == pytest.approx((gb / C.b_link_gbs + gb / C.b_vram_gbs) * 1e6)


def test_a_blob_with_no_gpu_is_an_error_not_a_default():
    """A silent fallback would put a guessed ceiling into a scheduling
    decision."""
    bad = _blob()
    bad["gpu_bench"]["devices"] = []
    with pytest.raises(ValueError, match="no GPU device"):
        Costs.from_blob(bad, cpu_us_fixed=55.0, cpu_us_per_row=2.0,
                        bytes_per_expert=1)


def test_a_missing_ceiling_raises_rather_than_defaulting():
    bad = _blob()
    del bad["cpu_bench"]["triad_best"]
    with pytest.raises(KeyError):
        Costs.from_blob(bad, cpu_us_fixed=55.0, cpu_us_per_row=2.0,
                        bytes_per_expert=1)


def test_decision_is_a_plain_value():
    assert isinstance(choose(4, 2, C), Decision)


def test_first_to_finish_is_the_same_rule_as_minimising_the_layer_join():
    """`choose` picks the engine that delivers this group soonest. The
    directive's objective is different on its face — minimise
    max(T_cpu_side, T_gpu_side) for the whole layer — so the two could
    disagree, and if they did the model would be optimising the wrong
    thing.

    They do not: over the whole parameter space the rules coincide. (The
    proof is short — if adding the group to the CPU makes the CPU the max,
    it is because Cc+cc exceeds Gc, and choosing CPU means Cc+cc < Gc+gc,
    so the GPU assignment is at least as large; the symmetric argument runs
    the other way.) Pinned by search rather than left as an assertion.
    """
    import random
    rng = random.Random(20260820)

    def first(cc_b, cc, gc_b, gc):
        return "cpu" if cc_b + cc < gc_b + gc else "gpu"

    def join(cc_b, cc, gc_b, gc):
        return "cpu" if max(cc_b + cc, gc_b) < max(cc_b, gc_b + gc) else "gpu"

    for _ in range(20000):
        a = [rng.uniform(0, 1000) for _ in range(4)]
        assert first(*a) == join(*a), a
    # and the degenerate corners search rarely hits
    for a in ((0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1), (5, 5, 5, 5)):
        assert first(*a) == join(*a), a


def test_the_committed_a2000_blob_reads_as_schema_2():
    """The first /2 blob (bench/cold-engine/calib-a2000-400/, the NAS A2000, gen 3 x8): the single-copy probe is
    present, from_blob needs no explicit link_eff, and on that link the two rates agree so the factor is 1.0."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "bench", "cold-engine", "calib-a2000-400", "calib.json")
    blob = json.load(open(path))
    assert blob["schema"] == "gnf4-hybrid-calib/2"
    link = blob["gpu_bench"]["devices"][0]["b_link"]
    assert link["h2d_64mb_single"]["reps"] == 10 and link["h2d_64mb_single"]["gbs"] > 0
    c = Costs.from_blob({"gpu_bench": blob["gpu_bench"],
                         "cpu_bench": {"triad_best": {"gbs": 22.35}}},   # CPU benches were skipped on that run
                        cpu_us_fixed=55.0, cpu_us_per_row=2.0, bytes_per_expert=3538944)
    assert c.link_eff == 1.0 and c.b_link_gbs == link["h2d_64mb"]["gbs"]


def test_the_committed_5090_blob_reads_as_schema_2():
    """Lane P69's receipt (bench/cold-engine/calib-5090-p69/, a gen 4 x16 RTX 5090): from_blob needs no explicit
    link_eff and reads 0.873 on run 1 -- a per-host figure, not the 0.64 lane P66's other 5090 host gave."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "bench", "cold-engine", "calib-5090-p69", "run1", "calib.json")
    blob = json.load(open(path))
    assert blob["schema"] == "gnf4-hybrid-calib/2"
    c = Costs.from_blob({"gpu_bench": blob["gpu_bench"], "cpu_bench": {"triad_best": {"gbs": 1.0}}},
                        cpu_us_fixed=55.0, cpu_us_per_row=2.0, bytes_per_expert=2_654_208)
    assert c.b_link_gbs == 20.58 and c.link_eff == pytest.approx(17.97 / 20.58)
    # the link term the consumer sees: 14 % longer than the bytes-over-b_link model on this host
    assert gpu_us(1, 1, c) / gpu_us(1, 1, c._replace(link_eff=1.0)) == pytest.approx(1.143, abs=0.003)
