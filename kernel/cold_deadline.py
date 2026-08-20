# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage 3 / workstream 4 — time-to-contribution for a cold expert group.

The question a cold row poses is not "which engine is faster for this
expert" but "which engine delivers its contribution to the layer join
FIRST, given what each is already committed to". Those differ whenever one
engine is busier, which is the whole premise of gate 2:

    T_isolated_GPU(E) < T_isolated_CPU(E)   and   T_join_CPU(E) < T_join_GPU(E)

A destination rule that ignores backlog cannot produce that receipt, because
it has no term that changes when the machine gets busy. The shipped
`offload_rows`-style threshold is exactly such a rule: it reads the step's
routing shape and nothing about load.

**What this module is, and is not.** It is the cost function; the policy
lives with the executor (the placement rule this repo keeps: kernels and
calibration here, scheduling decisions in the runtime). Everything here is
pure arithmetic over measured constants — no torch, no I/O, no clock — so
the decision is testable without a model, a GPU, or a box.

**Every constant is measured, never a spec sheet.** `b_vram`, `b_link` and
`b_dram` come from the calibration blob; `cpu_us_fixed` and `cpu_us_per_row`
come from the in-situ rows-curve fit the placement solver already consumes.
The B=16 lesson is the reason the CPU term is `fixed + per_row * rows`
rather than bytes/bandwidth: on the DRAM tier, call cost is dominated by a
per-call floor and a row-count slope, and bytes alone do not predict it.

**The honest limits, stated here rather than discovered later.** This is a
first-order model. It does not represent queueing beyond a linear backlog
sum, PCIe contention between concurrent transfers, or the possibility that
CPU and GPU work genuinely overlap rather than serialize. Its predictions
are recorded alongside the outcome (see `Decision.record`) precisely so the
model can be scored against reality instead of trusted.
"""
from __future__ import annotations

from typing import NamedTuple


class Costs(NamedTuple):
    """Measured constants. Microseconds and GB/s, from the calibration blob
    and the rows-curve fit — never spec sheets."""
    cpu_us_fixed: float          # per-call floor on the DRAM/CPU tier
    cpu_us_per_row: float        # marginal cost per routed row
    b_dram_gbs: float            # grouped-scatter read bandwidth
    b_vram_gbs: float            # device triad
    b_link_gbs: float            # H2D at transfer size (64 MB figure)
    bytes_per_expert: int

    @classmethod
    def from_blob(cls, calib: dict, *, cpu_us_fixed: float,
                  cpu_us_per_row: float, bytes_per_expert: int) -> "Costs":
        """Read the ceilings out of a gnf4 calibration blob by its own field
        names. A missing field is an error, not a default: a silent fallback
        would put a guessed number into a scheduling decision."""
        dev = calib["gpu_bench"]["devices"]
        if not dev:
            raise ValueError("calibration blob has no GPU device entry")
        b_vram = max(d["b_vram_triad_gbs"] for d in dev)
        b_link = max(d["b_link"]["h2d_64mb"]["gbs"] for d in dev)
        b_dram = calib["cpu_bench"]["triad_best"]["gbs"]
        return cls(cpu_us_fixed=cpu_us_fixed, cpu_us_per_row=cpu_us_per_row,
                   b_dram_gbs=b_dram, b_vram_gbs=b_vram, b_link_gbs=b_link,
                   bytes_per_expert=int(bytes_per_expert))


def cpu_us(rows: int, uniq: int, c: Costs) -> float:
    """Time for the CPU tier to turn ``rows`` routed rows over ``uniq``
    experts into contributions.

    ``fixed + per_row * rows`` is the measured shape (the rows-curve fit),
    not bytes/bandwidth — the B=16 finding is that bytes do not predict CPU
    call cost. The weight-read term is added on top because a cold expert's
    bytes are not already in cache; it is small next to the compute term at
    decode shapes and is kept so the model does not silently ignore a
    real cost at large ``uniq``.
    """
    if rows <= 0:
        return 0.0
    read_us = (uniq * c.bytes_per_expert / 1e9) / max(c.b_dram_gbs, 1e-9) * 1e6
    return c.cpu_us_fixed + c.cpu_us_per_row * rows + read_us


def gpu_us(rows: int, uniq: int, c: Costs) -> float:
    """Time for the GPU to do the same, from host-resident packed bytes.

    Dominated by ONE H2D per unique expert — flat in rows, which is exactly
    why the two engines cross over as batch grows: the CPU term scales with
    rows and the GPU term does not. The device-side read is included at
    ``b_vram``; the kernel's own compute is deliberately not modelled,
    because at these shapes the path is transfer-bound and adding an
    unmeasured compute term would be inventing precision.
    """
    if rows <= 0:
        return 0.0
    gb = uniq * c.bytes_per_expert / 1e9
    return (gb / max(c.b_link_gbs, 1e-9) + gb / max(c.b_vram_gbs, 1e-9)) * 1e6


class Decision(NamedTuple):
    dest: str                    # "cpu" | "gpu"
    cpu_join_us: float           # when the CPU would deliver, incl. backlog
    gpu_join_us: float
    margin_us: float             # how much the winner won by
    flipped_by_backlog: bool     # would the isolated comparison disagree?

    def record(self) -> dict:
        """The counterfactual, for the receipt. A scheduler that logs only
        its choice cannot be scored; one that logs what it predicted for
        BOTH paths can be caught being wrong."""
        return {"dest": self.dest, "cpu_join_us": self.cpu_join_us,
                "gpu_join_us": self.gpu_join_us, "margin_us": self.margin_us,
                "flipped_by_backlog": self.flipped_by_backlog}


def choose(rows: int, uniq: int, c: Costs, *, cpu_backlog_us: float = 0.0,
           gpu_backlog_us: float = 0.0) -> Decision:
    """Pick the destination that delivers this cold group to the join first.

    Backlog is what each engine is ALREADY committed to this layer — the
    DRAM rows queued on the CPU tier, the resident-expert work queued on the
    GPU. Adding it is the whole difference between a deadline estimate and a
    threshold: without it the comparison is isolated speed, and isolated
    speed cannot flip when the machine gets busy.

    Ties go to the GPU, deliberately: it is the pre-Stage-3 destination, so
    an exactly-balanced prediction changes nothing.

    **This is also the layer-join rule**, which is not obvious. The
    directive's objective is ``min max(T_cpu_side, T_gpu_side)`` over the
    whole layer, not "which engine finishes this group first" — but the two
    coincide everywhere (pinned by
    ``test_first_to_finish_is_the_same_rule_as_minimising_the_layer_join``).
    If adding the group to the CPU makes the CPU the max, that is because
    ``cpu_backlog + cpu_solo`` exceeds ``gpu_backlog``; and choosing CPU
    means it is below ``gpu_backlog + gpu_solo``, so the GPU assignment is
    at least as large. The symmetric argument runs the other way. So the
    cheaper local rule optimizes the global objective, and no separate
    join-aware variant is needed.
    """
    if cpu_backlog_us < 0 or gpu_backlog_us < 0:
        raise ValueError("backlog must be >= 0")
    c_solo, g_solo = cpu_us(rows, uniq, c), gpu_us(rows, uniq, c)
    c_join, g_join = cpu_backlog_us + c_solo, gpu_backlog_us + g_solo
    dest = "cpu" if c_join < g_join else "gpu"
    solo_dest = "cpu" if c_solo < g_solo else "gpu"
    return Decision(dest=dest, cpu_join_us=c_join, gpu_join_us=g_join,
                    margin_us=abs(c_join - g_join),
                    flipped_by_backlog=dest != solo_dest)
