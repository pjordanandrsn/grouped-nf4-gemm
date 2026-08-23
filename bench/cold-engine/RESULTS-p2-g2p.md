# RESULTS — P2-G2': REFUTED as scored — and the failing clauses contradict the spec's own equilibrium definition

Registered in [PREREG-p2-g2p.md](PREREG-p2-g2p.md) (#214). Run 2026-08-23
offline on the 16 committed rank traces, repo at `08ea58a`, cold start, one
`DevRowCache` per arm. Receipt:
[p2-g2p-2026-08-23-registered.json](p2-g2p-2026-08-23-registered.json).

**Scored verdict: REFUTED** — (a) 1/16, (b) 72/73, (c) **16/16 PASS**,
(d) 167/219; both spoilers fail 16/16 (informative). The verdict stands as
registered. What the clause anatomy shows is that the single-pool law did
exactly what §5 derives, and the two failing clause *operationalizations*
measure something else.

## The cure held

Against G2's partition: (b) went 16/56 → **72/73** arms within 1.10× ideal-
LRU (the single failure is granite_code@1024 at **23 vs 20 fills** over 384
steps — a 1.15 ratio on a microscopic denominator; the convergence clause
carries a +1.0 absolute guard for exactly this regime, (b) does not).
(c) went 4/16 → **16/16**: at every capacity-adequate arm,
`EWMA(fills) = EWMA(novelty)` to the second decimal on all 16 traces —
0.12/0.12, 0.03/0.03, 0.00/0.00 — the law fills on first arrivals and
nothing else. The 62× regime is gone; there is no partition to starve.

## Why (a) and (d) fail anyway

Clause (a)'s criterion (trailing-32 fill rate within 1.10× of the late
plateau + 1.0) was my operationalization in the G2 prereg. At adequate
capacity the late plateau is **0**, so the criterion demands trailing
fills ≤ 1.0/step — i.e. it demands **the trace's novelty itself** decay
below 1/step by step 64. These traces discover their working sets over
66–108 steps, so 14/16 traces "fail convergence" at their adequate arms
*while sitting exactly on §5's equilibrium predicate the whole time* —
the predicate clause (c) scores, and passes 16/16. The criterion measures
discovery decay, not the law. (Two genuinely slow points exist besides the
artifact: granite_code@512 (conv 292) and granite_dialogue@1024 (conv 224),
both capacity-inadequate arms settling under real eviction churn.)
Clause (d) inherits both parents: its convergence half compares
against (a)'s artifact, and its 1.05× fill half hits the same microscopic
denominators.

**The registered instruments disagreed with the registered spec.** §5
(frozen in #211, before any replay ran) defines equilibrium as
`EWMA(fills) ≤ (1 + η)·EWMA(novelty)`; the prereg's (a) operationalized
convergence as plateau-relative decay instead of time-to-that-predicate.
Where the two disagree, the spec's definition is the registered intent —
but the scoring is the scoring, so this run is REFUTED and the correction
goes forward as a new registration, not a re-score.

## G2'' — the correction, registered here from frozen text

* **(a)** becomes **time-to-sustained-equilibrium**: the first step t
  where §5's predicate (`EWMA(fills) ≤ (1 + η)·EWMA(novelty) + 1.0`,
  α = 1/16) holds and keeps holding through the trace end. Bar unchanged:
  ≤ 64 steps at every capacity on ≥ 14/16 traces.
* **(b)** gains the absolute guard the convergence clause always had:
  fills ≤ 1.10× LRU **+ k** (k = one routed set — the slack of a single
  step at microscopic denominators). Bar otherwise unchanged.
* **(d)** re-bases on the corrected (a); fill half gains the same +k guard.
* (c), the sweep, the spoilers, and every parameter are unchanged.

No re-scoring of this receipt: G2'' re-runs the (deterministic) replay
fresh at its own registration. No box; program spend unchanged (~$0.77).
