# PREREG — P2-G2': residency convergence, single pool

Registered before any measurement, from G2's anatomy alone
([RESULTS-p2-g2.md](RESULTS-p2-g2.md)); bars and scope frozen in spec §11's
G2' entry before this harness ran anywhere. Offline on the 16 committed
rank traces (`rank-2026-08-22/*.jsonl`), cold start, trace units.

## What changed from G2, and only this

The per-key persistent machinery is gone (spec §5, G2' re-founding): ONE
`DevRowCache` at the full capacity, `protected = rows − k` (I1 — the
allocator's own default), no partition, no PERIOD batches, no LFU set, no
θ, no ages. The shipped engine's *static* `hot_ids` persistent arena is
orthogonal to the law under test and absent from the replay; its capacity
is not counted. Everything else is G2's registered protocol unchanged:
SMOOTH_CAP budgeting via the cache's own `discard()` failed-fill API,
burst accounting, PROMO_FRAC arms {unthrottled, 1/16, 1/8, 1/4}, EWMA
α = 1/16, η = 0.25, plateau window 256–512, convergence = trailing-32 ≤
1.10 × plateau + 1.0, eval window 128–512.

**Registered scope corrections** (both named in G2's RESULTS before this
registration): capacity sweep **{128, 256, 512, 1024, 2048}** — per trace,
every sweep capacity ≤ its pairs runs, **plus above-pairs capacities until
one is capacity-adequate** (adequacy is pairs ≤ 0.9 × rows, which by
construction needs rows > pairs — "capped at pairs" would make adequacy
unreachable, so the cap applies only once an adequate arm exists); clause
(c) evaluates **capacity-adequate arms only** — all of them, not just the
largest.

## Registered claims

* **(a) Convergence** (unthrottled): every capacity ≤ pairs converges
  ≤ 64 steps; bar ≥ **14/16 traces**.
* **(b) Plateau quality** (unthrottled): eval-window fills ≤ **1.10 ×**
  same-capacity ideal-LRU at **every** arm.
* **(c) Equilibrium churn** (unthrottled): `EWMA(fills) ≤ (1 + η) ·
  EWMA(novelty) + 1.0` at **every capacity-adequate arm**.
* **(d) Throttle gracefulness**: at every arm and PROMO_FRAC: convergence
  ≤ 2 × max(unthrottled, 16) AND fills ≤ 1.05 × unthrottled.

PASS iff (a) ∧ (b) ∧ (c) ∧ (d). Refuted at any clause ⇒ per-clause report;
no tuning against these traces.

## Falsifiability — unchanged, both must fail

1. The I1 margin trap (`protected = 1`, margin ≫ k): must blow (c) at its
   arm or (b) at its capacity.
2. No-retention (fill counted then `discard()`ed): plateau ≥ 0.90 ×
   all-miss. Both at each trace's largest capacity, unthrottled.

## What would count as a miss

Any clause refuted ⇒ REFUTED, per-clause; either spoiler passing ⇒
UNINFORMATIVE; a trace with zero capacity-adequate arms under the extended
sweep ⇒ UNINFORMATIVE (the sweep was mis-extended — pairs max is 1,439 and
0.9 × 2048 = 1,843, so this should be impossible and its occurrence means
the loader or the pair count is wrong).
