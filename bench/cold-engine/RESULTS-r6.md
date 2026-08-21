# R6, scored in refills: the band is real, the gain in it is 15 rows

**Verdict: CONFIRMED as registered — and the confirmation is close to
worthless. My earlier CONFIRMED is vacated; it swept the wrong axis with the
retired metric.**

R6 (PREREG-tribrid-stage3):

> largest gains at active working set ≈1.1–2× protected fast-tier capacity
> — REFUTED IF gains are flat across pressure.

Scored offline on the committed `olmoe_routing_seq.jsonl` (512 steps × 16
layers × top-8 of 64; 989 distinct rows touched) with
`routing-trace/score_r6.py`, at **matched capacity** — both arms hold the
same physical rows, differing only in whether ownership is capped below that
number. That is R10's design and the only one where "gain" means anything: a
scheduler choosing reclaimable residency gives up owned rows, it does not
gain free ones. Metric is **physical refills**, `qd=1` (deterministic;
confirmed identical across three runs).

## Axis 1 — pressure, which is what R6 names

Reclaimable fraction held constant at 0.90 while `rows` varies:

| rows | protected | ws/prot | hard refills | soft refills | gain | gain % | P (retired) |
|---|---|---|---|---|---|---|---|
| 128 | 115 | 8.60× | 44298 | 44961 | −663 | **−1.50%** | 1.73% |
| 256 | 230 | 4.30× | 32169 | 32591 | −422 | −1.31% | 5.58% |
| 512 | 461 | 2.15× | 13616 | 13711 | −95 | −0.70% | 18.29% |
| 768 | 691 | 1.43× | 3340 | 3342 | −2 | −0.06% | 45.43% |
| **832** | **749** | **1.32×** | **2044** | **2029** | **+15** | **+0.73%** | 59.83% |
| 896 | 806 | 1.23× | 1311 | 1307 | +4 | +0.31% | 74.69% |
| 1024 | 922 | 1.07× | 989 | 989 | 0 | 0.00% | 100.00% |

Gains are **not** flat across pressure, so R6's registered falsifier does not
fire. The maximum is real: a plateau, not a spike — 800/832/864 rows give
+11/+15/+7 refills — deterministic, and located at **1.32×**, squarely inside
the registered 1.1–2.0× band. R6 named the right band.

It named a band worth 15 rows. The same policy costs **663 refills** at
8.60×. R6 locates the pressure at which reclaimable residency stops being
harmful, not one at which it pays; the best case is +0.73% and the curve is
negative across every other pressure measured. That is consistent with R10,
which refuted "reclaimable cuts churn" at matched capacity 10 times out of 10.

## Axis 2 — ownership, which is what I actually swept last time

Fixed `rows=512`, sweeping the ownership cap:

| frac | protected | gain % | P (retired) |
|---|---|---|---|
| 0.50 | 256 | −0.66% | 58.32% |
| 0.70 | 358 | −0.68% | 43.47% |
| 0.90 | 461 | −0.70% | 18.29% |
| 0.98 | 502 | −0.75% | 3.58% |

**P swings 16-fold. The refill gain does not move, and is negative
throughout.** This is the axis `RESULTS-tribrid-reclaimable.md` swept, and
the metric it read, when it recorded "R6 — CONFIRMED, P rises monotonically
as ownership tightens." That verdict is wrong twice over: it is not R6's axis
(R6 is a claim about working-set pressure, not ownership fraction), and P is
the metric STAGE3-SYNTHESIS retired as unable to carry a claim. Vacated here.

## Why P was retired, in one row of the table

At 1.07× pressure, **P = 100.00%** — every reclaimable row was reused before
being overwritten, a perfect score. The refill ledger for that same run:
**zero rows saved.** At that pressure the working set fits, so the hard arm
never evicted anything either, and every row P credits as "resurrected" is a
row a better-demoting cache would never have given up. That is precisely the
synthesis's stated reason for retiring the metric, now with a number on it.

The two metrics also peak in different places (P at 1.07×, gain at 1.32×) and
rank the ownership axis while the gain there is flat. P is not a conservative
proxy for savings; on axis 2 it is uncorrelated with them.

## Preregistration defect, same family as R2/R3/R9

R2, R3 and R9 each name a rate without pinning the configuration that sets
it. R6 has the sibling defect: it names *where* the gains peak without
requiring the peak be **positive**. A policy that loses everywhere satisfies
"largest gains at 1.1–2×" by losing least there, and its falsifier — "flat
across pressure" — tests for variation, not for benefit. A registered
threshold ("≥X% fewer refills somewhere in the band") would have made this
prediction decidable in the direction that matters.

## Receipts

`routing-trace/r6_final.json`, `r6_f0.75.json`, `r6_f0.90.json`,
`r6_fine.json`. Scorer `routing-trace/score_r6.py`; offline, no GPU, no spend.

## What this does not show

One trace, one model (OLMoE), one arena geometry. The +0.73% peak is 15
refills out of 2044 and I would not defend its exact location to two
significant figures across other traces — only that a positive region exists,
is deterministic here, and is small. The negative results at high pressure
are the robust part: they are 1–2 orders of magnitude larger than the peak.
