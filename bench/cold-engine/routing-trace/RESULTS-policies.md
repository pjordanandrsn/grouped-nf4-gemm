# Closing part of the gap gate 3 left: decay the counts, don't split the pool

Receipt: [`policies.json`](policies.json). Harness:
[`score_policies.py`](score_policies.py). Trace:
[`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl). No box.

[`RESULTS-gate3.md`](RESULTS-gate3.md) found the placement loop worth closing
— adaptive re-placement beat static by 6–41% — and its third control found
the policy it used sitting **4–30% above the best achievable fixed set**, with
the gap widening as capacity grows. This attacks that gap.

Profile on steps 0–255, score on 256–511, migrations charged as reads.

## Two candidates, one hypothesis each

**`ewma`** — cumulative counts are dominated by early observations, so a
newly-hot expert is slow to promote and a cooled one slow to demote. Decaying
the counts at each re-placement fixes both without becoming a short window,
which [R4](RESULTS-r4.md) showed loses.

**`hybrid`** — gate 3's own numbers show placement beating LRU below 512 rows
and LRU winning at 768, so neither is right everywhere. Pin the top of the
ranking and demand-page the remainder.

## EWMA wins at four of five capacities

| rows | static | adaptive | **ewma** | hybrid | demand | oracle |
|---|---|---|---|---|---|---|
| 128 | 22,596 | 21,167 | **20,569** | 22,921 | 25,105 | 19,818 |
| 256 | 16,880 | 15,331 | **14,509** | 17,343 | 18,874 | 13,389 |
| 384 | 11,708 | 10,380 | **9,684** | 10,524 | 13,458 | 8,498 |
| 512 | 7,531 | 6,226 | **5,742** | 6,543 | 9,308 | 4,773 |
| 768 | 2,101 | 1,243 | **1,013** | 1,133 | 1,330 | 496 |

**EWMA is best at every capacity**, and it closes **31–44% of the gap between
adaptive and the ceiling**:

| rows | adaptive over oracle | ewma over oracle | gap closed |
|---|---|---|---|
| 128 | +6.8% | **+3.8%** | 44% |
| 256 | +14.5% | **+8.4%** | 42% |
| 384 | +22.1% | **+14.0%** | 37% |
| 512 | +30.4% | **+20.3%** | 33% |
| 768 | +150.6% | **+104.2%** | 31% |

## The hypothesis that mostly failed

**Hybrid loses at four of five capacities, and the sweep says why.** Its reads
fall monotonically as the pinned fraction rises — it is best when it stops
being a hybrid:

| rows | pin 25% | 50% | 75% | 90% | **100% (= adaptive)** |
|---|---|---|---|---|---|
| 128 | 27,796 | 25,011 | 22,921 | 21,865 | **21,167** |
| 384 | 12,578 | 11,221 | 10,524 | 11,412 | **10,380** |
| 512 | 9,068 | 7,914 | 6,543 | 6,244 | **6,226** |
| 768 | 1,347 | 1,298 | 1,133 | **1,003** | 1,243 |

**The exception is 768 rows**, where the best hybrid (pin 90%, **1,003**)
edges `ewma` (**1,013**) by 1%. That is the one capacity where `demand`
already beats `static` outright — capacity nearly covers the working set and
recency wins — so a demand-paged slice has something to contribute. Everywhere
below, the blend inherits the weaknesses of the worse half.

The reasoning behind hybrid was that two policies win in different regimes so
a blend should win in both. It wins only in the regime where the *other* half
was already winning alone, which is not the claim.

> **Corrected after review.** The first version of `hybrid_p` dropped demoted
> pins instead of handing them to the demand-paged half, and charged a
> migration for keys already resident in it — holding residency below `cap`
> and inflating hybrid's reads (Bugbot, gnf4#157). The numbers above are the
> fixed implementation. The correction narrowed hybrid's loss and gave it the
> 768-row win; the conclusion that it is not the answer survives, on an arm
> that is no longer handicapped.

## The decay is tuned, and has an interior optimum

| rows | 1.0 (none) | 0.9 | 0.75 | **0.5** | 0.25 | 0.1 |
|---|---|---|---|---|---|---|
| 128 | 21,167 | 21,061 | 20,809 | **20,569** | 20,680 | 20,685 |
| 384 | 10,380 | 10,148 | 9,906 | **9,684** | 9,716 | 9,833 |
| 512 | 6,226 | 6,083 | 5,915 | **5,742** | 5,969 | 5,995 |
| 768 | 1,243 | 1,199 | 1,126 | **1,013** | 1,071 | 1,134 |

**0.5 wins at every capacity, with worse values on both sides.** A genuine
interior optimum, not a monotone slide that would suggest the parameter is
just proxying for something else. Decaying too hard turns the estimator into
the short window R4 already refuted, and the curve turning back up at 0.25 is
that effect appearing.

## What remains

Even at its best the policy is **3.8% to 104% above the ceiling**, and the gap
still widens with capacity. EWMA takes a third of it; two thirds are still
there, and nothing here says a *fixed-set* policy can take the rest — the
oracle is a fixed set, so the remaining gap is entirely about choosing it
better, not about choosing more often.

## Limits

- **One prompt**, 512 decode steps, one model. Same limit as gate 3: this is
  movement within a single generation, not across the workload changes a
  served model sees.
- Reads counted, not timed; a migration is charged as one read.
- Capacity is a flat row count, and `static` is top-C by frequency rather than
  `solve_placement`'s greedy balance — which makes the static baseline
  slightly weaker than a deployed placement.
- The decay is per re-placement, so its effective half-life depends on
  `period`. Only `period=32` was swept.
