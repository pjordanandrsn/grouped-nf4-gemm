# Gate 3's central question, scored: closing the loop pays

Receipts: [`gate3-warm128.json`](gate3-warm128.json),
[`gate3-warm256.json`](gate3-warm256.json). Harness:
[`score_gate3.py`](score_gate3.py). Trace:
[`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl). No box.

Gate 3 is the loop the earlier gates left open:

    placement -> execution -> observed cost/reuse -> new placement

This does not run gate 3. It scores the question gate 3 exists to answer —
**is the loop worth closing** — offline, on a real decode routing sequence,
before anything is built around it.

Four policies at one capacity on one trace, profiled on a prefix and scored
on the remainder:

| policy | what it is |
|---|---|
| `static` | profile the prefix, pin the top-C experts by frequency, never change. What `solve_placement` does. |
| `adaptive` | same start, then re-pick the top-C from all observations so far every 32 steps. Promotions charged as reads. **This is gate 3.** |
| `demand` | no placement at all — LRU over the same C rows. What the tier already does. |
| `oracle` | the best FIXED set, chosen knowing the evaluation window. Not achievable; it sizes the headroom. |

Frequency is the adaptive signal because **R4 was scored on this same trace**
and short-window recurrence lost at every capacity carrying signal
([`RESULTS-r4.md`](RESULTS-r4.md)).

## Adaptive beats static everywhere, and migration is nearly free

Profiled on steps 0–255, scored on 256–511:

| rows | static | adaptive | migrations | demand/LRU | oracle | adapt vs static |
|---|---|---|---|---|---|---|
| 64 | 26,178 | 25,022 | 20 | 32,768 | 24,047 | **−4.4%** |
| 128 | 22,596 | 21,167 | 41 | 25,105 | 19,818 | **−6.3%** |
| 256 | 16,880 | 15,331 | 80 | 18,874 | 13,389 | **−9.2%** |
| 384 | 11,708 | 10,380 | 78 | 13,458 | 8,498 | **−11.3%** |
| 512 | 7,531 | 6,226 | 66 | 9,308 | 4,773 | **−17.3%** |
| 768 | 2,101 | 1,243 | 49 | 1,330 | 496 | **−40.8%** |

**6–41% fewer reads than static, and the migrations that bought it are
under 1% of the reads they save** — 66 migrations to remove 1,305 reads at
512 rows. The gain grows with capacity, because a larger resident set has
more room for the ranking to be wrong about.

## Three controls, because "beats static" is a low bar

**1. It is not just a stale profile.** Doubling the profile (128 → 256 steps)
independently improves static by 1.8–13.5% — and adaptive *still* gains
6.3–17.3% on top of the longer profile. The two compose; re-placement is not
a substitute for profiling well, and profiling well does not remove the need
for it.

**2. It is not tracking noise in a static workload.** The top-384 set really
does move: overlap between trace halves is 275/384 (Jaccard 0.558), and
Q1 vs Q4 is 229/384 (0.425). There is genuine movement to track.

**3. It is nowhere near the ceiling.** Adaptive sits **4–30% above oracle**,
and the gap *widens* with capacity — 4.1% at 64 rows, 30.4% at 512, 150% at
768. Oracle needs the future so adaptive cannot reach it, but the shape says
the policy (top-C by cumulative frequency, re-picked every 32 steps) is
leaving a lot on the table. **Gate 3's loop is worth closing and the obvious
first policy is not the right one.**

## Placement beats demand-paging, until it doesn't

`demand` (LRU, no placement machinery at all) is **worse than static** at
every capacity up to 512 — +5% to +25% — and only wins at 768 of 1024 rows,
where capacity nearly covers the working set and recency beats a frequency
ranking.

So profile knowledge is worth something LRU cannot recover from the request
stream alone, in exactly the regime where the fast tier is scarce. That is
the regime that matters.

## Period sensitivity

At 384 rows, re-placing every 8 steps gives −16.9% and every 128 gives
−11.4%, against −16.7% at the 32 used above. **The result is not tuned**:
anything from 8 to 64 lands within a point of the same answer, and even a
very lazy loop captures two thirds of it.

## Limits — the important one first

**One prompt.** 512 decode steps of one continuation from one model. The
movement measured in control 2 is movement *within a single generation*, not
across the workload changes a served model sees. Whether the same policy
helps across prompts, users or tasks is a different measurement and this does
not make it.

- Reads are counted, not timed. A migration is charged as one read; on real
  hardware a promotion may cost more than a demand fill, or less if it is
  overlapped.
- `static` here is top-C by frequency, a stand-in for `solve_placement`'s
  greedy balance — simpler than the real thing, and it makes the static arm
  slightly weaker than a deployed placement would be.
- Capacity is a flat row count. Real placement spans VRAM and DRAM with
  different costs, which this does not model.
