# R8, scored — the nominal miss rate cannot see the variable that decides I/O

Registered (`PREREG-tribrid-stage3`, R8):

> nominal placement miss rate becomes a poor I/O metric; physical refill
> rate is the operational one — **refuted if the two stay close**.

**CONFIRMED.** They are not close, and the reason is sharper than a gap: the
two quantities do not depend on the same things.

Scored offline on the captured routing sequence
(`olmoe_routing_seq.jsonl`, 512 decode steps × 16 layers × top-8 of 64 =
65,536 routed invocations) driven through the real `ColdTier`. No GPU: both
numbers fall out of one replay, and neither depends on how fast anything
runs. Receipt: `r8.json`. Scorer: `score_r8.py`.

| cold | hot_rows | nominal misses | physical refills | nominal rate | refill rate | nominal / refill |
|---|---|---|---|---|---|---|
| 5% | 128 | 3270 | 978 | 5.0% | 1.49% | **3.3×** |
| 5% | 256 | 3270 | 261 | 5.0% | 0.40% | **12.5×** |
| 5% | 384 | 3270 | 261 | 5.0% | 0.40% | **12.5×** |
| 10% | 128 | 6521 | 2828 | 10.0% | 4.32% | **2.3×** |
| 10% | 256 | 6521 | 758 | 10.0% | 1.16% | **8.6×** |
| 10% | 384 | 6521 | 358 | 10.0% | 0.55% | **18.2×** |
| 20% | 128 | 13055 | 7349 | 19.9% | 11.21% | **1.8×** |
| 20% | 256 | 13055 | 3215 | 19.9% | 4.91% | **4.1×** |
| 20% | 384 | 13055 | 969 | 19.9% | 1.48% | **13.5×** |

`misses` and `disk_reads` agree exactly at every point, so "physical refill"
is a read the tier actually issued, not a proxy for one.

## The column that does not move

Read the table down instead of across. **`nominal misses` is identical at
every capacity** — 3270 at 5% cold whether the tier holds 128 rows or 384;
13055 at 20% regardless. It is a property of the placement alone and is
blind to the tier by construction.

Physical refills over the same three capacities, at 20% cold: **7349 →
3215 → 969**. Same placement, same workload, same trace — **7.6× less real
I/O**, and the nominal metric reports no change whatsoever.

That is the whole prediction, and it is stronger than "the numbers differ".
A metric can be biased and still rank options correctly. This one cannot
rank the option that matters most, because its value does not depend on it.

## Where they come closest, and why that is the wrong reassurance

The narrowest gap is **1.8× at 20% cold with 128 rows** — the most
capacity-starved point measured, where the tier thrashes and nearly every
nominal miss really is a read. So nominal is least misleading exactly where
the engine is performing worst, and diverges as the configuration improves.
A metric that agrees with reality only in the regime you are trying to leave
is not a metric to tune against.

## What this does not establish

- One trace, one geometry, `order="tail"` placement, three capacities. The
  ratio is workload-dependent by construction — a trace with no reuse would
  drive it toward 1×, which is what the 128-row column is approaching.
- **The refutation threshold was chosen by me, not registered.** R8 says
  "stay close" without a number. I read it as *within 2×*, on the grounds
  that two metrics ranking placements the same way is what "close" has to
  mean for the prediction to have content. Eight of nine points clear that
  bar and the ninth (1.8×) does not — so the verdict does not rest on where
  the line sits, but a stricter reading would score that point as
  not-yet-diverged.
- This measures the metric, not a solver. Nobody has yet built a placement
  solver that optimizes refill rate directly; showing that nominal is the
  wrong objective does not show what the right one costs.
