# The demote sort is not gnf4#153's residual — tested, and refuted

`#153` measured the cold tier faster with resurrection **off** and named a
residual it could not attribute: soft reads only 1.2–1.5% more but runs
2–8% slower, and the gap **grows with pool size**. That shape suggested
bookkeeping paid per request rather than per resurrection.

`#159` removed the obvious candidate — `ColdTier._demote_locked` sorted the
whole candidate list to take the smallest `over`, on every demand ensure,
measured at 1.24× to 2.38× the early-return path. **This is the test of
whether that was the residual. It was not.**

## Prediction, recorded before the run

The hard arm sets `protected_rows == hot_rows`, which early-returns before
the changed code, so only the soft arm can speed up. If the demote sort were
the residual, the wall delta should collapse toward the reads delta.

| rows/prot | #153 wall Δ | reads Δ | residual to remove |
|---|---|---|---|
| 128/120 | +3.0% | +1.5% | +1.5 pts |
| 256/248 | +5.8% | +1.4% | +4.4 pts |
| 384/376 | +9.9% | +1.2% | +8.7 pts |

## Result: the residual is unchanged

`#153`'s exact invocation, same trace, same 3.34 MB rows, pinned, page cache
dropped between arms — on `main` with **both** demote fixes in. At the one
capacity resolvable on this box, with 9 repeats:

| | #153 | with both fixes |
|---|---|---|
| wall Δ | +5.8% | **+5.5%** |
| reads Δ | +1.4% | **+1.4%** |
| **residual** | **+4.4 pts** | **+4.1 pts** |

Indistinguishable. The reads delta reproduces to the digit, so the setup is
faithful; the residual simply did not move.

**This also independently replicates #153** on different hardware, which is
worth as much as the negative result: its finding is not an artifact of its
box.

## The instrument was blunter than #153's, and that is the lesson

The three-capacity sweep is in `post-demote-fix/wall.json` but only one of
its points is readable:

| rows/prot | effect | arm spread | resolvable |
|---|---|---|---|
| 128/120 | 0.1% | 3.2% | no |
| 256/248 | 9.3% | 5.1% | yes |
| 384/376 | 6.2% | 6.8% | no |

This box is ~4× faster than #153's — 30.5 s per repeat against 117.7 s at
128 rows. Shorter runs give fixed overheads a larger relative share, so
per-arm spread was **3–7% against #153's 0.34%**. Hosts were selected here by
disk bandwidth, which is precisely the wrong axis: a *slower* disk gave #153
the sharper instrument, because the signal had longer to accumulate.

Raising repeats 3 → 9 did **not** tighten it (spreads went 5.1% → 10.3% hard,
3.6% → 20.1% soft, and the arms' distributions overlap). It did stabilise the
median, which moved the residual estimate from +7.9 to +4.1 — so the
three-repeat number was itself unreliable, and no single-point claim from the
first sweep should be quoted.

## What stands, and what does not

**Refuted:** that `_demote_locked`'s sort is #153's residual. `#158` and
`#159` are still real speedups — 1.24–2.38× on `ColdTier`, 1.4–3.5× on
`VramSlots`, behaviour-identical at qd=1 across 8 configurations, all
measured directly rather than inferred from this. Their *justification*
narrows to their own micro-benchmarks.

**Still open:** what the residual actually is. It survives the removal of the
per-request sort, so "bookkeeping that scales with rows" is either wrong or
names something else — `_victim` still scans all slots per state class and
was deliberately left untouched, which makes it the next candidate rather
than a proven one.

**R5 is unchanged.** Its refutation rested on #153's wall numbers, and those
reproduce.
