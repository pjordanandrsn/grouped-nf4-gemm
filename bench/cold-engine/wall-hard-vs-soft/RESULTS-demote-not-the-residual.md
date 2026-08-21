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

---

## Follow-up: three more candidates eliminated

`_victim` was named above as "the next candidate, not a proven one". It was
not tested directly; three cheaper eliminations came first and none of them
found the residual either. Recorded so the next attempt does not repeat them.

### 1. The demote path as a whole — worth ≤0.8 of 4.1 points

A CPU profile of both arms (qd=1, offline replay) put the largest
soft-minus-hard entry at the **key lambda inside `_demote_locked`**: +0.404 s
across 1,965,572 calls. (That profile was run ad hoc and its instrument was
never committed; `routing-trace/profile_ensure.py` now reproduces it —
+0.422 s across the same 1,965,572 calls. See `RESULTS-ensure-profile.md`.) `#159` removed the *sort* but kept one key evaluation
per candidate per request, so that cost survived — which looked like #161's
negative result being an incomplete fix rather than a wrong target.

It is not. Replacing the victim choice with an O(over) arbitrary pick — a
deliberately **incorrect** policy, purely as a cost ceiling — closes only
**32%** of the offline soft-hard gap (0.56 s → 0.38 s). Scaled against the
real run's 23.5 s wall, the entire demote path is worth **≤0.8 of the 4.1
residual points**, and all tier bookkeeping combined about 2.4.

### 2. "Soft achieves lower effective bandwidth" — a restatement, not a mechanism

The soft arm's bytes/second deficit tracks the residual across all seven
measurements to within 0.8 points (−1.4/1.5, −4.2/4.4, −7.9/8.7, +1.6/−1.6,
−7.3/7.9, −4.7/5.0, −3.9/4.1). That agreement is exact because it is
algebraic: `residual = wallΔ − readsΔ`, bandwidth is `bytes/wall`, so the
deficit **is** the residual rearranged. It buys a redirect — the residual is
a per-read cost, not a read-count effect — and nothing more.

### 3. CPU–I/O overlap interference — flat across queue depth

If bookkeeping between reads cost bandwidth by letting the queue drain, the
residual should shrink at qd=1 where there is no queue to lose. Same box,
same arena, 5 repeats, only queue depth varying:

| qd | hard | soft | wall Δ | reads Δ | residual |
|---|---|---|---|---|---|
| 4 | 24279 ms | 25799 ms | +6.3% | +1.3% | **+5.0** |
| 1 | 36303 ms | 38660 ms | +6.5% | +1.4% | **+5.1** |

qd=1 is 50% slower overall, so the knob works and overlap is real — the
soft-hard gap simply does not depend on it. Receipts in `qd-probe/`.

> **Still open after optimisation.** Making the tier 41-49% cheaper did NOT
> dissolve the residual: 61% less non-read work, 16% less soft-hard gap, and
> the residual is now 101.9% outside the read
> ([`RESULTS-residual-after-opt.md`](RESULTS-residual-after-opt.md)).
>
> **Partitioned.** The residual was measured in
> [`RESULTS-read-timing.md`](RESULTS-read-timing.md): at qd=1 the soft arm's
> reads cost what their count says (+2.2% on +1.4% more reads) while its work
> *around* the reads costs **+26.6%**. 86.5% of the residual is outside the
> read. Not a mechanism yet, but a named region with a size.
>
> **Followed up.** Read *locality* — named here as the untested shape, since
> nothing recorded offsets — was measured and **refuted**: the two arms read
> the arena in the same order, with identical median gaps at four capacities.
> See [`RESULTS-read-locality.md`](RESULTS-read-locality.md). Five candidates
> are now eliminated.

### Where that leaves it

The residual is a **per-read cost, invariant to queue depth, not accounted
for by tier bookkeeping**. The remaining shape worth testing is read
*locality*: whether the soft arm's slot-reuse pattern spreads its reads
across the arena differently at the same count. That needs offset tracing,
which nothing here records.

Four eliminations and a characterisation is where this stops rather than a
fifth hypothesis — two of the four were positions this campaign had already
started drafting as conclusions.
