# #153's residual is gone

**Registered outcome: CONFIRMED.** Removing `_demote_locked`'s scan removed the
wall-clock residual. Three measurements on one box put it at **+0.22, −0.24,
−0.21 points**, against **5.61** before the change and 4.4–7.0 across five
earlier boxes.

Registered in [`PREREG-residual-after-demote.md`](PREREG-residual-after-demote.md)
before the box was rented, with CONFIRMED / PARTIAL / REFUTED named in advance.

## Result

`rows=256`, `protected=248`, O_DIRECT, arms interleaved, bracketed.

| | Δwall | Δreads | residual | residual 95% CI | gate |
|---|---|---|---|---|---|
| leading control (9 reps) | +1.58% | +1.36% | **+0.22 pts** | [−0.37, +2.40] | BELOW-RES |
| trailing control (7 reps) | +1.12% | +1.36% | **−0.24 pts** | [−0.35, +1.12] | BELOW-RES |
| timed qd=1 (7 reps) | +1.15% | +1.36% | **−0.21 pts** | [−1.85, +0.77] | **UNUSABLE** |

The **trailing control's CI upper bound of +1.12 clears the registered ≲2 pts on
its own**, and all three intervals exclude 5.61 decisively.

## The mechanism, in the form the preregistration demanded

The prereg said absolute shrinkage proves nothing — `non_read_ns` was going to
fall whatever happened, because that is what #182 did. **The question was the
delta**, and the delta collapsed:

| | before | now |
|---|---|---|
| `Δnon_read_ns` | **+56.46%** | **+10.24%** |
| `Δread_ns` | +1.24% | +0.87% |
| `read_ns` share of wall | 89.6% | 97.1% |

`Δread_ns` is again *below* the +1.36% read delta, so the soft arm's reads
remain individually no more expensive. What is left of the asymmetry is a
+10.24% delta on a bucket that is now 2.9% of wall.

## The chain, every link measured

1. The residual is **101.9% outside the read** — real NVMe, `RESULTS-read-timing.md`
   and `RESULTS-residual-after-opt.md`.
2. **90.9%** of the tier's asymmetric CPU is `_demote_locked` — additive
   accounting, `RESULTS-bounding-the-residual.md`.
3. Removing that scan removes **76–87%** of the tier gap — `RESULTS-demote-heap.md`.
4. **And it removes the wall residual** — this document.

Step 4 was the one that could have broken. Steps 1–3 are three separate
instruments, and "the CPU I measured offline is on the wall-clock critical
path" was an inference until now. The registered REFUTED branch said so
explicitly: a surviving residual would have retired the offline accounting
method, not just this hypothesis.

## My own gate flags one of these runs

`instrument_gate.py` marks the timed qd=1 run **UNUSABLE** — hard-arm IQR
2.28% against a 2.0% threshold. So the conclusion rests on the two controls,
both clean; the timed run's partition corroborates and is not load-bearing.

The gate fired on my own data an hour after I wrote it, which is the behaviour
I wanted from it.

## Eight candidates, and what actually found it

| candidate | outcome |
|---|---|
| the demote sort | refuted (#161) |
| the whole demote path | bounded ≤0.8 of 4.1 pts (#163) |
| "lower effective bandwidth" | a restatement, not a cause |
| CPU–I/O overlap | flat across queue depth |
| read locality | same read order, identical medians |
| per-read storage cost | absent (`Δread_ns` below read count) |
| dispatch | ~10%, and arithmetically incapable |
| the victim sweep | ≤16% |

Every one was tested individually and none dominated, which is why the
"spread, not concentrated" reading survived as long as it did. **Additive
accounting found in one afternoon what eight eliminations had not**: measure
what the soft arm does that the hard arm does not, and add it up, rather than
testing suspects one at a time.

## What this does not show

One trace, one model, one capacity, one queue depth. The residual is measured
as a *difference of ratios*, which survives the fact that #182 also made the
hard arm cheaper; absolute `non_read` figures do not and are not compared
across runs.

Storage remains ~2% of decode cost. This closes a measurement question, not a
performance one — the practical stakes were always small, and the value here is
that the method held up end to end.

## Receipts

`residual-after-demote/` — `a_ctl_lead`, `a_ctl_trail`, `a_timed_qd1`. Two
boxes (one destroyed unused after refusing SSH). Total spend for this
measurement **$4.61**; session total the same.
