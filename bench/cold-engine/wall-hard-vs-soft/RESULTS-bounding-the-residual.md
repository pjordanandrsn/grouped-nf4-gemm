# Bounding the total: 93% of the soft-hard gap is `_demote_locked`

**The asymmetric work is concentrated, not spread — and my "spread" hypothesis
was wrong.** After seven candidates tested one at a time, adding them up gives
the answer none of them did individually.

## The accounting

Everything the soft arm does that the hard arm does not is the reclaimable
machinery: with `protected_rows=None`, `_demote_locked` early-returns, nothing
is ever reclaimable, and nothing is resurrected. Timed with targeted counters
on those functions — thousands of calls, not millions, so per-call timer
overhead is negligible.

`rows=256`, `protected=248`, `qd=1`, arms interleaved, min-CPU repeat:

| term | s | share of gap |
|---|---|---|
| **`_demote_locked`** | **0.575** | **90.9%** |
| `_resurrect_locked` | 0.001 | 0.2% |
| `_victim` (reclaimable rank term) | 0.001 | 0.2% |
| 436 extra reads × rest-of-tier | 0.010 | 1.5% |
| **accounted** | **0.587** | **92.8%** |
| unaccounted | 0.046 | 7.2% |

## Inside `_demote_locked`

Instrumented internally, no double-counting (total 0.547 s, against 0.579 s
measured from outside — the gap is the wrapper's own overhead):

| phase | s | share |
|---|---|---|
| guard (`over <= 0` early-outs) | 0.002 | 0.4% |
| **`cands` build** | **0.157** | **28.6%** |
| **selection** (`nsmallest`) | **0.348** | **63.7%** |
| apply (write `_reclaimable`, `_vpush`) | 0.040 | 7.4% |

Both dominant phases are **O(resident) per request**: build a ~252-element
candidate list, then evaluate a rank for every one of them to choose ~4
victims. `_demote_locked` runs on all 8054 non-early-out requests.

This is the same shape `_victim` had before #176 — a linear scan per operation
— in the one function that only the soft arm executes. That is why the
asymmetry survived making the tier 41–49% cheaper: #175 and #176 fixed
`_victim`, which both arms run, and left the soft-only scan untouched.

## Correction to `RESULTS-dispatch-refuted.md`

That document says the profile's 85% attribution to demote "is an artifact."
**That was wrong, and this supersedes it.** Demote really is ~91% of the gap.
What the profile got wrong was *where inside* demote: it put the cost on the
key lambda's 1.97M calls, and the selection form turns out not to matter at
all. Measured in isolation at realistic sizes (252 candidates, `over`=4, 8054
demotions):

| shape | s |
|---|---|
| `key=lambda` | 0.202 |
| generator of tuples | 0.210 |
| list-comp of tuples | 0.212 |

Indistinguishable. So the profile was **right about the function and wrong
about the line** — and the 6.3% my hoisting fix recovered is the honest size of
the *lookup* overhead, not of demote.

The generator rewrite was implemented, verified equivalent, measured at 0.599 →
0.579 s (inside noise), and **reverted**: it preserved tie-order through an
enumeration index, which is real complexity for no gain.

## What this bounds

- The residual is **not** spread across the tier. **91% is one function.**
- It is **not** the selection *form*, **not** dispatch (~10%, and arithmetically
  incapable — a per-read cost cannot exceed the +1.36% read delta), **not**
  locality, **not** per-read storage cost, **not** resurrection (0.2%), and
  **not** the reclaimable term in `_victim`'s rank (0.2%).
- It **is** two O(resident) operations that run on every request, in the one
  code path the hard arm never enters.

7.2% remains unaccounted, which is the honest error bar on this decomposition.

## Done: see `RESULTS-demote-heap.md`

The fix below was implemented and **confirmed**: the scan is replaced by a lazy
heap, closing **76–82%** of the soft-hard gap (`_demote_locked` 0.575 s →
0.102 s), with the accounting *improving* to 97.0% attributed. Preregistered at
≥70%.

## The fix this implies, and why it is not in this commit

`_demote_locked` needs what `_victim` got in #176: an incremental structure
instead of a per-request scan. That is a second eviction-policy rewrite with
the same silent-failure profile — a wrong demotion set changes which rows are
reclaimable, which changes resurrections and reads, and raises nothing. It
deserves its own preregistration and verifier, not momentum from this
measurement.

## Receipts

`../routing-trace/bound_soft.json`, `bound_soft_after.json`. Instrument
`../routing-trace/bound_soft_overhead.py`. Offline, no GPU, no spend.
