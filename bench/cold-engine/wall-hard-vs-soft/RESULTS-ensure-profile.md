# I re-ran a profile #163 had already run. Here is what that is worth.

**No new mechanism.** The headline finding of this profile is one the campaign
already had, already tested, and already bounded. What is new is small and
stated plainly below.

## The duplication, first

`#163` recorded:

> the largest soft-minus-hard entry at the **key lambda inside
> `_demote_locked`**: +0.404 s across **1,965,572** calls

This profile: **+0.422 s across 1,965,572 calls**. The same call count to the
digit. I profiled the `ensure()` path without checking whether the region had
already been profiled — after writing, in this same campaign, that the value
of an elimination is that *the next person does not re-run it*. I was the next
person.

It is a faithful replication, which is worth something and not much.

## What is actually new

**The instrument now exists.** `#163` ran its profile ad hoc; there is no
`cProfile` anywhere in the repo and no profile receipt. Its central claim
therefore could not be re-derived from the tree — the same defect `#167` fixed
for `--qd`, in the same campaign, found again here.
`routing-trace/profile_ensure.py` makes it reproducible, and reproducing it is
how the duplication above was detected at all.

**The narrowing holds, and it now rests on two independent legs.**
`RESULTS-read-timing.md` measured 86.5% of the residual outside the read, on
real hardware. `#163` bounded the **demote path** at ≤0.8 of 4.1 points by
substitution — replacing victim choice with a deliberately incorrect O(over)
pick closed only 32% of the offline gap. Those combine:

> The residual is outside the read, and it is **not demote**. It is the rest
> of the `ensure()` path.

Neither result gives that alone. The profile's top entry is demote-related, so
a profile *by itself* would point straight at the thing already excluded —
which is exactly why #163 ran the substitution test rather than trusting the
ranking.

## The ranking, with the trap it sets

Soft-minus-hard, `tottime`, qd=1, `rows=256/protected=248`:

| Δ tottime | Δ ncalls | function |
|---|---|---|
| +0.422 s | +1,965,572 | `nvme_residency.py:575(<lambda>)` — the `nsmallest` key |
| +0.383 s | +8,054 | `heapq.nsmallest` |
| +0.279 s | 0 | `_demote_locked` |
| +0.228 s | +2,077,015 | `dict.get` |
| +0.224 s | +75,648 | `lock.acquire` |
| +0.166 s | +436 | `_victim` |
| +0.111 s | +18,692 | `threading.wait` |

Demote-related entries are ~59% of the soft-minus-hard non-read total. Read
naively that says "demote is the residual" — the conclusion #163 refuted by
experiment. **A profile ranks suspects by cost, not by causal contribution**,
and here the two disagree by enough to invert the answer.

The `preadv` row is excluded: 0.139 s hard against 1.724 s soft for 1.36% more
reads is a 12× gap, which is laptop filesystem variance, and it is I/O anyway —
already measured at +2.19% on the real box.

## The largest absolute cost is not the largest difference

`_victim` is the biggest single consumer in **both** arms — 2.131 s hard,
2.296 s soft — while contributing only +0.166 s of difference. It scans all
slots per state class. It does not explain the soft-hard *gap*, but it sets
the *level* of the non-read work both arms pay, and it has never been
measured on its own. That is a standing item, not a finding here.

## What this does not show

The toy arena reproduces control flow, not bytes; absolute times are not the
real box's. cProfile's per-call overhead inflates many-small-call functions,
so the arm *difference* is meaningful while the shares are not. And a profile
cannot establish causation — the one time this campaign tested a profile's top
entry directly, the test refuted it.

## Receipts

`../routing-trace/ensure_profile.json`. Offline, no GPU, no spend.
