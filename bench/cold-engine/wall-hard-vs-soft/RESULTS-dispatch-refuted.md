# Dispatch is not the residual, and the profile that pointed at demote was inflated

**Dispatch: REFUTED.** It fails on arithmetic before it fails on measurement.
**Demote selection: ≥6.3% of the gap, not the 85% the profile implied.**
Neither is the mechanism.

## Why dispatch could never have been it

Per-read dispatch — a Future, an Event and several lock round-trips per read —
is a **per-read** cost. The soft arm issues **+1.36%** more reads, so a
per-read cost can contribute **+1.36%**. It cannot produce the **+56.46%**
non-read delta that `RESULTS-residual-after-opt.md` measured. For dispatch to
explain the residual, soft's dispatch would have to be dearer *per read*, which
is a different claim than the one that was proposed.

The profile agrees. Soft-minus-hard `tottime`, current code, qd=1:

| bucket | Δ | share of non-read delta |
|---|---|---|
| dispatch (`lock.acquire`, `SimpleQueue.get`, `threading.wait`, `as_completed`) | +0.166 s | ~10% |
| demote (`nsmallest` key lambda, `nsmallest`, `_demote_locked`, its `dict.get`, `heapreplace_max`) | +1.364 s | ~85% |

**I proposed dispatch on the wrong criterion.** It was the largest *absolute*
cost after the LFU heap (~28% of tier CPU). Absolute cost is the wrong ranking
for an asymmetry — dispatch is large in *both* arms, which is precisely why it
cannot explain a difference between them. This campaign had already recorded
that error once, in `RESULTS-ensure-profile.md`, and it was repeated here.

## The 85% is an artifact

> **Corrected by [`RESULTS-bounding-the-residual.md`](RESULTS-bounding-the-residual.md).**
> This section is wrong. Demote really is ~91% of the gap; an additive
> accounting shows it directly. What the profile got wrong was *where inside*
> demote — it charged the key lambda, and the selection form turns out not to
> matter (lambda 0.202 s, generator 0.210 s, list-comp 0.212 s, indistinguishable).
> The 6.3% below is the honest size of the *lookup* overhead, not of demote.
> The reasoning that follows about cProfile inflating many-call entries stands;
> the conclusion drawn from it does not.

`_demote_locked` still ran the pattern removed from `_victim` in #175: the
`nsmallest` key doing `self._freq[k]` and `self._last_use.get(k, 0)`, evaluated
once per candidate — **1,965,572 calls** across the trace, the single largest
entry in the profile.

Applying the same fix (hoisted lookups, direct index, identical victims):

| build | hard | soft | gap |
|---|---|---|---|
| current | 0.711 s | 1.424 s | 0.713 s |
| demote-opt | 0.725 s | 1.394 s | **0.669 s** |

**Closed: +6.3%**, against a hard-arm noise reference of **2%** and identical
read counts. Real, and an order of magnitude below what the profile implied.

cProfile charges roughly a microsecond per call, so a 1.97M-call entry accrues
seconds of overhead that belong to the profiler. The ranking put demote at 85%;
the unprofiled test says the *selection overhead* is worth 6.3%. **The second
time in this campaign a profile's top entry has failed an unprofiled test** —
the first was `RESULTS-ensure-profile.md`, where the top entry was a cause
#163 had already refuted by substitution.

This is a lower bound: the fix removes attribute-lookup overhead, not the ~244
key evaluations per demotion. Demote selection sits somewhere between 6.3% and
#163's 32% substitution ceiling. Neither end makes it the mechanism.

## Three measurements were thrown away before this one

Worth recording, because each failed differently and only the last is usable.

**1. Invalid cost ceiling.** Replacing the demote selection with an arbitrary
`cands[:over]` pick — #163's technique — changed the *workload*: soft reads
32605 → **38432**, +18%. A wrong demote policy causes more misses, so the probe
removed selection cost while adding read work. Its "+14.3% closed" is
uninterpretable. A cost ceiling has to hold the work constant, which for this
code means an optimisation, not a substitution.

**2. Ordering confound.** Running all of build A then all of build B penalised
whichever loaded second: the **hard** arm — identical code in both builds, and
it never demotes — moved 0.765 → 0.814 min and 0.800 → **1.313** median. That
is the A/B/A rule this campaign already runs on *arms*, never applied to
*builds*. Interleaving dropped the reference to 2%.

**3. The first attempt at (2)** had a 10% noise reference and was discarded on
that basis alone.

## What is left

~~Dispatch ~10%, demote selection ≥6.3%. Neither dominates... The asymmetric
work is spread rather than concentrated.~~

**Superseded.** Adding the pieces up rather than testing them one at a time
shows the work IS concentrated: `_demote_locked` is **90.9%** of the gap, in two
O(resident) operations per request. See
[`RESULTS-bounding-the-residual.md`](RESULTS-bounding-the-residual.md). The
"spread" reading came from measuring candidates individually, each of which
looked small.

## Receipts

Kept in the commit message rather than as JSON: these are laptop CPU timings
whose value is the *ratio* against a same-run noise reference, and shipping
them as receipts would invite exactly the cross-run comparison that produced
failures 1–3.
