# Preregistration — evicting by position in the layer cycle

Registered before measurement. No cyclic policy has been run on these traces.

## The structure general cache theory does not have

[`RESULTS-adaptive-policy.md`](RESULTS-adaptive-policy.md) closed six attempts
on the frequency/recency split — five explanations and ARC — and ended by
saying that anything which closes the ~1.9× gap "will have to use something
about MoE routing that general cache theory does not have."

There is such a thing, and it is not subtle. **A decode step walks the layers
in a fixed order, once each.** So at layer ℓ of step *t*, for any resident row
belonging to layer *m*, the distance to its next possible use is known
exactly, with no future knowledge:

```
m >  ℓ :  m - ℓ            (needed later in THIS step)
m <= ℓ :  (L - ℓ) + m      (not needed until the NEXT step)
```

LRU evicts the least-recently-used row. Under this access order that row
belongs to the previous step's *highest* layers — which is to say, **the rows
needed soonest**. LRU is not merely uninformed here; it is systematically
choosing the worst available victim, which is the mechanism behind the
zero-hit region already documented in
[`RESULTS-crossover.md`](routing-trace/RESULTS-crossover.md).

## The policy

**cyclic**: evict the resident row with the greatest cyclic distance as
computed above, breaking ties by LRU.

This is Belady's rule restricted to the component that is structurally known.
It is fully online — it uses the layer index and the current position, both of
which the engine already has — and it needs no scores, no ghost lists and no
adaptation.

What it does **not** know is whether a given expert will be routed again at
all. The cyclic distance is the distance to that layer's next *visit*, not to
the row's next *use*, so the policy is exactly as wrong as the assumption
"this expert recurs" is wrong. That is the interesting part: it converts an
unknown about experts into a certainty about layers.

Scored against **LRU**, **LFU**, **ARC** and **Belady** on the 48 published
cells — four models × four prompts × three capacities — with
`policy_headroom.py`'s transfer counts.

## C1 — it works on every model

> `cyclic` closes **≥ 30%** of the LRU-to-Belady gap on **all four** models.

The "all four" is the point. LFU closes 49 / 49 / 30 / **2** %; ARC closes a
flat 5–6%. Both fail the thing that matters, which is working without knowing
which model you are on. A policy that closes half the gap on three models and
nothing on gpt-oss is another per-model story, and this program already has
one of those.

* **Confirmed** only if every model clears 30%.
* **Refuted** if any model falls below — reported per model, and explicitly
  not softened into "works on most".

## C2 — it beats LRU everywhere

> `cyclic` makes strictly fewer transfers than LRU in **at least 44 of 48**
> cells.

A structural argument predicts LRU is systematically wrong here, so a policy
built on that argument should beat it nearly always, not on average. Four
cells of slack for the capacities where nothing is evicted at all.

* **Refuted** below 44, which would mean the structural argument is wrong
  rather than merely incomplete.

C1 and C2 are independent. C2 confirmed with C1 refuted would mean the
structure is real but too weak to matter — a genuine result, and reported as
one rather than as a partial win.

## The two preconditions, unchanged

**Falsifiability**, checked before scoring: synthetic routing swept across
stickiness and popularity skew, confirming both predictions can move. Anything
forced is reported as uninformative.

**Implementation validation**, before the policy is trusted:
* at capacity ≥ the whole key space, cyclic must equal LRU, LFU and Belady —
  nothing is ever evicted, so no policy can differ;
* transfers must lie between Belady and all-miss;
* with `L = 1` the cyclic distance is constant, so cyclic must reduce
  **exactly** to its tie-break, LRU — a degenerate case with a known answer,
  which is the cheapest way to catch a distance computed with the wrong sign.

A failure of any means the implementation is being scored, not the policy.

## Out of scope

Router scores. The full score vector over non-selected experts is the other
thing cache theory lacks, and it would likely predict recurrence better than
any structural rule — but the committed traces record only selected ids, so
testing it needs new captures. Registered here as the next candidate if this
one fails, and deliberately not mixed into this test.
