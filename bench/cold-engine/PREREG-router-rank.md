# Preregistration — does the router's own confidence predict recurrence?

Registered before measurement. No trace records rank; the captures this needs
do not exist yet.

## The last lever

[`RESULTS-cyclic-policy.md`](RESULTS-cyclic-policy.md) retired a family: LRU,
MRU, cyclic and ARC differ in **93% of victim choices** and in **~1% of
transfers**, so the order in which resident rows are discarded is close to
irrelevant on this workload. LFU is the exception, and only because it is not
an ordering rule at all — it pins experts recurring across many steps and
shrinks the effective working set.

That leaves one identified lever that is neither ordering nor already tried:
**the router's own confidence**. When layer ℓ selects its top-k, it ranks all
E experts. An expert selected *first* is a different proposition from one that
scraped in at rank k, and nothing in the cache sees the difference.

**The traces have been discarding this all along.** `capture_routing.py`
writes `sorted(idx.tolist())` — the routed ids in ascending numeric order.
Rank order costs exactly the same bytes and was thrown away.

## What gets captured

Two **additive** fields, both cheap:

* `routed_rank` — the same ids in rank order (score-descending);
* `near_miss` — the ids ranked `k+1 … 2k`, the experts that did not make the
  cut and are the natural candidates to arrive next.

`routed` keeps its existing sorted order and is not touched. That is not
tidiness: `positional_transfers` compares routed sets **by index position**,
and `replay_dev_cache.py` documents that the sorted order deliberately makes
the positional baseline optimistic — "being generous to the incumbent is the
conservative direction for the claim being made". Rewriting `routed` in rank
order would silently change the denominator of every ratio this program has
published. New information goes in new fields.

Availability is online and needs saying. At layer ℓ of step *t* the engine has
layer ℓ's scores because its router just ran; for every other layer it has
that layer's scores from its most recent visit, one step ago. So a policy may
use "rank at this layer's last visit" without seeing the future. That is the
whole point — it is information the engine already has and the cache ignores.

## S1 — rank predicts recurrence

The premise, and it is checkable without building any policy.

> An expert selected at rank 1 at layer ℓ's last visit recurs at ℓ's next
> visit **at least 1.5× as often** as one selected at rank k.

* **Confirmed** if the ratio is ≥ 1.5 on all four models.
* **Refuted** otherwise — and then S2 is not worth running, because a policy
  cannot exploit a signal that is not there. Reported as "rank does not
  predict recurrence", not as a policy failure.

This is registered first deliberately. The last two experiments each built a
policy on an argument and then discovered the argument's premise was the thing
to have measured.

## S2 — a rank-aware policy closes the gap

> Evicting by rank-at-last-visit — highest rank retained, lowest evicted, LRU
> as tie-break — closes **≥ 30%** of the LRU-to-Belady gap on **all four**
> models.

Same bar as the cyclic policy, and for the same reason: 30% on every model, not
half the gap on three and nothing on gpt-oss, which is the per-model story this
program is trying to escape.

* **Refuted** if any model falls below 30%.
* **S1 confirmed with S2 refuted** is a real and interesting outcome: rank
  carries information that a replacement policy cannot convert into transfers,
  which would say the gap is not reachable by *any* eviction rule, ordering or
  otherwise. Reported as such rather than as a near-miss.

## The two preconditions, unchanged

**Falsifiability** of both predictions, checked before scoring, by the
synthetic sweep. Anything forced is reported as uninformative.

**Implementation validation** before the policy is trusted:
* capacity ≥ key space ⇒ rank-aware equals LRU, LFU and Belady;
* transfers between Belady and all-miss;
* with all ranks equal, the policy must reduce **exactly** to its LRU
  tie-break — the degenerate case that catches a comparison with the wrong
  sign.

**And one specific to this capture:** the rank-ordered ids must be a
permutation of the sorted ids the old path produced for the same model,
prompt and step. Same experts, different order. If that fails, the capture is
reordering something other than what it claims to.

## Out of scope

The full E-wide score vector. The near-miss band is the actionable part of it
at 2× the trace size rather than 8×, and a policy that cannot use ranks
`1…2k` is unlikely to be rescued by ranks `2k…E`. If S1 confirms and S2
refutes, the full vector is the next thing to consider — not before.
