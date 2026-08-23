# Preregistration — can an adaptive policy stop us needing the explanation?

Registered before any measurement. No adaptive policy has been run on these
traces.

## Why stop explaining

[`RESULTS-policy-headroom.md`](RESULTS-policy-headroom.md) found a
frequency-aware victim rule closes about half the LRU-to-optimal gap on OLMoE
and Granite, a third on Qwen, and **2% on gpt-oss** — the model every wall
measurement was made on. Five single-variable explanations for that split are
now dead:

| candidate | verdict | where |
|---|---|---|
| popularity skew | refuted | [`RESULTS-frequency-signal.md`](RESULTS-frequency-signal.md) |
| working-set pressure | refuted | same |
| reuse distance | refuted | same |
| `k/E` | ruled out by inspection | OLMoE and gpt-oss share 0.125 at opposite ends |
| **top-k** | **refuted, preregistered** | [`RESULTS-topk-frequency.md`](RESULTS-topk-frequency.md) |

Both of the last two were registered before measurement and both failed. After
five, the prior on a sixth single-variable story is low, and this program has
already recorded what happens when a rule is fitted to four points.

So the question changes from *why* frequency helps on some models to whether a
policy can **avoid needing to know**. That is an engineering question with a
falsifiable answer, and it is the first thing in this line that could change
what ships.

## The policy

**ARC** (Adaptive Replacement Cache): two lists, one recency-ordered and one
frequency-ordered, with ghost entries recording what was recently evicted from
each, and a target split `p` that moves toward whichever list is generating
ghost hits. It arbitrates exactly the axis on which these models disagree, it
is online, and it is implementable — none of which is true of Belady.

Scored against **LRU**, **LFU** and **Belady** on the 48 published cells: four
models × four prompts × three capacities, transfers, exactly as
`policy_headroom.py` already measures them.

## A1 — ARC is never worse than the better of the two

> Per model, the median `ARC ÷ min(LRU, LFU)` is **≤ 1.02**.

* **Confirmed** if that holds for all four models.
* **Refuted** by any model where ARC is more than 2% worse than whichever
  fixed policy wins there. Adaptivity that loses to both alternatives it is
  adapting between is not worth shipping, and 2% is the slack for ARC paying a
  small price for not knowing in advance.

## A2 — ARC helps where the fixed policy does not

This is the one that matters, and A1 can pass while it fails.

> On **gpt-oss**, ARC closes **≥ 15%** of the LRU-to-Belady gap.

LFU closes 2% there. If ARC also closes ~2%, adaptivity buys nothing on the
model the wall measurements were made on, and the idea is dead regardless of
how it does on OLMoE and Granite.

* **Confirmed** at ≥ 15%.
* **Refuted** below it, and reported as "adaptivity does not rescue gpt-oss"
  rather than softened into a partial result.
* A1 confirmed with A2 refuted is a **negative** outcome overall: it would mean
  ARC is safe but pointless, and the ~1.9× gap stays open.

## Two checks the protocol requires before any of this is reported

**Falsifiability.** Both predictions get the treatment that
[`structural_check.py`](routing-trace/structural_check.py) applies elsewhere:
synthetic routing swept across stickiness and popularity skew, to confirm the
verdicts can move. This program has published a confirmation that no input
could have refuted, and will not do it twice. If A1 or A2 turns out to be
forced by the harness rather than decided by the data, it is reported as
uninformative and the confirmation is withdrawn.

**Implementation.** ARC is subtle enough to get wrong in ways that still
produce plausible numbers, so it is validated before it is trusted:
* at capacity ≥ the whole key space, ARC, LRU, LFU and Belady must all reach
  the same fill count — everything fits, so no policy can differ;
* ARC's transfers must never exceed the all-miss count, nor fall below
  Belady's;
* the ghost lists must never let resident-plus-ghost exceed `2c`, which is
  ARC's own invariant and the thing a broken port silently violates.

A failure of any of these means the implementation is scored, not the policy.

## What would count as a miss

* A1 refuted ⇒ ARC is not safe to prefer over the fixed rules; report and stop.
* A2 refuted ⇒ adaptivity does not solve the gpt-oss problem, which is the
  problem. Reported first, plainly, whatever A1 did.
* Either prediction found to be unfalsifiable ⇒ reported as uninformative.
* LIRS, 2Q, and any tuned hybrid are **out of scope**. One policy, registered
  in advance; trying several and reporting the winner is the failure mode this
  file exists to avoid.
