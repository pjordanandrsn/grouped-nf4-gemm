# Which row you evict barely matters: C1 and C2 refuted, and the reason rules out a whole family

Receipt: [`cyclic-2026-08-22/cyclic.json`](cyclic-2026-08-22/cyclic.json).
Harness: [`cyclic_policy.py`](routing-trace/cyclic_policy.py).
Preregistration: [`PREREG-cyclic-policy.md`](PREREG-cyclic-policy.md), merged
before measurement. 48 cells, no box.

| prediction | outcome |
|---|---|
| **C1** — cyclic closes ≥ 30% of the LRU→Belady gap on all four models | **REFUTED** |
| **C2** — cyclic beats LRU in ≥ 44 of 48 cells | **REFUTED** |

## The results

Fraction of the LRU-to-Belady gap closed:

| model | cyclic | LFU | ARC |
|---|---|---|---|
| OLMoE | **0%** | 49% | 6% |
| Granite | **3%** | 49% | 6% |
| Qwen1.5-MoE | **0%** | 30% | 5% |
| gpt-oss | **2%** | 2% | 6% |

Cyclic beats LRU in **23 of 48** cells — a coin flip — and its median
`cyclic ÷ LRU` is 0.984 to 1.000.

## The structural argument was right, and it did not matter

The registered reasoning was that LRU is *systematically* wrong here: under a
fixed layer walk, the least-recently-used row belongs to the layer about to be
visited next, so LRU evicts the row needed soonest. That reasoning is
**correct**, and the policy built on it does pick different victims — this was
checked rather than assumed:

| | evictions |
|---|---|
| cyclic and LRU chose the **same** layer | 1,143 |
| cyclic and LRU chose a **different** layer | **14,946** |

**93% of victim choices changed, and transfers moved by about 1%.** The
victim-layer distributions are visibly different too: cyclic concentrates on
the layers just processed, LRU on the layers about to be.

So the finding is not "LRU is better than expected". It is that **the ordering
decision is nearly irrelevant on this workload**, which is a much stronger and
more useful statement than the one that was registered.

## Why, and what it rules out

At these capacities the cycle length exceeds what the cache can hold, so a row
is evicted before its next use under *any* ordering. Which row leaves first
changes who loses, not how many lose. That is the same regime the zero-hit
region in [`routing-trace/RESULTS-crossover.md`](routing-trace/RESULTS-crossover.md)
describes, seen from the other side: below one step nothing survives, and just
above it, reordering the survivors does not change the count.

This retires a whole family at once. LRU, MRU, cyclic, ARC — and by the same
argument LIRS, 2Q, and any other policy whose lever is **the order in which
resident rows are discarded** — are all choosing among outcomes that differ by
~1%. Six of the eight things tried against this gap have now been ordering
policies or explanations of ordering.

It also explains, without a new hypothesis, why LFU is the one thing that
works where it works: LFU is not solving the ordering problem. It identifies
experts that recur across *many* steps and pins them, which shrinks the
effective working set until it fits — a different mechanism, and the only one
of the eight that has moved the number.

## Both preconditions were met

**Implementation validated** before the policy was trusted:

| check | result |
|---|---|
| capacity ≥ key space ⇒ cyclic = LRU = Belady | pass at L = 4 and L = 16 |
| Belady ≤ cyclic ≤ all-miss | pass, 30 random workloads |
| **L = 1 reduces exactly to LRU** | pass, 20 random workloads |

That last one exists because a distance computed with the wrong sign still
produces plausible numbers on real traces; with one layer every distance is
equal, so the policy must collapse onto its tie-break exactly.

**Both predictions were falsifiable**, checked before scoring: across twelve
synthetic conditions C1 would have been refuted in 12 and C2 in 10. The
refutations are decisions about the data, not artifacts of the harness.

## What this leaves

The ~1.9× gap stands. What has changed is where it can be attacked: not by
reordering evictions, which is now measured rather than assumed to be a dead
end. The remaining candidate named in the previous preregistration is the
router's **score vector over non-selected experts** — the margin by which an
expert missed the cut is information about whether it will be routed next
step, it is genuinely outside general cache theory, and it is the only lever
identified that is not an ordering rule. The committed traces do not record
scores, so testing it needs new captures.
