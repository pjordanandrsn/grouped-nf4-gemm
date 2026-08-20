# The device row cache on a real routing sequence

Artifacts: [`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl) (the trace),
[`replay_dev_cache.py`](replay_dev_cache.py) (the harness),
[`replay.json`](replay.json) (the receipt).

`RESULTS-dev-row-cache.md` validated the cache's mechanism on a fixture whose
arena held **every expert in the layer** — the best case that exists — and
said so. This is the measurement it named as the honest next one, and it
found the shipped defaults giving away nearly all of the available win.

## The trace

512 autoregressive decode steps of **OLMoE-1B-7B-0924**, 16 layers, top-8 of
64 — the per-step routed expert ids, not aggregate mass. Captured by hooking
every router and reading its own top-k index tensor.

The committed `olmoe_profile.jsonl` is `tokens_routed` per `(layer, expert)`.
That cannot answer a caching question: a cache lives or dies on **when** an
expert is routed again, and a total has no order in it. Decode is
autoregressive on purpose — teacher-forced routing is not what a served model
sees, because after the first token the routing conditions on what the model
itself generated.

Reusable beyond this: R4 is registered against a real captured routing
*sequence* and has not had one until now.

## The bar is not zero

| | device transfers |
|---|---|
| no cache | 65,536 |
| **positional cache the engine already has** | **56,408 (86.1%)** |

`slots64` row *i* already holds whatever expert routed to position *i* last
step, and the gather's address test skips it. Any new cache has to beat
86.1%, not 100%. (The positional figure here is *optimistic* — the trace
stores each step's routed set sorted, which makes position more stable than
the router's own top-k order that the engine actually uses. Being generous to
the incumbent is the conservative direction.)

## As shipped, the cache lost to the cache we already had

| rows | % of the 1024 (layer,expert) pairs | transfers | vs positional |
|---|---|---|---|
| 128 | 12.5% | 61,110 | **108.3%** |
| 192 | 18.8% | 58,202 | 103.2% |
| 256 | 25.0% | 54,819 | 97.2% |
| 384 | 37.5% | 49,708 | 88.1% |
| 1024 | 100% | 26,642 | 47.2% |

At the sizes anyone would actually give it, **it was worse than not adding
it**. And the last row is the tell: with capacity equal to the entire working
set, a correct cache makes only the ~1024 compulsory fills. It made 26,642 —
evicting rows it never needed to evict.

## Two defects, separated by measurement

**1. `protected` defaulted to `rows // 2`.** `_demote` reduces the ACTIVE set
to `protected`, so half the arena churned regardless of capacity. At 1024
rows, raising `protected` alone took fills from 26,642 → **989**, which is
the compulsory floor:

| protected (of 1024) | fills |
|---|---|
| 512 (the default) | 26,642 |
| 768 | 9,757 |
| 896 | 3,775 |
| 1000 | **989** |
| ideal LRU | 989 |

The demotable margin only ever has to absorb **one request**. Every row
beyond it is retention being paid for in VRAM and thrown away. The default is
now `rows - routed`.

**2. `VramSlots` incremented a `_clock` it never read.** Victim selection and
allocation both walked `range(n_slots)`, so eviction was by slot **index**.
With `protected` fixed, index order still cost 1.12–1.33× more fills than an
ideal LRU of the same capacity. Both now pick least-recently-used within each
preference class, and a hit refreshes recency.

## After both fixes it tracks ideal LRU

| rows | % of pairs | transfers | vs positional | ideal LRU | gap |
|---|---|---|---|---|---|
| 128 | 12.5% | 43,338 | **76.8%** | 49,697 | 0.87× |
| 192 | 18.8% | 42,719 | 75.7% | 42,811 | 1.00× |
| 256 | 25.0% | 36,868 | 65.4% | 36,951 | 1.00× |
| 384 | 37.5% | 26,468 | **46.9%** | 26,613 | 0.99× |
| 512 | 50.0% | 17,438 | 30.9% | 17,559 | 0.99× |
| 1024 | 100% | 989 | 1.8% | 989 | 1.00× |

At **12.5% of the expert set resident it removes 23% of the device transfers
the positional cache still makes**, and it matches ideal LRU everywhere. It
beats LRU slightly at the smallest size because protected rows survive a pass
that pure LRU would evict — segmented-LRU behaviour, not an error.

## What this is not

**Not a latency measurement.** This counts transfers. Whether removing 23% of
them moves wall time depends on the extra device-side write each miss now
costs, and on what else the step is bound by — the cold path's dominant term
is per-call software cost, not bytes. Untimed here.

**Not a claim about other models.** One model, one prompt, 512 steps. OLMoE
routes top-8 of 64 with high churn; a model with sharper routing locality
would do better and a flatter one worse.

**Weights are irrelevant here and that is deliberate.** The replay drives the
real `DevRowCache` with real routing; what it does not exercise is the
engine's fill path, which `RESULTS-dev-row-cache.md` covers on GPU.
