# RESULTS — training-step cost attribution

Registered in [`PREREG-train-cost-attribution.md`](PREREG-train-cost-attribution.md)
(stamped `1c27ab12…`, filed before the measurement). Receipts in
`train-attrib/`.

## Headline: in training, cold cost is **0% storage**

RTX 5090 + EPYC 9755 (Zen 5), OLMoE-1B-7B NF4 arena, 256-token microbatch,
forward + backward + AdamW, gradient checkpointing on, `hot_rows=512`,
gnf4 `ee1eb3d` / e4b `dc3712c`.

| arm | median step | Δ vs control | disk reads in window | tier hits | tier misses |
|---|---|---|---|---|---|
| control (0% cold) | 566.1 ms | — | 0 | 0 | 0 |
| cold-5 | 1140.3 ms | **+574 ms (+101%)** | **0** | 28,944 | **0** |
| cold-20 | 1889.5 ms | **+1323 ms (+234%)** | **0** | 61,920 | **0** |

**Every cold access is a tier hit. Zero disk reads. And the step still
doubles.** Forcing 5% of routed mass cold costs +101% of a training step
with storage contributing exactly nothing.

That is the decode finding, harder. In decode, storage was **5–11%** of cold
cost at 1–10% cold mass ([gate 1 addendum](RESULTS-tribrid-gate1.md)); in
training at this tier sizing it is **0%**, and the cold path costs an order
of magnitude more of the step.

## The clauses as registered

| clause | registered | measured | verdict |
|---|---|---|---|
| **T1** expert-weight *movement* is a minority of the step | <25% | **0%** (0 reads) | **PASS**, in the strongest form |
| **T2** >95% of experts routed per layer | >95% | **92.4%** (59.1/64, max 63) | **MISS on the letter** |
| **T3** tier hit rate >60% | >60% | **100%** (0 misses) | **PASS** |

**T2 is scored a miss.** 92.4% is not 95%, and the threshold was registered
before the run. The substance it was testing — that a training microbatch
leaves no cross-step locality to cache — holds comfortably at 92.4%, but the
clause is reported as filed rather than rounded into a pass.

**T3 passing at 100% is the load-bearing one for any port.** The training
path's docstring says "the tier is the recompute cache", and it is: with
gradient checkpointing a cold expert is touched three times per step
(forward, recompute, dgrad) and every touch after the first is a hit. The
forward→backward reuse is **algorithmic, not statistical**.

## What this means for porting Stage 3 to training

**Do not port the storage-oriented half.** Speculative prefetch, the
hide-ratio clause, deadline-scheduling of disk — all act on a term measured
at 0% here. The gate-1 lesson repeats exactly: an optimization aimed at
storage cannot move a wall that storage is not part of.

**Reclaimable residency is moot at this sizing, not disproven.** With
`hot_rows=512` holding a 265-row cold set there are **no evictions**, so
there is nothing to make reclaimable — `resurrections` and
`logical_evictions` are both 0 by construction. It becomes relevant only
where the tier cannot hold a step's routed set, which T2 says is the whole
arena. That is the configuration to test, and it was not tested here.

**The entire cost is the cold path's per-call staging.** +101% of a step for
5% cold mass, with zero I/O, is per-group setup — `_TieredStack` →
`segment_tensor` → per-group tensor construction — paid per layer, per pass,
three passes per step under checkpointing, at 92.4% routing density. That is
the same term the direct scatter attacks, and training is far deeper into
its regime than decode ever was (where it was worth −12.5% only at 20% cold).

**Priority order this implies**, replacing the ranking argued from theory:

1. **Cold-path per-call staging cost** — the whole delta. The direct scatter
   is the existing lever; whether it applies is gated on
   experts4bit-qlora#180, since a training run routing cold rows to the GPU
   cannot use an external landing.
2. **Tier sizing** — with no misses at `hot_rows=512`, the interesting
   question is what happens when the tier cannot hold the step, which is
   where both eviction and reclaimable residency start to exist.
3. Everything else, including the deadline model, which gate 2 already
   found loses to doing nothing in decode and has no storage term to
   schedule here.

## What this does not establish

One box, one model, one microbatch (256 tokens), one tier sizing. No
comparison against a non-hybrid training baseline, so the +101% is measured
against the hybrid control, not against ordinary QLoRA training. LoRA
adapters were the only trainable parameters. Instrument law 7 applies. The
backward's CPU kernel (`dgrad_nf4_grouped_cpu`) is a different kernel from
the forward GEMV, and nothing here fits constants for either.
