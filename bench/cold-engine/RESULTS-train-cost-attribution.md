# RESULTS — training-step cost attribution

Registered in [`PREREG-train-cost-attribution.md`](PREREG-train-cost-attribution.md)
(stamped `1c27ab12…`, filed before the measurement). Receipts in
`train-attrib/`.

## Headline: in training, cold cost is **0% storage**

RTX 5090 + EPYC 9755 (Zen 5), OLMoE-1B-7B NF4 arena, 256-token microbatch,
forward + backward + AdamW over **58.7M trainable LoRA parameters**,
`model.train()`, `use_cache=False`, gradient checkpointing on,
`hot_rows=512`, gnf4 `0a10eab` / e4b `36c0aee`.

| arm | median step | Δ vs control | disk reads | tier hits | tier misses | **disk share of Δ** |
|---|---|---|---|---|---|---|
| control (0% cold) | 634.0 ms | — | 0 | 0 | 0 | — |
| cold-5 | 1155.4 ms | **+521 ms (+82%)** | 7 | 28,013 | 7 | **0.1%** |
| cold-20 | 2069.2 ms | **+1435 ms (+226%)** | 8 | 61,108 | 8 | **0.0%** |

**Cold accesses are essentially all tier hits, disk time is 0.1% of the
cost, and the step still nearly doubles.** Forcing 5% of routed mass cold
costs +82% of a training step; storage accounts for 0.36 ms of the 521 ms.

Disk is charged at the box's measured **sequential** ceiling (5.67 GB/s
qd=16); the random peak (7.36) is deliberately ignored, since using it would
understate the disk share.

That is the decode finding, harder. In decode, storage was **5–11%** of cold
cost at 1–10% cold mass ([gate 1 addendum](RESULTS-tribrid-gate1.md)); in
training at this tier sizing it is **0%**, and the cold path costs an order
of magnitude more of the step.

## The clauses as registered

| clause | registered | measured | verdict |
|---|---|---|---|
| **T1** expert-weight *movement* is a minority of the step | <25% | **0.1%** of the cold delta | **PASS**, in the strongest form |
| **T2** >95% of experts routed per layer | >95% | **92.5%** (59.2/64, max 63) | **MISS on the letter** |
| **T3** tier hit rate >60% | >60% | **99.98%** (28,013 hits / 7 misses) | **PASS** |

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

## Post-hoc: tier stats were read from one attachment, not all of them

Bugbot found that both the warmup baseline and the post-window read walked
every module carrying a `_e4b_cold_tier` and kept the **last one**, so
hits/misses/evictions described a single attachment while `engaged` was 16.
Now summed over the DISTINCT tiers, with `reuse_before_overwrite` recomputed
from the summed numerator and denominator rather than averaged.

(A second finding on the same file — that the disk-time charge took the max
over every NVMe point and so used a random-QD ceiling, 8.08 GB/s against a
sequential best of 6.07 — was fixed when this work merged as #129 and is
already reflected above.)

**Neither moves a number in this document, and the reason is checkable.**

T1 reads **0%** because `reads_in_window` is **0**. The bandwidth constant
multiplies a zero numerator, so 0/6.07 and 0/8.08 are the same figure; the
ceiling defect would have mattered on an arm that actually read the disk and
these arms did not.

T3 reads **100%** (0 misses). A tier miss is by construction a row that is
not resident, which forces a read — and the aggregate `disk_reads`, taken
from `hy.cold_stats(model)` which was already summing across modules
correctly, is **0**. Zero aggregate reads means no tier anywhere took a miss,
whatever the per-tier accounting said. A hit rate cannot fall below 100% when
the miss denominator is provably empty.

Recorded rather than fixed quietly: a reader who finds these fixes later
should not have to work out whether the numbers above predate them.
