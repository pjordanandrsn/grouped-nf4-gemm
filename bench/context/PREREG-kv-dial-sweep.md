# PREREG — pricing the KV dial against a fast step

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
gnf4 @ `f039700`, e4b @ `3510248`. Both local, unpushed.

## Why now

#36 found attention at **44.5%** of the 235B step — nearly equal to expert
compute — at a **48-token** context where the KV is ~9 MB. That is not intrinsic:
every benchmark in this session ran `residence="host"`, so each attention call
pulls the cache across the link and dequantizes NF4.

#17 priced the NF4 KV dial at **1.7–1.9× at 4K** and #19 found it costing
**0.5%** at 235B/32K — but #19's step was **6.1 s**, dominated by 16× surplus
expert traffic that #22 then removed. Against today's **0.62 s** step the same
dial is a completely different fraction. Nothing has priced it here.

## Fixture

Qwen3-235B-A22B, routed+grouped+speculative (the current best), 3×3 sweep:
cache ∈ {`bf16` DynamicCache, `nf4_resident`, `nf4_host`} × context ∈
{48, 4096, 32768}. Median of 3, one load, peak VRAM recorded per cell.

## Predictions

- **K1a — host residence is expensive at short context.** At 48 tokens,
  `nf4_host / nf4_resident` ≥ **1.5×**. #36 attributes 0.28 s of a 0.62 s step to
  attention under host KV; moving 9 MB to the device should remove most of it.
  *Falsified below 1.15×* — which would mean #36's attention figure is intrinsic
  and the KV dial is not the lever.
- **K1b — NF4 costs time even resident.** At 48 tokens `nf4_resident / bf16` ∈
  **[1.0, 1.4]** — the dequant is real but small at 9 MB. *Falsified above 1.8.*
- **K1c — the host penalty grows with context.** `nf4_host / nf4_resident` at
  32768 > at 48. More KV means more bytes per attention call. *Falsified if it
  shrinks.*
- **K1d — memory ordering holds.** Peak VRAM `bf16` > `nf4_resident` >
  `nf4_host` at 32768. *Falsified on any inversion.*

## Pre-committed decisions

- **K1a confirmed** → `nf4_host` stops being the benchmark default; the session's
  measured ladder is re-stated against the *correct* KV setting, and the planner's
  KV dial carries this measured penalty rather than #17's isolated 1.7–1.9×.
- **K1a falsified** → attention's 44.5% is intrinsic, the KV dial is not the
  lever, and the next target is attention itself.
- **K1c falsified** → the host penalty is a fixed per-call cost rather than a
  bandwidth one, which points at the dequant/materialization path rather than the
  link.

## Confounds

1. One box, one link. The host↔resident gap is a link-rate quantity and will
   differ on faster PCIe.
2. `bf16` at 32768 needs ~6.2 GB of KV on top of ~20 GB of weights and buffers;
   if it OOMs that is reported, not worked around.

## Outcome — residence is not the lever; quantization is, and I read #36 wrong

Qwen3-235B-A22B, routed+grouped+speculative, median of 3.

| ctx | bf16 | nf4_resident | nf4_host | peak bf16 / nf4res / nf4host |
|---:|---:|---:|---:|---|
| 48 | **0.6209** | 0.7877 | 0.8134 | 19.83 / 19.83 / 19.82 GiB |
| 4096 | **0.5715** | 0.8139 | 0.6743 | 21.70 / 21.17 / 20.96 GiB |
| 32768 | **0.7421** | 0.8230 | 0.8982 | 43.71 / 39.49 / 37.84 GiB |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| K1a host/resident @48 | ≥1.5× | **1.033×** | **FALSIFIED** |
| K1b nf4_resident/bf16 @48 | [1.0, 1.4] | **1.269×** | **CONFIRMED** |
| K1c host penalty grows | — | 1.033 → 1.091 | **CONFIRMED** |
| K1d peak ordering @32K | — | 43.71 > 39.49 > 37.84 | **CONFIRMED** |

**Host residence costs almost nothing** — 3.3% at 48 tokens, 9.1% at 32K — and at
4096 the ordering *inverts* (host 0.6743 **faster** than resident 0.8139). With
3 reps against the 10–35% spreads seen elsewhere in this session, resident-vs-host
is **not separable from noise** and should not be ranked.

**What is separable: bf16 beats NF4 at every context, 3 for 3** — 1.27× at 48,
1.42× at 4096, 1.11× at 32K. **The cost is the dequantization, not the
residence.** That is #17's 1.7–1.9× reappearing at a smaller magnitude against a
faster step.

### Correcting #36

#36 attributed attention's 44.5% to the host-KV setting and said "roughly half
the step was paying for it". **Wrong.** Moving to the best KV setting buys
**1.31×** at 48 tokens (0.8134 → 0.6209), not ~1.8×. Most of that attention cost
is **intrinsic**, and the pre-committed branch for a falsified K1a fires:
**attention itself is the next target, not the cache.**

### What the dial actually trades

At 32K on this model, NF4 KV saves **4.2 GB** (43.71 → 39.49) for **1.11×** time,
and host residence saves a further **1.65 GB** for another **1.09×**. Both are
real trades and neither is free — but on an 80 GB card at 32K, **bf16 is simply
the right answer**, and every benchmark in this session ran the wrong one.

**Cross-pod caution:** `nf4_host` at 48 tokens measured 0.8134 here against
0.6263 in #34 on a different host. Absolute numbers do not transfer between pods;
only within-pod ratios do, which is what is reported above.
