# PREREG — replacing the dequant oracle with a fused kernel

**Tier: CONFIRMATORY. Status: STAMPED before the kernel was written.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `3150ac7`,
e4b `claude/e4b-gemma-inflight-d41f93` @ `c26dd72`. Both local, unpushed.

## Why

G1 split #17's slowdown into **wrapper 1.138×** and **dequant 1.475×** at ctx
4096, so the dequant is the target. And the thing doing the dequantizing is
`dequant_kv_ref` — *"Reference dequant of a packed cache. Test oracle."* It is
named as an oracle and written as one:

```
packed.to(torch.int32)          full-size int32   (4x the output's bytes)
(b >> 4) & 0xF, b & 0xF         two more int32
torch.stack([hi, lo], dim=2)    another, 2x size
lut[codes]                      full-size float32 gather
absmax.repeat_interleave(64)    expands ONE scale into 64 copies, full-size fp32
vals * am                       another full-size fp32
.to(dtype)                      and another
```

Seven full-size intermediates, most of them 4-byte, for a 2-byte result. The
memory-bound floor for one layer's K+V at 4096×16×128 is ~43 MB of traffic —
about **0.15 ms** on this card. The measured dequant is **~6 ms per layer**.
Roughly **40× of headroom**, and none of it is inherent to 4-bit.

**How this differs from D1, which is the standing warning.** There a Triton
kernel lost to `scaled_dot_product_attention` by 11.6× — a vendor flash-decode
path at 72% of memory bandwidth. Here there is **no vendor path**: torch has no
NF4 dequant, and the competitor is our own oracle. Losing to cuDNN is ordinary;
losing to seven materialized intermediates would not be.

## Predictions

The card's run-to-run variance is **±12%** (G1: F1a re-measured 1.887 → 1.679),
so intervals are set wide enough that the verdict is not decided by noise.

- **H1a.** Standalone fused dequant vs `dequant_kv_ref`, `[4096, 16, 128]` →
  bf16: **≥ 5×** faster. *Falsified below 2×.*
- **H1b — gate.** Fused output is **bit-identical** to the reference,
  `torch.equal`, for bf16 and fp32. Both multiply fp32 LUT values by fp32
  scales and round once, so this is equality and not a tolerance. *Any mismatch
  voids H1a/H1c/H1d* — a faster wrong dequant is worthless, and the cache's
  byte-exactness tests all run through this path.
- **H1c.** The end-to-end dequant factor (nf4 / nf4-raw at 4096) falls from
  **1.475** to **≤ 1.20**. *Falsified above 1.35.*
- **H1d.** The end-to-end headline (nf4 / bf16 at 4096) falls from **~1.68** to
  **≤ 1.35**. *Falsified above 1.55.*

## Pre-committed decisions

- If **H1b** fails, the kernel is discarded outright regardless of speed.
- If **H1a** holds and **H1c** does not, the dequant was not the bottleneck the
  decomposition said it was, and G1's attribution gets revisited rather than the
  kernel getting tuned.
- If **H1c and H1d** hold, the fused path becomes the default, the reference
  stays as the test oracle it says it is, and **#17's headline is re-measured
  and restated** — the ~1.7–1.9× would then be a property of an oracle in the
  hot path rather than of 4-bit KV.

## Confounds, stated in advance

1. **Per-channel keys are not covered.** `token_group` scaling has a different
   absmax layout; the fused path will refuse it and fall back, as `evict_index`
   already does. Not a regression — that dial is off by default and measured
   worse (#9).
2. One device, one geometry. The floor argument (43 MB, ~0.15 ms) is bandwidth
   arithmetic on this card and does not transfer as a number.
3. #17's headline will need re-measuring if this lands, and that re-measurement
   inherits the same ±12% band — so it gets reported as a range, as #17's now is.

## Outcome — 4 for 4, and #17 has to be restated

| prediction | predicted | measured | verdict |
|---|---|---|---|
| H1b **gate** bit-identical | exact | exact, 5 shapes × 2 dtypes | **CONFIRMED** |
| H1a standalone speedup | ≥ 5× | **12.62×** (2.717 → 0.215 ms) | **CONFIRMED** |
| H1c dequant factor | ≤ 1.20 | **1.148** (was 1.475) | **CONFIRMED** |
| H1d end-to-end nf4/bf16 @4K | ≤ 1.35 | **1.133** (was 1.679) | **CONFIRMED** |

The gate ran first and on ragged and degenerate shapes — `777×3×64`,
`256×2×256`, `1×1×128` — because a dequant that is only correct on round
numbers is a dequant that is wrong in production.

Standalone: **7.9 → 99.9 GB/s**. Still only ~35% of the card's ~288 GB/s, so the
kernel is not tuned, merely written; there is more left. It was not pursued
because H1c and H1d already cleared.

**End-to-end at 4096:**

| cache | ms/step | vs bf16 |
|---|---:|---:|
| bf16 `DynamicCache` | 206.55 | 1.00× |
| NF4 resident | **233.97** | **1.133×** |
| NF4 streamed | 254.40 | 1.232× |

And the decomposition collapses: **path_overhead 0.987, dequant 1.148** — the
wrapper is now free and the arithmetic is what is left.

**Pre-committed decision fires: #17's headline is restated.** NF4 KV costs
**~1.13× at 4K and ~1.24× at 16K**, not the ~1.7–1.9× / ~2.2–2.6× that finding
published. That number was measuring **an oracle in the hot path**, which is
what this prereg suspected and why it was written. #17 is corrected in place
with the old figures kept and labelled.

**Two things this run does NOT establish, despite what its output prints.** The
harness re-prints F1a and G1b against their original intervals; those were
scored on pre-fix code and are **not** re-scored here. And the greedy-divergence
line reads "position None" (identical output) where the earlier run said
"position 1" — that is not evidence about the dequant, because **the prompt is
`torch.randint` with no seed**, so it differs between runs. Within a run all
arms share one prompt and the comparison is valid; across runs it is not. A
harness defect, recorded rather than read as a result.

**This reopens something that was closed.** E2 closed prefetch on the grounds
that "the transfer is 24.4 ms of a ~262 ms step — 9% — and machinery to hide 9%
costs more than 9%." That ratio has moved: with the dequant gone, the streamed
arm at 16384 exposes **270.6 ms of a 591 ms step — 46%**. The closure was
conditioned on a number that this change invalidated. Prefetch is **not**
reopened here, because doing so on the strength of an argument rather than a
registration is the mistake this document set exists to prevent — it would need
its own prereg, and it inherits three prior falsifications.
