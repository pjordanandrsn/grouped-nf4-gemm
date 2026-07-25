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

## Amendment 1 — H2: tune it. Registered before the tuning.

99.9 GB/s is ~35% of this card's ~288. Two things in the kernel are obviously
wrong for a bandwidth-bound job, and naming them before measuring keeps the
result from being a post-hoc story:

1. **The stores are strided.** Elements land at `2i` and `2i+1` in two separate
   `tl.store`s, so each touches every other address. Indexing the *output* by
   element and gathering the byte (`j // 2`) instead reads each byte twice out
   of L1 and stores one contiguous run.
2. **The programs are tiny.** `ROWS=4` at D=128 loads 256 bytes and stores 1 KB,
   over a 16,384-program grid — launch and index arithmetic against almost no
   work.

- **H2a.** Tuned fused dequant reaches **≥ 150 GB/s** (≥ ~52% of peak) at
  `[4096,16,128]` → bf16. *Falsified below 120 GB/s.*
- **H2b — gate.** Still bit-identical to `dequant_kv_ref`, same shapes as H1b.
  *Any mismatch and the tuning is discarded.*

The config search itself is a search, not a hypothesis: the sweep is reported in
full, and the chosen config is pinned in source with its measurement beside it
rather than left to `triton.autotune`, matching the choice #12 made for
comparability.

## Outcome of H2 — falsified, and the remaining bottleneck is identified not fixed

| prediction | predicted | measured | verdict |
|---|---|---|---|
| H2b **gate** bit-identical | exact | exact, 4 shapes × 2 dtypes | **CONFIRMED** |
| H2a tuned bandwidth | ≥ 150 GB/s | **113.1 GB/s** | **FALSIFIED** |

Both named defects were real and fixing them bought less than predicted.
Contiguous stores plus a 28-point sweep over `rows_per_prog` × `num_warps` moved
99.9 → **113.1 GB/s** (+13%), against the ≥150 registered. Winner **rows=8,
warps=1**, pinned in source with the sweep in the receipts.

Sweep shape, which is itself informative: performance peaks at 8 rows and falls
off in *both* directions — 1 row starves the machine (72.5 GB/s), 32+ rows spill
(97 and below). And `num_warps=1` wins at every row count, which says the kernel
is not warp-parallel-limited.

**The bottleneck is the codebook gather, diagnosed and NOT fixed.** Every output
element does `tl.load(lut_ptr + code)` — 8.4M indexed loads per call against a
16-entry table. That is a random access per element, and no amount of store
coalescing or launch shaping removes it. The standard fix is to evaluate the
16-entry codebook with a 4-level tree of `tl.where` selects and drop the gather
entirely.

Not attempted here, and the reason is scope rather than doubt: H1 had already
cleared its thresholds, the end-to-end number is now 1.13×, and a further 2×
on the dequant would move that to roughly 1.07×. Recorded as identified
headroom with the mechanism named, so the next person does not have to
rediscover it.

The reference re-measured at **3.052 ms** here against 2.717 ms in H1 — the same
±12% band. The honest speedup is therefore **~13–16×**, not a point estimate.

## Amendment 2 — H3: replace the codebook gather with a select tree

Registered before it was written. H2 named the bottleneck and did not fix it:
every output element does `tl.load(lut_ptr + code)`, **8.4M indexed loads per
call** against a 16-entry table. The codebook is small enough to evaluate in
registers — four levels of `tl.where` on the bits of `code`, 15 selects, no
memory touched.

**Track record.** I have now predicted this kernel's bandwidth once and been
wrong: H2a said ≥150 GB/s and measured 113.1. The floor argument says the DRAM
traffic is ~21.5 MB, which at this card's ~288 GB/s is 0.075 ms — so ~287 GB/s
is the ceiling and there is room. Having room is not the same as reaching it,
which is exactly what H2a got wrong.

- **H3a.** Bandwidth ≥ **150 GB/s** at `[4096,16,128]` → bf16.
  *Falsified below 130* — 113.1 plus this card's ±12% is ~127, so anything under
  130 is indistinguishable from H2's kernel.
- **H3b — gate.** Bit-identical to `dequant_kv_ref`, same shapes as H1b/H2b.
  The 16 constants are read from `NF4_LUT` at call time and passed in, never
  transcribed, so the tree cannot silently disagree with the reference about the
  codebook. *Any mismatch discards it.*

Reported, not predicted: end-to-end nf4/bf16 at 4096, currently 1.133×.

## Outcome of H3 — falsified, and the diagnosis was wrong

| prediction | predicted | measured | verdict |
|---|---|---|---|
| H3b **gate** bit-identical | exact | exact, 4 shapes × 2 dtypes | **CONFIRMED** |
| H3a bandwidth | ≥ 150 GB/s | **92.8 GB/s** | **FALSIFIED** |

Not merely short of the target — **worse than the kernel it replaced**. The
gather version reached 113.1 GB/s over the same 28-point sweep; the select tree
peaked at 92.8 (rows=8, warps=2). **Reverted.** The gather is the shipped kernel.

**H2 named the codebook gather as the bottleneck and H2 was wrong.** Fifteen
`tl.where`s per element are 126M select operations per call, while the gather
they replaced reads a **16-entry table that lives in L1** — a broadcast hit, not
a trip to DRAM. Trading memory for ALU is the right instinct when the memory is
far away, and this memory was not far away.

So the remaining headroom from 113 GB/s toward the ~287 the byte count allows is
**unexplained**, and no third diagnosis is offered here. Two have now been
tested: strided stores (real, worth +13%) and the codebook gather (wrong, −18%).

**Config drift, recorded rather than chased.** Re-sweeping the reverted kernel
puts the peak at **117.3 GB/s at rows=8, warps=2**, where H2's sweep of the
same code said 113.1 at rows=8, warps=1. The peak is a broad plateau and the
argmax wanders inside this card's ±12%; the pinned `rows=8` is on that plateau
under every sweep taken and is left alone. Re-pinning to whichever config won
the most recent noisy sweep would be fitting the noise.

**Running tally on this kernel, since it is the point of keeping score:** H1a
confirmed (12.6×), H2a falsified (113 vs ≥150), H3a falsified (92.8 vs ≥150).
Both performance *diagnoses* I have offered for it have been half right and
fully wrong respectively, while the one prediction grounded in an existing
measurement rather than a mechanism — H1 — landed. That is the same pattern F1
noted and it is now three preregs deep.
