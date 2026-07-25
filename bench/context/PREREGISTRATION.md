# Preregistration — KV context work, experiments A and B

Hypotheses, predictions, and the analysis plan, **written before the runs**.
Committed ahead of results so the predictions can be scored rather than
narrated.

## Why this document exists

Across findings #6–#11 the pattern was consistent: predict, measure, discover
the prediction was wrong, then explain the result convincingly *after the fact*.
The scoreboard:

| prediction | outcome |
|---|---|
| values-only knee generalizes across architectures | **wrong** — 1.7× on Gemma-4 vs 6.4× on OLMoE (#10) |
| GQA amplifies key sensitivity (motivated the whole cross-arch run) | **wrong** — no trend at all (#10) |
| per-channel key scaling helps, since keys have channel outliers | **wrong** — 3.3× worse at equal bytes (#9) |
| low-rank composes with NF4 as a second axis | **wrong** — dominated everywhere (#6) |
| sparsity is the biggest remaining lever | **wrong** — 28× the quality cost at matched bytes (#11) |

Zero for five. Every explanation afterwards was mechanistically plausible, and
several were probably correct — but a hypothesis that is only stated after the
data cannot be wrong, which makes it worthless as evidence. The fixtures were
worse than useless twice, because each one confirmed the hypothesis built into
it (#7 iid fixture, #9 outlier fixture).

So: predictions below are numeric, falsification thresholds are stated, and the
analysis plan is fixed. If a result lands outside a stated interval, the entry
gets marked wrong rather than reinterpreted.

## Standing analysis rules (both experiments)

1. **Compare at matched BYTES, never matched token count or matched "ratio".**
   Any two schemes are compared at the memory they actually consume, because
   the whole point is a VRAM budget.
2. **A control arm always runs**: the same cache class with the feature
   disabled. Four architectures' control arms matching their fp16 reference
   exactly is what let the `get_query_offset` bug survive (#11) — controls are
   necessary, not sufficient.
3. **Report absolute baselines, not only deltas.** The broken sparsity run had
   plausible-looking deltas over a baseline of 330 where the truth was 5.97.
4. **A fixture that encodes the hypothesis proves nothing.** Where a fixture is
   built to contain the mechanism, it is labelled a mechanism test and is not
   evidence about real data.
5. No arm is dropped after seeing results. Arms listed here all get reported.

---

## Experiment A — attention-based selection (H2O-style) vs recency

**Motivation.** #11 measured *recency* eviction (sink + recent window) and found
it costs 28× more quality than quantization at matched bytes. Two confounds were
flagged in advance of that conclusion: (a) sink+recent is the weakest possible
selection rule, and (b) wikitext next-token prediction is the least favourable
task, since it depends on exactly the dense recent context the policy deletes.
A tests both.

**Fixtures.** Two, specified exactly:

- `wikitext`: as #11 — first 1024 tokens of wikitext-2 test, chunked teacher
  forcing at chunk 128, ppl over all predicted positions.
- `induction`: 256 token ids drawn uniformly from [1000, 20000), concatenated
  with themselves (512 tokens total). Perplexity measured **on the second copy
  only**. A model with functioning induction heads scores far lower on the
  second copy than the first; a recency window of 128 makes that impossible,
  because the matching first-copy token sits exactly 256 positions back. This
  isolates a *sparse, long-range* dependency, which is the regime eviction is
  supposed to serve.

**Arms.** full-fp16, full-nf4, recency (sink4+rec128), H2O (sink4 + top-K by
accumulated attention + rec64), each × {fp16, nf4} — on both fixtures. H2O
accumulates attention mass per (layer, kv head, key position), summed over query
positions and over the query heads mapping to each kv head, and selects at each
chunk boundary using only attention observed so far.

### Predictions (A)

- **A1.** On `induction`, recency eviction is catastrophic: ppl on the second
  copy within 25% of the *first* copy's ppl, i.e. induction destroyed.
  *Falsified if* recency retains more than half the full-cache induction gain.
- **A2.** On `induction`, H2O recovers most of it: second-copy ppl within **2×**
  of full-cache second-copy ppl, at a budget ≤ 25% of tokens.
  *Falsified if* H2O lands closer to recency than to full cache.
- **A3.** On `wikitext`, H2O beats recency but still loses to NF4 at matched
  bytes: H2O Δppl ∈ **[+0.4, +2.5]** against recency's +3.316 and NF4's +0.118.
  *Falsified if* H2O ≤ +0.3 (would mean selection alone closes the gap) or
  ≥ +3.3 (would mean selection buys nothing).
- **A4.** NF4 stays near-free on top of *any* selection rule, as in #11:
  incremental Δppl from quantizing ∈ **[+0.05, +0.40]** on both fixtures.
  *Falsified outside that interval.*

**Pre-committed decisions.** If A2 holds, selection-based eviction earns a place
as a real tier for long-range workloads and the #11 conclusion is scoped to
dense prediction rather than to sparsity as a whole. If A2 fails, token-axis
sparsity is closed out for this project and the remaining lever is quantization
granularity. If A3 is falsified downward, #11's headline needs rewriting.

**Known confounds, stated in advance.** (i) OLMoE-1B-7B is a 1B-active model and
may have weak induction heads; if full-cache induction gain is under 2× the
fixture cannot test A1/A2 and the experiment is void, not "negative". (ii) H2O
scoring needs `output_attentions`, which forces eager attention — this changes
speed only, not values, and speed is not measured here. (iii) Selecting at chunk
boundaries with chunk 128 is coarser than per-token H2O and should, if anything,
*understate* H2O.

---

## Experiment B — single-pass fused decode kernel

**Motivation.** The v1 fused-cache path costs 2.5–3× fp16 SDPA (15.42 vs 6.27 ms
at 32K), which is why `NF4KVCache` dequantizes per layer instead of using the
kernels. A single-pass online-softmax kernel is the standard fix.

**Correction to my earlier reasoning, recorded because it changes the
prediction.** I described the win as "halving cache traffic." That is wrong: the
two-pass path already loads K exactly once (scores) and V exactly once (weighted
sum). Fusing does not reduce *cache* traffic at all. What it removes is the
**scores intermediate**, and at decode shapes that term dominates:

at T=32768, H_q=64, H_kv=4, D=128 —
- packed K+V (nibbles + fp32 absmax): **18.9 MB**, read once either way
- scores `[64, 32768]` fp32: **8.39 MB**, touched ~4× in the two-pass path
  (scores writes, softmax reads+writes, weighted-sum reads) = **33.6 MB**
- two-pass total ≈ **52.5 MB**; fused ≈ **18.9 MB** (scores stay in registers)

So fused should cut traffic ~64%, and three kernel launches become one.

### Predictions (B)

- **B1.** Fused vs two-pass speedup at 32K: **1.8–3.0×**.
  *Falsified outside.* (Pure-memory-bound would give 2.8×; dequant ALU work does
  not shrink, so the low end allows for being partly compute-bound.)
- **B2.** Fused vs fp16 SDPA ratio at 32K: **0.8–1.4×** (i.e. roughly parity,
  possibly faster, because a 4-bit cache moves ~4× fewer bytes than bf16).
  *Falsified if* > 2.0× — that would mean the two-pass penalty was never about
  the scores intermediate and my traffic model is wrong.
- **B3.** The advantage grows with context: the fused/fp16 ratio at 32K is lower
  than at 4K, since the scores term scales with T.
  *Falsified if* the ratio is flat or worsens with T.
- **B4.** Numerics: fused agrees with two-pass to < 2e-3 relative, including at
  scale=1.0 with logits ~40× (where the running-max rescale is load-bearing).

**Pre-committed decisions.** If B2 holds at ≤ 1.4×, the fused path becomes the
default for decode in `NF4KVCache` and the "memory dial costs 2.5–3× latency"
caveat is removed from the docs. If B2 is falsified, the kernel stays
experimental and the docs keep the caveat.

**Known confounds.** (i) The A2000 is shared with home-lab services; timings run
with a warmup and take the median of ≥ 20 reps, and free VRAM is recorded.
(ii) Triton autotuning is not used, so `block_t` is fixed at 128 for both paths
— this is a like-for-like comparison, not a tuned-vs-untuned one.
(iii) fp16 SDPA is torch's kernel and is *not* being claimed as a fair
apples-to-apples baseline for a quantized path; it is the reference a user would
otherwise run.

---

## Scoring

Results land in `receipts-*/`, and each prediction above gets marked
**confirmed / falsified / void** in the finding that reports it, with the
measured value beside the predicted interval. Falsified predictions stay in this
document; they are not edited to match what happened.
