# PREREG — KV context work: attention-selection (A) and fused decode (B)

**Tier: EXPLORATORY (first-look measurement, not a confirmatory replication).
Status: STAMPED.** Code under test: gnf4 `kernel/nf4-kv-cache` local branch
(`cd25df5..a489b71`), e4b `claude/e4b-gemma-inflight-d41f93` (`0cb3577..60e9db5`).
Both local, unpushed.

Hypotheses, predictions, and the analysis plan. Committed and stamped so the
predictions can be scored rather than narrated.

## Stamp-ordering disclosure — read before trusting any B1–B4 verdict

The repo rule is **"NO box fires before the stamp"** (see
`bench/homelab/PREREG-session4-replication.md`). That rule was **followed for
Experiment A and for B5**, which had not run when this document was stamped.

It was **not** followed for **B1–B4**. Those predictions were committed to git
(`a489b71`) before the benchmark ran, so their ordering is established *within
the repository*, but a git commit date is author-controlled and is not an
external timestamp. B1–B4 therefore carry **weaker evidential status than the
stamp on this file implies**, and the honest reading is: predictions written
before the run, ordering attested only by git.

Recorded rather than repaired, per the receipts convention — re-stamping to make
the sequence look clean would destroy the very property a stamp exists to
provide. B5 and Experiment A are covered properly.

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

### Amendment 3 — Experiment A pre-run specification

Written and stamped **before A ran**, because implementing A surfaced six places
where the text above does not determine what to do, and choosing after seeing
results is the exact failure this document exists to prevent. Nothing here
changes a prediction or an interval; it fixes free parameters and the
arithmetic used to score them.

1. **K is set so H2O and recency hold the same number of tokens.** The Arms
   paragraph names `top-K` without a value. Standing rule 1 is matched BYTES,
   and equal token counts at equal quantization *is* matched bytes, so
   K = 64: sink4 + rec64 + top64 = 132 = sink4 + rec128.
2. **A3 is scored at two budgets, both registered.** The Arms paragraph says
   recency = sink4+rec128, but A3's interval is anchored to "recency's +3.316",
   which is #11's sink4+**rec256** arm (+4.536 was rec128). The document is
   internally inconsistent and I am not going to pick the convenient reading
   after the fact. **Primary for A3: the 260-token pair** (recency sink4+rec256
   vs H2O sink4+rec64+top192), because that is where A3's own cited reference
   number lives. **Secondary: the 132-token pair**, because that is what Arms
   lists. Both are reported with verdicts.
3. **The keep-set is per LAYER, not per head.** The prereg accumulates attention
   per (layer, kv head, key position) but never says at what granularity
   selection happens. Per-head is not representable: the packed store is
   `[T, H, D]` with one token axis shared by every head. Scores are therefore
   summed over kv heads. This is coarser than published H2O and, like confound
   (iii), should if anything **understate** H2O.
4. **Induction runs 3 seeds (0, 1, 2); A1/A2 are scored on the mean**, per-seed
   values reported. The fixture is one draw of 256 random ids and the seed was
   never specified; a single draw invites a lucky or unlucky sequence to decide
   a threshold.
5. **Ratio thresholds are evaluated in log space.** A1 is falsified if recency
   "retains more than half the full-cache induction gain", where gain =
   ppl(first copy) / ppl(second copy); half of a multiplicative quantity is
   `log g_recency / log g_full > 0.5`. A2's "lands closer to recency than to
   full cache" is likewise a log-distance on second-copy ppl. Log space is the
   natural reading for a ratio and is the choice that makes **my own
   predictions easier to falsify** (a gain of 270x counts as "half" of 74,000x),
   which is why it is chosen rather than the linear alternative.
6. **A4 is read strictly**: the incremental Δppl from quantizing is computed for
   *every* registered policy arm on both fixtures — wikitext on all-token ppl,
   induction on second-copy ppl — and A4 is falsified if **any** value falls
   outside [+0.05, +0.40]. An absolute-ppl interval transfers badly between
   fixtures whose baselines differ by orders of magnitude; that is a defect in
   the prediction as written, and it gets marked rather than reinterpreted.

Two further disclosures, also pre-run:

- **A2's budget clause is missed by 0.8pp.** 132 tokens is 25.8% of the
  induction fixture's 512, against A2's "≤ 25%". A larger budget favours H2O, so
  a *failed* A2 at 25.8% would also have failed at 25%. If A2 **passes**, it is
  re-run at exactly 128 tokens before being called confirmed.
- **All arms run eager attention**, not just H2O's. Confound (ii) accepted a
  kernel difference between arms; there is no reason to, since the cost is speed
  and speed is not measured here. The consequence is that this experiment's
  full-fp16 baseline need not reproduce #11's 5.968 exactly, and it is reported
  as its own absolute number per standing rule 3.

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

## Outcome of B1–B4, and an amendment (B5)

Run before this section was written; results in
`receipts-*/fused_latency.json`.

| prediction | predicted | measured | verdict |
|---|---|---|---|
| B1 fused vs two-pass @32K | 1.8–3.0× | **0.79×** | **FALSIFIED** |
| B2 fused vs fp16 @32K | 0.8–1.4× | **2.66×** | **FALSIFIED** |
| B3 ratio improves with ctx | 32K < 4K | 2.66× < 3.64× | confirmed |
| B4 numerics < 2e-3 | — | passes, incl. scale=1.0 @ logits ~40× | confirmed |

**Why B1/B2 failed: the traffic model was not wrong, it was incomplete — it
counted bytes and ignored occupancy.** The fused kernel launches `grid=(H_q,)` =
64 programs, each looping over `T/BLOCK_T` = 256 token blocks *sequentially*.
The two-pass scores kernel launches `(H_q, cdiv(T, BLOCK_T))` = 16,384 programs.
On 26 SMs the fused version leaves the device largely idle while 64 programs
grind serial loops, which is why it wins slightly at 4K (32 iterations) and
loses at 32K (256). B3 confirms the scores-intermediate term is real; it is
simply dominated by the parallelism lost to fusing.

Not reinterpreted as a success: at 32K the single-pass kernel as written is the
wrong shape, and the "fused path becomes the default" decision does not trigger.

### B5 — flash-decoding (split token axis + combine)

The standard fix: partition the token axis across `S` programs, each producing a
partial `(m, l, acc)`, then a cheap second kernel merges them by log-sum-exp.
Parallelism becomes `H_q × S` while the scores intermediate still never reaches
memory — the partials are `[H_q, S, D]`, ~2 MB at S=8 against the 33.6 MB the
two-pass path moves. S is chosen to target ≥ ~500 programs.

Predictions, with lower confidence than B1/B2 carried — that pair was stated
with equal confidence and both failed:

- **B5a.** split-K vs two-pass @32K: **1.2–2.5×** faster.
  *Falsified if* ≤ 1.0× (i.e. still no better than two-pass).
- **B5b.** split-K vs fp16 SDPA @32K: **1.0–2.2×**.
  *Falsified if* > 2.66× (no better than the un-split fused kernel).
- **B5c.** The 4K case does **not** regress below the un-split fused kernel's
  1.11× vs two-pass. *Falsified if* it does.
- **B5d.** Numerics unchanged: agrees with two-pass < 2e-3, including the
  extreme-logit case. The combine step is a second place for the rescale to be
  wrong, so this is not a formality.

**Pre-committed decision.** If B5a and B5b both hold, the split path becomes the
decode default and the latency caveat comes out of the docs. If B5a fails, the
conclusion is that a 4-bit cache cannot be read competitively at decode shapes
on this hardware without a different data layout, and the memory dial keeps its
documented latency cost.

### Outcome of B5, and amendment 1 (B6)

| prediction | predicted | measured | verdict |
|---|---|---|---|
| B5a split vs two-pass @32K | 1.2–2.5× | **0.90×** | **FALSIFIED** |
| B5b split vs fp16 @32K | 1.0–2.2× | 2.63× | outside interval |
| B5c 4K no regression | ≥ 1.11× | **0.96×** | **FALSIFIED** |
| B5d numerics < 2e-3 | — | passes, incl. splits 1/2/3/8/64 | confirmed |

Split-K improved exactly what it targeted (17.32 → 15.46 ms at 32K, so the
occupancy diagnosis was directionally right) and still lost to the plain
two-pass path. **The pre-committed decision fires: two-pass stays the decode
default, the fused and split kernels are experimental, and the docs keep the
2.5–3× latency caveat.** That is not reopened by what follows.

**Diagnostic (exploratory, not a registered test).** At 32K the NF4 path moves
~19 MB against fp16's ~67 MB — 3.5× fewer bytes, 2.4× slower — so it was never
memory-bound and the B1/B5 traffic-and-occupancy models were both answering the
wrong question. Holding H_kv=4 and varying H_q, with **byte-identical inputs**:

| H_q | GQA | ms |
|---:|---|---:|
| 4 | 1:1 | 5.593 |
| 8 | 2:1 | 5.332 |
| 16 | 4:1 | 4.598 |
| 32 | 8:1 | 8.334 |
| 64 | 16:1 | 13.834 |

Below H_q=16 time falls as occupancy fills; above it, time scales linearly with
query heads at constant bytes. `grid=(H_q, ...)` makes each query head
re-dequantize the same kv bytes, so GQA 16:1 does **16× redundant dequant ALU**.
At 1:1 with 4 heads the kernel is 5.59 ms against fp16's 5.87 ms for 16× more
query heads.

### B6 — one program per KV head, dequantize once, batch the query heads

Restructure to `grid=(H_kv, block)`: dequantize each K/V block once, then
compute scores for all `GQA` query heads sharing it. Dequant work drops by the
GQA factor and the per-head GEMV becomes a small `[GQA, D] x [D, BLOCK_T]` GEMM.

**Track record disclosure:** B1, B2, B5a and B5c were all falsified, so my
predictions about this kernel's performance have been wrong four times. The
interval below is deliberately wide and the mechanism is measured rather than
modelled — but treat the point estimate as weakly held.

- **B6a.** GQA-batched vs two-pass @32K, H_q=64: **1.5–6.0×** faster.
  *Falsified if* ≤ 1.0×.
- **B6b.** GQA-batched vs fp16 SDPA @32K: **≤ 1.3×**, i.e. at or near parity.
  *Falsified if* > 2.0×.
- **B6c.** The speedup tracks the GQA ratio: gain at 16:1 exceeds gain at 4:1,
  since the redundancy removed is proportional to GQA.
  *Falsified if* the 4:1 gain is within 20% of the 16:1 gain.
- **B6d.** Numerics agree with two-pass < 2e-3, including extreme logits.

**Pre-committed decision.** If B6a and B6b both hold, the GQA-batched kernel
becomes the decode default and the latency caveat is removed. If B6a fails, the
kernel line is closed for this project — three registered attempts is enough,
and the conclusion stands that a 4-bit cache is a memory dial with a latency
cost on this hardware, not a free one.

### Outcome of B6 — confirmed

| prediction | predicted | measured | verdict |
|---|---|---|---|
| B6a gqa vs two-pass @32K 16:1 | 1.5–6.0× | **2.81×** | **CONFIRMED** |
| B6b gqa vs fp16 SDPA @32K | ≤ 1.3× | **0.82×** | **CONFIRMED** |
| B6c gain tracks GQA ratio | 16:1 > 1.2 × 4:1 | 2.81× vs 1.12× | **CONFIRMED** |
| B6d numerics < 2e-3 | — | passes at ieee; 8/8 | **CONFIRMED** |

T=32768, H_kv=4, D=128, A2000, median of 25 after warmup, block_t 128 for every
arm:

| H_q | GQA | two-pass | split | gqa-ieee | gqa-tf32 | fp16 SDPA |
|---:|---|---:|---:|---:|---:|---:|
| 16 | 4:1 | 5.943 ms | 5.572 | 5.317 | 2.030 | 1.158 |
| 64 | 16:1 | 13.997 ms | 15.950 | **4.975** | 2.355 | 6.055 |

**Pre-committed decision fires: the GQA-batched kernel becomes the decode
default and the 2.5–3× latency caveat is removed** — replaced by the measured
figure, which is regime-dependent (below).

B6c matters more than B6a: it validates the *diagnosis*, not just the outcome.
The gain is proportional to the redundancy removed, which is what
"dequant ran per query head" predicts and what a lucky tuning change would not.

**Scope, stated because the headline is easy to over-read.** The 0.82× holds at
GQA **16:1** and 32K. At 4:1 the same kernel is 1.12× vs two-pass and **4.59×
slower than fp16**, because there is little redundancy to remove and fp16 SDPA
at 16 heads is simply fast (1.158 ms). So: "a 4-bit cache can be read at or
below fp16 cost **in the high-GQA long-context regime**" — which is where
current models live (Qwen3 16:1, gpt-oss 8:1) — not "4-bit attention is free
everywhere". One device, one shape family.

`input_precision="ieee"` is the default and costs **2.11×** against tf32
(4.975 vs 2.355 ms). tf32 measured 2.7e-3 relative error at logits ~40×, which
is small but is a second error source stacked on quantization error; ieee keeps
the two separable. The knob is exposed and the cost is recorded rather than
buried.

## Scoring

Results land in `receipts-*/`, and each prediction above gets marked
**confirmed / falsified / void** in the finding that reports it, with the
measured value beside the predicted interval. Falsified predictions stay in this
document; they are not edited to match what happened.
