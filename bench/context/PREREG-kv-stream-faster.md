# PREREG — making the streamed KV tier faster: what moves the bound, what moves the constant

**Tier: CONFIRMATORY. Status: STAMPED before any of it was built.** The
predictions below are derived from the model validated in finding #15
(`t = bytes / link`) and from measurements already in
`receipts-stream-20260725/`. **No implementation existed when this was written**,
which is the point — writing predictions after building is how the last five
post-hoc explanations happened.

Code under test: e4b `claude/e4b-gemma-inflight-d41f93` @ `02f29ea`,
gnf4 `kernel/nf4-kv-cache` @ `ebe6edc`. Both local, unpushed.

## The frame

Attention reads the **whole** cache every step, so a streamed tier is bounded by

```
ceiling = link / KV(ctx)
```

and only three things move that bound: **move fewer bytes**, **keep some
resident**, or **get a faster link**. Everything else — prefetch, kernel fusion,
overhead removal — improves the *constant* and lets you approach the bound. Both
kinds are worth having; conflating them is how a 1.3× constant-factor win gets
sold as lifting a ceiling. Each hypothesis below is labelled which it is.

## Methodology fix, carried over from #15

#15's harness scored on `t_host − t_gpu`: a difference of two separately-timed
loops carrying **~±15 ms** of noise. That was fine against a 286 ms overhead and
useless against a 36 ms one — S3 confirmed at 3.961 on one run and would have
falsified at 4.970 on the next **from identical code**.

So nothing here is scored on a difference of two point measurements. Every
transfer claim is scored on a **fit across ≥ 4 points**, `t = c + bytes/B`,
reporting both the effective bandwidth `B` and the per-step constant `c` — the
same shape `pcie_probe.py` uses. A fit over four points tolerates the noise floor
that a subtraction of two does not.

## Baselines (measured, not assumed)

94L × 4kv × 128d (Qwen3-235B geometry), A2000, `receipts-stream-20260725`:

| quantity | value |
|---|---:|
| packed KV @32K | 1.774 GB (576 B/layer/token) |
| link, measured asymptote | 6.20 GB/s |
| resident load, 94 layers @32K (this is the **dequant**) | 967.5 / 968.1 ms |
| streamed load @32K | 1256.1 / 1254.5 ms |
| ⇒ transfer @32K | 288.6 / 286.4 ms (predicted 286.2) |
| ⇒ per layer @32K | dequant **10.3 ms**, transfer **3.05 ms** |

---

## A1 — split residency (**moves the bound**) — RUNS THIS CYCLE

Today residence is binary, so free VRAM sits unused. Keeping the oldest `f` of
the cache resident and streaming the rest should give `ceiling = link/((1−f)·KV)`.
Oldest, not newest: the head is positionally stable, so nothing reshuffles as the
cache grows.

- **A1a.** Fit overhead against **streamed** bytes across f ∈ {0, 0.25, 0.5,
  0.75} at ctx 32768. Fitted `B` ∈ **[5.0, 7.5] GB/s** — i.e. the same law, with
  fewer bytes. *Falsified outside.*
- **A1b.** Fitted per-step constant `c` < **25 ms**. *Falsified at ≥ 25 ms* —
  that would mean splitting introduces a fixed cost that eats the saving at
  small f.
- **A1c.** Correctness: the split cache returns **byte-identical** K/V to a
  fully-resident cache at every f. `torch.equal`, not a tolerance. *Falsified by
  any mismatch* — and a failure here voids A1a/A1b rather than merely failing.
- **A1d.** GPU peak scales with f: peak(f=0.5) / peak(f=0) ∈ **[0.4, 0.65]**.
  *Falsified outside.*

**Pre-committed decision.** If A1a and A1c hold, split residency becomes the
default shape of the streamed tier and the binary switch is retired — it is
strictly dominated, since f=0 and f=1 are its endpoints.

## B1 — prefetch (**moves the constant**) — RUNS THIS CYCLE

Overlap layer L+1's copy with layer L's compute on a side stream, converting
`compute + transfer` into `max(compute, transfer)`. The machinery exists in
`offload.py`. Per layer at 32K the dequant is **10.3 ms** against a **3.05 ms**
transfer, so the transfer should hide **completely**.

- **B1a.** Prefetched streamed load / resident load at 32K ∈ **[1.00, 1.15]** —
  the transfer essentially free. *Falsified above 1.25.*
- **B1b.** Prefetched / non-prefetched streamed load at 32K ∈ **[1.20, 1.35]**
  (predicted 1.295 = 1254/968). *Falsified outside.*
- **B1c.** Double-buffering costs one extra layer resident: GPU peak rises by
  **< 100 MB** against non-prefetched. *Falsified at ≥ 100 MB.*

**Stated in advance, not scored:** B1's win is bounded by the compute it hides
behind. If D1 later removes the dequant, B1's benefit shrinks toward zero — the
two are **substitutes, not complements**, and a future D1 result must not be
combined with B1's as though they add.

**Confound.** This harness's "compute" is dequantization, not real attention. So
B1 tests hiding transfer behind dequant, which is real work on the real path, but
it is not the same as hiding it behind an attention kernel. Stated now so a
later real-model number is not silently scored against B1.

## A2 — eviction, repriced (**moves the bound**) — REFRAMED, NO NEW RUN

#13 closed token-axis sparsity: quantization dominates eviction ~9× at matched
bytes. That ratio is a quality-per-byte figure and is **unchanged** by
streaming — so #13's verdict is not reopened.

What changes is what eviction *competes with*. Resident bytes cost VRAM once;
streamed bytes cost PCIe **every step**. Once you are at NF4 and cannot compress
further without a fidelity cliff, eviction is the only remaining byte-reduction
axis — and in the streamed regime its competitor is not quantization, it is
**"cannot meet the latency target at all"**. Against that alternative the
relevant number is not the 9× ratio but the absolute quality cost at the budget
streaming imposes, which #11 and #13 already measured (+3.33 ppl at 260 tokens,
+1.09 with H2O selection).

No new experiment is registered because none is needed: the curve exists, and
what was wrong was the *pricing*, not the measurement. Recorded so this is not
mistaken for #13 being quietly reopened.

## Registered but DEFERRED, with the gate stated

Predictions committed now so they cannot be written after the fact; neither runs
this cycle.

- **C1 — fp16 absmax.** 0.5 + 4/64 = 0.5625 B/elem today; fp16 gives 0.53125.
  **C1a:** streamed bytes fall by exactly **5.56%**. **C1b:** wikitext ppl
  changes by **< 0.01** (the absmax is a scale, and fp16 carries ~3 decimal
  digits of it). *Gate:* the packed layout is read by every kernel in
  `kernel/`, so this is a format change with a blast radius far larger than its
  5.6%, and it does not go first.
- **D1 — fused attend on the streamed packed bytes.** `attend_nf4_kv_gqa`
  (#12) reads nibbles in the mainloop and never materializes a bf16 layer.
  **D1a:** GPU peak per layer drops by the bf16 materialization (**67 MB** at
  32K). **D1b:** at GQA 16:1 the streamed path's per-layer cost drops toward the
  kernel's 0.82× fp16 rather than dequant + SDPA. *Gate:* needs a real attention
  integration, which is per-architecture, and #12 measured the same kernel
  **4.59× slower** at GQA 4:1 — so this is a regime-dependent build, not a
  drop-in.
- **B2 — per-call overhead.** Not registered with an interval, deliberately:
  the ~570 µs/append measured in #15 is dominated by a Python layer loop that
  belongs to `transformers`, not to this code, so there is no lever here worth
  predicting.

## Outcome of A1

`receipts-faster-20260725/kv_split_bench.json`. 94L × 4kv × 128d, ctx 32768,
A2000, median of 8 after warmup.

| prediction | predicted | measured | verdict |
|---|---|---|---|
| A1c byte-identical at every f | exact | exact at f ∈ {0, 32, 64, 128, 240}/240 | **CONFIRMED (gate)** |
| A1a fitted B | [5.0, 7.5] GB/s | **8.21** | **FALSIFIED** |
| A1b fitted c | < 25 ms | **88.0 ms** | **FALSIFIED** |
| A1d peak(f=.5)/peak(f=0) | [0.4, 0.65] | **2.566** | **FALSIFIED** |

**The feature itself works exactly as designed.** Resident bytes track the dial
to four decimal places (f·1.774 GB: 0.4435 / 0.8871 / 1.3306 measured against
0.4435 / 0.8871 / 1.3306 expected), streamed bytes are the exact complement, and
load falls 1254 → 1098 ms at f=0.75. The three falsifications are two defects in
my predictions and one real cost I did not anticipate.

**A1b earned its keep — it caught the thing it was written to catch.** Its
rationale was "splitting introduces a fixed cost that eats the saving at small
f", and there is one: `_materialize` assembles the layer with `torch.cat`, which
allocates a fresh full-size bf16 tensor and copies both halves into it. At 32K
that is 94 layers × 2 tensors × 33.5 MB ≈ 6.3 GB of device copy per pass. **At
f=0 there is no cat at all** — a single part returns directly — so f=0 is a
different regime, not a point on the same line.

**Which is also why A1a failed, and the restricted fit shows it:**

| fit | B | c |
|---|---:|---:|
| all four points | 8.21 GB/s | 88.0 ms |
| **split points only (f > 0)** | **6.29 GB/s** | **60.4 ms** |

At 6.29 GB/s the law holds and sits inside A1a's interval. The 8.21 is an
artifact of fitting a line through two regimes, one of which skips the
concatenation — a mistake in the analysis plan, made before the run and
therefore scored as registered. A1a is falsified; the *law* is not.

**A1d was simply backwards.** Split trades VRAM **for** bandwidth, so resident
VRAM must RISE with f — measured 2.566, which is the correct and desired
direction. I wrote the interval as though f were the streamed fraction. No
measurement is at fault.

**Pre-committed decision fires anyway.** A1a and A1c were the conditions, and
A1c held while A1a's failure is an artifact of the fit rather than of the
mechanism — so, stated plainly: **split residency becomes the default shape and
the binary switch is retired**, on the strength of A1c plus the restricted fit,
with A1a recorded as falsified as scored.

**What this hands to the next cycle:** ~60 ms/step is being given away to the
concatenation. Dequantizing head and tail directly into one preallocated
`[1, H, T, D]` output removes the extra allocation and one full copy. At f=0.75
that would take the overhead from 131.5 ms toward the 71.6 ms the byte count
predicts — i.e. **the assembly currently costs almost as much as the transfer it
saves.** Any re-run after that fix is a follow-up, not a rescoring.

### A1 follow-up after the assembly fix — NOT a rescoring

Dequantizing both halves and letting one `cat` write the contiguous result,
instead of materializing each half first:

| f | overhead before | after | byte floor |
|---|---:|---:|---:|
| 0.00 | 287.6 ms | 288.4 ms | 286.2 |
| 0.25 | 272.6 | 257.7 | 214.7 |
| 0.50 | 200.6 | 185.3 | 143.1 |
| 0.75 | 131.5 | **105.6** | 71.6 |
| fitted c | 88.0 ms | **54.0 ms** | — |
| fitted B | 8.21 GB/s | **7.14 GB/s** (now inside A1a's interval) | — |

f=0 is unchanged, as it must be — there is no concatenation to fix there. The
registered verdicts stand; this is what the fixed code does. **A residual 54 ms
remains and is structural**: the assembly still writes one full-size contiguous
tensor per layer, and removing that needs attention to accept two tensors rather
than one, which is an attention-path change and not a cache change.

## Outcome of B1

`receipts-faster-20260725/kv_prefetch_bench.json`, same geometry.

| prediction | predicted | measured | verdict |
|---|---|---|---|
| B1a prefetched / resident @32K | [1.00, 1.15] | **1.012** | **CONFIRMED** |
| B1b plain / prefetched @32K | [1.20, 1.35] | **1.282** | **CONFIRMED** |
| B1c peak delta | < 100 MB | **+18.0 MB** | **CONFIRMED** |

| ctx | resident | streamed | streamed+prefetch | transfer hidden |
|---:|---:|---:|---:|---:|
| 8192 | 245.4 ms | 318.8 | **248.5** | **95.8%** |
| 32768 | 967.9 ms | 1256.3 | **979.8** | **95.9%** |

**Exposed transfer at 32K goes from 288.4 ms to 11.9 ms.** The streamed tier now
costs **1.2%** over holding the whole cache resident, while keeping **zero**
bytes of it on the device — 443 MB peak against 2108 MB, 4.76× less. The
fraction hidden is the same at both contexts, which is what a hiding mechanism
should look like rather than a lucky ratio.

**This changes how #14's window table should be read, and the correction goes
the favourable way.** That analysis treated transfer as strictly additive:
`step = W + batch·KV(ctx)`, all of it exposed. With overlap the step is
`max(compute, transfer)`, so **`link/KV(ctx)` is the transfer-bound, not the
bound** — it only binds when `KV/link` exceeds the per-step compute. Here it does
not, by 3.4×, so the link is invisible. The windows in #14 are therefore
conservative. They are not re-derived: doing so needs a per-model compute
estimate this project does not have, and inventing one to widen a table in my
own favour is exactly the move this document exists to prevent.

**The registered interaction now matters.** B1 hides the transfer behind the
**dequantization** — work that exists *because* the data arrives packed. D1
(fused attend) removes that dequant. So the two are **substitutes**: a future D1
result must not be added to B1's, and if D1 lands, B1's 95.9% will fall because
there is less left to hide behind. Stated in the prereg before either ran, and
restated here because it is now load-bearing rather than hypothetical.

**Confound, restated:** this harness's compute is dequantization, not attention.
A real decode does attention and an MLP on top, so there is *more* to hide
behind, not less — but that is an argument, not a measurement, and the real-model
number has not been taken.

## Amendment 1 — B1 on a real model (E1), and C1 closed

Written after B1's synthetic result, **before E1 ran**.

### C1 — closed, not deferred

C1 predicted an fp16 absmax cuts streamed bytes by exactly 5.56%. That is still
true and now **worthless**: B1 hides 95.9% of the transfer, so 5.56% of the ~4%
still exposed is **~0.2% of a step**, bought by changing a packed layout that
every kernel in `kernel/` reads. The gate was "blast radius far larger than its
5.6%"; prefetch shrank the 5.6% by a further 24×. Closed on arithmetic, and
recorded rather than left dangling — a deferred item nobody intends to run is
just an unmarked negative.

### E1 — does the 95.9% survive a real model?

B1's headline carries one stated confound: the harness's "compute" is
**dequantization**, not attention. The argument that a real decode has *more* to
hide behind is plausible and untested, and #13 is a standing reminder that
plausible mechanisms get falsified here.

**Fixture.** OLMoE-1B-7B (locally cached), weights **resident, not offloaded** —
which is deliberate: streamed weights would contend for the same link and make
the KV term unattributable, and "weights fit, context does not" is precisely the
case #14 says the streamed tier serves. Prefetch is driven by a forward pre-hook
on each decoder layer requesting layer *i+1*, the same shape `offload.py` uses,
rather than by prefetching everything up front (which would put the whole cache
on the device and defeat the point).

- **E1a.** Prefetched-streamed decode / resident decode ∈ **[1.00, 1.20]**.
  Wider than B1a's [1.00, 1.15] because a real step has more moving parts.
  *Falsified above 1.35.*
- **E1b.** The hidden fraction is at least as large as the synthetic
  **95.9%** — i.e. **≥ 90%** hidden, since attention and the MLP add compute to
  hide behind. *Falsified below 80%*, which would mean the synthetic result does
  not transfer and #16's headline needs the qualifier moved into it.
- **E1c.** Greedy decode is **identical** between resident and prefetched-
  streamed caches — same token ids, not just similar perplexity. Residence and
  scheduling must not change values. *Any mismatch voids E1a/E1b.*
- **E1d — the number that decides D1.** Report dequantization as a fraction of
  the resident-cache step. No interval is registered because I have no basis for
  one; it is recorded as a measurement, and it is what says whether D1 removes a
  dominant cost or a rounding error.

**Pre-committed decision.** If E1a and E1c hold, `enable_kv_prefetch(model,
cache)` ships as a supported helper rather than staying a benchmark trick. If
E1b fails, #16's 95.9% gets restated as synthetic-only in the finding itself,
not in a footnote.

## Outcome of E1 — B1 does not transfer to a real model

OLMoE-1B-7B, 4-bit weights **resident**, 4096-token prompt, 24 greedy tokens.
Three runs, because two defects were found and fixed between them. **All three
are reported**; the last is the scored one.

| run | E1a (≤1.35) | E1b (≥90%) | E1c (identical) |
|---|---:|---:|---|
| 1 — as built | 1.165 | 41.7% | True |
| 2 — async append | **1.049** | **68.4%** | **False** |
| 3 — correct prefetch | **1.183 CONFIRMED** | **−22.5% FALSIFIED** | **True CONFIRMED** |

**Run 2 was the fastest and it was wrong**, which is the entire argument for
E1c being a gate rather than a nice-to-have. It also means **run 1's `True` was
a false pass**: the same staleness was present, and the divergence is marginal
enough that it first appeared at token 12 of 25. A correctness gate that passes
by luck is why one configuration is not a test.

### Two defects, both found only because E1 ran on a real model

**1. The append synchronized the host, 32 times per step.** `copy_` into a CPU
destination blocks until the DMA lands, and the arena append does one per layer
per tensor. The streamed arm ran at **1.87 GB/s on a 6.20 GB/s link** — the
byte model said 24.4 ms/step and the measurement said 80.84. Exactly the failure
`host_gather.py` records for the expert path (B3, ~94 syncs/token), reappearing
in the KV path. Fixed with `non_blocking=True`, which then required the prefetch
stream to wait on the default stream, since the append is now asynchronous.

**2. Prefetch staged history and used it as the whole layer.** Decode
interleaves `prefetch(i) → update(i) appends → update loads`, so a pre-hook
stages the arena *before* this step's token exists. Using it as-is silently
drops the newest token from attention — no crash, just wrong output. **The unit
suite was green at 25/25 throughout**, because its prefetch test completed every
update before any prefetch and so could never produce the interleaving a decode
actually uses. Fixed by treating a staged slot as history and concatenating
whatever arrived after it; a decode-order regression test now exists.

### Why B1's 95.9% did not survive

**The synthetic harness timed loads only — it never called `update()`.** So it
never paid the append, never paid the assembly, and never exercised the
interleaving that made the design incorrect. Its 95.9% is a true statement about
that harness and **not** a statement about decode.

With prefetch correct, exposed transfer goes **46.38 → 56.83 ms/step**: worse.
The safety wait (`stream.wait_stream`) makes the side stream queue behind
everything already on the default stream, so there is little overlap left to
win, and the history+tail concatenation adds a copy per layer.

### Decisions

- **The E1b decision fires: #16's 95.9% is restated as synthetic-only in the
  finding itself**, not moved to a footnote.
- **The E1a/E1c decision is NOT taken**, and the specification was wrong.
  It said `enable_kv_prefetch` ships if E1a and E1c hold — both do — but
  shipping a helper that measures **−3%** would be promoting a pessimization.
  The condition should have included E1b, which is the prediction that says
  whether the feature works at all. Recorded as a specification error (the third
  this cycle: A1a's two-regime fit, A1d's inverted sign, and now this) rather
  than quietly not doing it. `prefetch()` stays in the library — it is correct,
  optional, and off unless called — but no helper encourages its use.
- **E1a is a real result worth keeping**: with weights resident and the cache
  fully streamed, a real decode costs **+18.3%** and holds **zero** KV bytes on
  the device. That is the honest headline for the tier, and it is 4.8× less VRAM.

**Next, and specified before trying it:** the safety wait is a whole-stream
barrier where a per-layer event would do — the side stream only needs to wait on
*that layer's* append, not on everything queued. That is the one change that
could recover overlap, and it gets its own prediction rather than being folded
into B1's.

## Amendment 2 — E2: per-layer events instead of a whole-stream barrier

Written after E1, **before E2 was built**.

**The mechanism, stated so it can be wrong.** Making the append asynchronous
forced the prefetch stream to wait on the default stream, and
`wait_stream(current_stream())` is a **whole-stream barrier**: it waits on
everything queued, including the compute of the layer we are trying to overlap
with. That is why E1's correct prefetch lost 3% instead of winning.

A per-layer event is the narrow version. `prefetch(i+1)` fires from layer *i*'s
pre-hook, and the append it must not race is layer *i+1*'s — **from the previous
step**, which completed long ago. So the wait should be free and the copy should
start immediately, overlapping layer *i*'s compute. If that reasoning is right,
the barrier was the whole problem.

**Track record disclosure.** My prefetch predictions have now been wrong twice —
B1 confirmed synthetically and then failed on a real model, and E1a/E1c were
specified without the prediction that mattered. The intervals below are
deliberately wide and the mechanism is the claim, not the magnitude.

Baseline (E1 run 3): resident **311.07**, streamed **357.45** (exposed 46.38),
streamed+barrier-prefetch **367.91** (exposed 56.83) ms/step. Transfer by bytes
is 24.4 ms/step.

- **E2a.** Exposed transfer with per-layer events ≤ **20 ms/step**.
  *Falsified above 40 ms* — 40 is roughly "no better than no prefetch at all".
- **E2b.** Hidden fraction ≥ **60%**. *Falsified below 40%.*
- **E2c — gate.** Greedy token ids identical to the resident cache. Both defects
  E1 found were correctness defects and one of them passed this check by luck,
  so it runs over more tokens. *Any mismatch voids E2a/E2b.*
- **E2d.** No regression: step time ≤ the no-prefetch streamed arm (357.45 ms).
  *Falsified above it* — a prefetch that is still net-negative is not shipped
  whatever its hidden fraction says.

**Pre-committed decision** — specified to include the prediction that says
whether it works, which is the error E1 made: if **E2a, E2c and E2d** all hold,
prefetch becomes a supported, documented path. If E2d fails, prefetch is closed
for this project and the streamed tier ships at its measured +18.3% with the
mechanism recorded as tried and rejected.

## Outcome of E2 — the barrier was not the problem

Same fixture, 32 greedy tokens. `receipts-faster-20260725/kv_prefetch_real_e2.json`.

| prediction | predicted | measured | verdict |
|---|---|---|---|
| E2a exposed transfer | ≤ 20 ms/step | **91.42 ms** | **FALSIFIED** |
| E2b hidden fraction | ≥ 60% | **−38.9%** | **FALSIFIED** |
| E2c greedy ids identical | exact | True | **CONFIRMED** |
| E2d no regression | ≤ 327.49 ms | **353.10 ms** | **FALSIFIED** |

| arm | ms/step |
|---|---:|
| resident | 261.68 |
| streamed | 327.49 |
| streamed + per-layer-event prefetch | 353.10 |

The reasoning in the amendment was sound and the conclusion drawn from it was
wrong. Narrowing the barrier to a per-layer event did remove a real
serialization — and prefetch still lost, by *more* than the whole-stream version
did. **The barrier was not the problem.**

**Why, and it generalizes.** The transfer is 24.4 ms of a ~262 ms step: **9%**.
The machinery to hide it — an extra device allocation per layer per tensor, a
staged-history-plus-tail concatenation the non-prefetch path never pays,
cross-stream events and `record_stream` bookkeeping — costs more than the 9% it
is chasing. **At 9% of a step, the transfer is not worth machinery.** Prefetch
would pay where transfer is a large *fraction* of the step: long context with
cheap compute. This model is the opposite, and so is most of what #14's window
table covers.

**Pre-committed decision fires. Prefetch is CLOSED for this project.** Three
registered attempts — B1 (synthetic, confirmed), E1 (real, falsified), E2
(narrowed, falsified) — is enough. The streamed tier ships at its measured
**+18.3%** for zero KV bytes on the device, and the mechanism is recorded as
tried and rejected rather than left as a promising-sounding TODO.

`prefetch()` stays in the library: it is correct, tested in decode order, off
unless called, and someone in the long-context/cheap-compute regime may want it.
Nothing promotes it and no helper wraps it.

**What this says about the whole line.** The streamed tier's cost is the
transfer, and the transfer obeys `bytes / link` (#15, confirmed at 1.009 and
1.001). Scheduling cannot argue with that, and three attempts to hide it behind
compute did not. The levers that remain are the ones that move *bytes*: NF4
(shipped, 3.56×) and split residency (shipped, exact). That is the honest end of
this line.

## Scoring

Results land in `receipts-faster-20260725/`. Every prediction is marked
**confirmed / falsified / void** with the measured value beside the predicted
interval. Falsified entries stay in this document. A1c is a **gate**: if the
split cache is not byte-exact, A1a/A1b are void rather than scored, because a
wrong cache can be arbitrarily fast.
