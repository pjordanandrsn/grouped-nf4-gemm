# PREREG — the two things the rented run left open

**Tier: CONFIRMATORY. Status: STAMPED before either was measured.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `9bcceb6`,
e4b @ `5594538`. Both local, unpushed.

## K1 — is the dequant kernel's ceiling real, or is it my stopwatch?

The A100 run left this unresolved: 113 GB/s on the A2000 (39% of ~288),
276.5 GB/s on the A100 (13.6% of ~2039). It captured ~35% of a 7.1× memory
improvement — neither bandwidth-bound nor obviously kernel-bound.

**A suspicion about the measurement, not the kernel.** Every timing today
brackets *one* call with `torch.cuda.synchronize()` on both sides. On the A100
the kernel takes **78 µs**; launch plus a round-trip sync is on the order of
10–20 µs, i.e. **13–26% of what was recorded**. On the A2000 the same kernel
takes ~180 µs, so the same absolute overhead is ~5–10%. That would inflate the
A2000's apparent efficiency relative to the A100's and manufacture exactly the
"scales sub-linearly" signature that was observed.

If true, it does not just explain R1a — it means **H2a and H3a were scored
against a stopwatch that included launch overhead**, and the sweeps that chose
`rows`/`warps` were partly ranking launch cost.

The clean test needs no rented box: hold the kernel fixed and grow the problem.
Fixed overhead is constant; real work is not.

- **K1a.** Fused dequant GB/s at **T=32768** ≥ **1.30×** its GB/s at **T=4096**,
  same H and D. *Falsified below 1.05* — that would mean the per-call cost is
  already negligible and the ceiling is genuinely the kernel.
- **K1b.** Launch-amortized timing (N back-to-back calls inside **one** sync
  window) is ≥ **5%** faster per call than the synced-per-call number at
  T=4096. *Falsified below 0%*, i.e. if amortizing changes nothing.
- **K1c — gate.** Whatever the timing says, still bit-identical to
  `dequant_kv_ref`. Unchanged code, so this is a regression check, not a
  discovery.

**Pre-committed decision.** If K1a or K1b confirms, the recorded bandwidths for
H2/H3/R1a are **measurement artifacts to the extent shown**, every one of them
gets an amortized figure recorded beside it, and the "unexplained headroom" is
restated by that amount rather than left implying a kernel defect.

## K2 — what actually predicts prefetch's crossover?

P1's transfer-share bracket was **withdrawn** after the A100 won at a 17.3%
share where the A2000 lost at 17.8%. A ratio-of-bandwidths story was floated and
explicitly *not* adopted, because it makes a prediction it was never asked to
survive: if prefetch's cost is `α·ctx/HBM` and its benefit is `β·ctx/PCIe`,
**both scale linearly in context**, so the ratio is context-independent and
prefetch should win at every context or none. It does neither. So that story is
already contradicted by data in hand.

The surviving shape is a **fixed** machinery cost plus a linear one:
`cost = c + m·ctx`, against a benefit that is purely linear. That predicts
losing at short context and winning past a crossover — which is what both cards
show — with the crossover landing at a different context per card.

- **K2a.** Fitting `prefetch_cost = c + m·ctx` over ctx ∈ {2048, 4096, 8192,
  16384} on the A2000 gives a **fixed term c > 5 ms**. *Falsified at c ≤ 0*,
  which would kill the fixed-cost explanation outright.
- **K2b.** The crossover implied by that fit falls **between 4096 and 16384**,
  matching where prefetch was observed to flip on this card.
  *Falsified if it lands outside.*

**Pre-committed decision.** If K2a confirms, prefetch is documented as "pays
above a per-device crossover context", with the fitted `c` and `m` recorded and
the crossover computed rather than asserted. If K2a falsifies, no model is
offered at all and the guidance stays purely empirical: two measured points per
card and no interpolation.

**Track record note, again.** Every mechanism argument tested today has been
falsified — the traffic model, the occupancy model, prefetch-hides-behind-
dequant, the barrier diagnosis, the codebook gather. K1 and K2 are both
mechanism arguments. They are written to be cheap to kill.

## Outcome of K1 — the ceiling was the stopwatch

| prediction | predicted | measured | verdict |
|---|---|---|---|
| K1c **gate** bit-identical | exact | exact | **CONFIRMED** |
| K1b amortized faster than synced @4096 | ≥ 5% | **45–51%** | **CONFIRMED** |
| K1a GB/s(32768)/GB/s(4096) amortized | ≥ 1.30 | **1.13–1.24** | **outside interval** |

A2000, same kernel, two stopwatches:

| T | synced GB/s | **amortized GB/s** | ratio |
|---:|---:|---:|---:|
| 4096 | 108.3 | **220.2** | 2.03× |
| 8192 | 150.8 | 226.9 | 1.51× |
| 16384 | 186.1 | **248.7** | 1.34× |
| 32768 | 200.0 | 248.1 | 1.24× |
| 65536 | 208.5 | 246.5 | 1.18× |

**The kernel runs at 248.7 GB/s against this card's ~288 — 86% of memory
bandwidth.** It was never slow. **Roughly half of every timing taken today was
`torch.cuda.synchronize()` round-trips**, and the fraction was largest at the
smallest problem, which is exactly the shape that manufactured a "sub-linear
scaling" signature.

**K1a is outside its interval for the reason that confirms K1b.** It predicted
throughput would climb with problem size once overhead was diluted — but
amortizing *already* removes the overhead, so amortized 4096 is already
saturated (220 GB/s) and has only 13% left to gain. The prediction was written
as though the two effects were independent. They are the same effect.

**Pre-committed decision fires, and it reaches backwards.**

- **#18's "unexplained headroom" is closed as explained.** 113 → ~287 GB/s was
  never a kernel deficit; the kernel is at 86% of peak and the gap was
  instrumentation.
- **H2a was falsified by a defective instrument.** It predicted ≥150 GB/s and
  scored 113 — under corrected timing the same kernel measures **220**, which
  would have **confirmed** it. The original scoring stays visible, but the
  prediction was right about the world and my stopwatch said otherwise. Same
  shape as #17: a faithful measurement of a defective input.
- **H3's verdict stands.** Tree 92.8 vs gather 113 were both synced, so the
  common overhead cancels in the *absolute* difference — the tree really is
  ~51 µs/call slower and the revert was correct. Only the *ratio* was
  compressed.
- **R1a is NOT resolved and is now known to be compromised.** The A100's 276.5
  was synced at T=4096, where overhead is the largest share and an A100 is
  furthest from saturation. Both biases run the same way, against the faster
  card. The honest position is that the cross-device scaling question remains
  open and its earlier answer was measured badly.

**And a defect in this run, recorded rather than smoothed.** The first pass
printed bandwidths 1000× too high — `synced()` returns seconds and carried a
`×1e3` meant for milliseconds. Caught before anything was written down, but it
is the second unit-or-instrument error in this file's neighbourhood today, and
both were in the measuring apparatus rather than the thing measured.

## Outcome of K2 — VOID. The instrument cannot resolve the effect.

| ctx | resident | streamed | +prefetch | prefetch cost | P1a |
|---:|---:|---:|---:|---:|---:|
| 2048 | 283.50 | 314.92 | **258.14** | **−56.8 ms** | 0.820 |
| 4096 | 182.64 | 222.41 | 264.08 | +41.7 ms | 1.187 |
| 8192 | 235.78 | 314.72 | 305.49 | −9.2 ms | 0.971 |
| 16384 | 295.33 | 382.57 | 383.76 | +1.2 ms | 1.003 |

**Not falsified — void, and the data says so on its face.** At ctx 2048 the
prefetched arm (258.14) is *faster than the fully resident cache* (283.50),
which is not physical: it does strictly more work on identical data. And the
resident baseline is **non-monotonic in context** — 283.50 at 2048 against
182.64 at 4096 — when it must rise with context. Both are the signature R1d
identified: this card carries ±12% and these arms ran at REPS=2, so the noise
is comfortably larger than the effect being fitted.

Fitting `c + m·ctx` to costs of −56.8, +41.7, −9.2, +1.2 ms would produce a
number, and it would be a number about the allocator and the thermal state of a
shared card. **No fit is reported.** K2a and K2b are unscored.

**Consistency check against the same day's data, which the noise also explains.**
P1 measured this card at 4096 as prefetch *losing* (1.096) and at 16384 as
prefetch *winning* (0.865). This run measures 1.187 and 1.003 — the 16384 sign
flipped. Two runs of the same code on the same card disagree about the direction
of the effect. That alone retires any A2000-based crossover model.

**Pre-committed decision fires in its second branch.** No model is offered.
Guidance stays purely empirical, and it now rests only on measurements taken
where variance was quantified: **the A100 run** (MAD 0.28%), where prefetch lost
at ctx 4096 (1.047) and won at 16384 (0.889, 64.1% hidden). Everything the
A2000 said about prefetch's crossover is withdrawn.

**What would resolve it:** the same four-context sweep on a quiet card. That is
~15 minutes and well under a dollar, and it is now the only open question in
this line that money can answer — which is a different situation from every
earlier refusal, where the question was answerable for free or not at all.
