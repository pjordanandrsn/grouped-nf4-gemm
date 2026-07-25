# PREREG — host-resident KV: does the streamed tier cost what #14 says it costs?

**Tier: CONFIRMATORY (a stated model, tested against measurement, on hardware
that already exists). Status: STAMPED before any arm ran.** Code under test:
e4b `claude/e4b-gemma-inflight-d41f93` @ `2057066` (`NF4KVCache(residence=
"host")`, 18/18 correctness), gnf4 `kernel/nf4-kv-cache` @ `335c255`. Both
local, unpushed.

## What is being tested, and why it can be tested here

Finding #14 derived C4's viability from a single model:

```
step_bytes = W + batch × KV(ctx)        step_time = step_bytes / link
```

Every number in its window table — the `link / KV(ctx)` ceiling, `batch* = W/KV`,
the 0/1/6/10 window counts — inherits that model. The model has never been
measured. It was built from a bandwidth probe and arithmetic.

**The A2000 is the right box for this, which reverses what I said when scoping
C4.** I claimed a gen3 ×8 link makes this unmeasurable here. That is true of
*absolute throughput* and false of *the law*: a slow link makes the transfer
term dominate everything else, so `t_host − t_gpu ≈ bytes / link` is easier to
falsify at 6.20 GB/s than at 26.74. A rented box would change the constant, not
the relationship. Recorded because the earlier framing was wrong and the
correction is what makes this run free.

**Fixture.** Synthetic — the KV geometry of Qwen3-235B (94 layers, 4 kv heads,
head_dim 128) with no model and no weights, on the same rationale as
`kv_verify.py`'s rung-one probe: KV transfer cost is a function of geometry, not
of weight values. This is what lets a 235B-shaped cache be measured on a 12 GB
card, and it is labelled a geometry fixture, not a model result.

**Arms.** `residence="gpu"` vs `residence="host"`, × {nf4, bf16}, at contexts
{4096, 8192, 32768}. Measurement is split deliberately:

- **load-only** — per step, load every layer. This is the transfer term alone
  and it is what S1–S3 predict. Both arms dequantize identically; the *only*
  difference is that the host arm crosses PCIe first.
- **load+append** — a full decode step. Separated because the two arms have
  structurally different append costs (S5), which would otherwise contaminate
  the transfer measurement.

## Predictions

Predicted transfer times use the **measured** 6.20 GB/s asymptote from
`receipts-c4-20260725/pcie_probe.json`, not a spec number. Packed KV is
576 B/layer/token (2 × 4 × 128 × 0.5625), so the whole cache is
576 × 94 × ctx bytes: **0.222 GB at 4K, 0.444 GB at 8K, 1.774 GB at 32K** →
predicted overheads **35.8 / 71.5 / 286.2 ms** per step.

- **S1 — the law holds.** At ctx 32768, nf4:
  `(t_host − t_gpu) / (packed_bytes / 6.20 GB/s)` ∈ **[0.85, 1.25]**.
  *Falsified outside.* 4K and 8K reported alongside; the small-slice regime runs
  at ~5.9 GB/s rather than 6.20 (probe), which biases those ratios ~5% high and
  is stated in advance rather than discovered afterwards.
- **S2 — the mechanism is bytes, not calls.** bf16 overhead / nf4 overhead at
  ctx 8192 ∈ **[3.0, 4.1]** (exact byte ratio 32/9 = 3.5556).
  *Falsified outside* — outside means something per-call dominates, and the
  per-token compression story does not transfer to the streamed tier.
- **S3 — linear in context.** nf4 overhead at 32768 / at 8192 ∈ **[3.4, 4.6]**
  (exact context ratio 4.0). *Falsified outside.*
- **S4 — the capacity claim, which is the entire point.** Peak GPU allocated,
  streamed / resident, at ctx 32768 nf4: **< 0.10** (predicted ~0.05).
  *Falsified at ≥ 0.10.*
- **S5 — the resident append is O(T) and the arena append is O(1).** Noticed
  while writing the harness, registered before running it: the resident path
  appends via `torch.cat`, which reallocates and re-copies the whole packed
  store per layer per step, while the arena writes one token at an offset.
  Predict resident append time at 32K / at 4K ≥ **4.0** (an 8× context step),
  and host append ∈ **[0.5, 2.0]** (flat). *Falsified outside either.*

## Pre-committed decisions

- **S1 holds** → #14's model is validated on measured hardware; its window table
  stands, and a faster link changes only the constant. No rented box is needed
  to trust the shape of that table.
- **S1 fails high** → the model is missing a term, #14's ceilings are optimistic,
  and they get re-derived before any deployment claim is made from them.
- **S1 fails low** → suspect the measurement before celebrating: something is
  not actually crossing the bus.
- **S4 fails** → host residence does not deliver the capacity it exists for and
  is not worth shipping, whatever the transfer cost turns out to be.
- **S5 holds** → the resident path carries an O(T)-per-step append cost that is
  a bug in its own right, independent of streaming, and gets fixed separately.

## Known confounds, stated in advance

1. **No prefetch.** Copies and dequant serialize on the default stream, which is
   exactly what makes an *additive* model the right one to test. A prefetched
   implementation would break additivity by design; this experiment says nothing
   about it, and a later prefetch result must not be scored against S1.
2. **Shared device.** The A2000 also serves the home lab's voice-tts (~3 GB);
   free VRAM is recorded per run and timings are the median of ≥ 10 after
   warmup.
3. **Pinned views are asserted, not assumed.** The arena hands out `arena[:used]`
   prefix views; if one were not pinned, the copy would silently stage through a
   bounce buffer and inflate S1 for a reason that has nothing to do with the
   law. The harness asserts `is_pinned()` as a precondition and aborts.
4. **Arena allocation is outside the timed region.** `pin_memory()` page-locks
   1.77 GB at 32K; timing it would measure the setup, not the steady state.
5. **bf16 at 32K is not run** — a 6.3 GB resident cache plus transients does not
   leave safe headroom next to voice-tts on a 12 GB card. S2 is therefore scored
   at 8192, which is where it was registered, not chosen after seeing 32K fail.

## Outcome

Scored on the **registered run** (`receipts-stream-20260725/kv_stream_bench.json`).
A2000, 94L × 4kv × 128d, median of 10 after warmup.

| prediction | predicted | measured | verdict |
|---|---|---|---|
| S1 overhead / (bytes/link) @32K | [0.85, 1.25] | **1.009** | **CONFIRMED** |
| S2 bf16/nf4 overhead @8K | [3.0, 4.1] | **59.05** | **FALSIFIED** |
| S3 overhead 32K/8K | [3.4, 4.6] | **3.961** | **CONFIRMED** |
| S4 peak GPU streamed/resident @32K | < 0.10 | **0.2031** | **FALSIFIED** |
| S5 append 32K/4K resident ≥ 4.0 | ≥ 4.0 | **0.63** | **FALSIFIED** |

**S1 is the one that mattered and it is confirmed to 1%.** Measured overhead
288.6 ms against 286.2 ms predicted from the packed byte count and the measured
6.20 GB/s asymptote. It replicated at **1.001** on an independent later run.
**Finding #14's model is validated on hardware**: `step_time = bytes / link`
describes the streamed tier that exists, so the window table's shape stands and
a faster link changes only the constant. The pre-committed decision fires — no
rented box is needed to trust that table.

**S2 was falsified by a real bug, which is the most useful thing this run did.**
The raw (bf16) host path allocated its arena as `[1, H, cap, D]` and handed out
prefix views sliced on **dim 2** — not contiguous. `is_pinned()` still returns
**True** for such a view, so the harness precondition (confound 3) passed while
the DMA fell off a cliff: **0.09 GB/s against 0.95** on the same device, a 17×
penalty on the full cache. Fixed by making the raw arena token-major
`[cap, H, D]`, matching the packed layout. The lesson is worth more than the
fix: **pinned is necessary but not sufficient — contiguous is the other half,
and nothing in the API says so.**

**S4 was falsified by my threshold, not by the feature.** Peak GPU allocated is
429 MB streamed against 2112 MB resident — a real **4.9×** reduction, and cache
residency itself is 0 (`device_bytes()`). The 0.10 threshold assumed peak would
be dominated by the cache; it is dominated by ~340 MB of prefill transients
present in *both* arms, which the ratio cannot cancel. The prediction was
measured after `build()`, so it scored peak-including-prefill rather than steady
state. Recorded as falsified rather than re-operationalized.

**S5 was falsified in the direction I did not consider.** The resident append
*is* O(T) — `torch.cat` reallocates the whole packed store per layer per step —
but at 1.77 GB against the A2000's ~200 GB/s of device bandwidth that is ~9 ms,
small enough that per-call overhead across 94 layers dominates and the ratio
runs *backwards* (0.63). The mechanism is real; the prediction that it would be
visible at these sizes was wrong.

### Post-fix follow-up — reported, NOT a rescoring

Re-run after the contiguity fix (`kv_stream_bench_postfix.json`). The registered
verdicts above stand; this is what the fixed code does.

| arm | predicted | run 1 | run 2 (post-fix) |
|---|---:|---:|---:|
| nf4 32768 | 286.2 ms | 288.6 (1.009) | 286.4 (**1.001**) |
| bf16 8192 | 254.4 ms | 4302.3 (16.9×) | 271.4 (**1.067**) |
| bf16 4096 | 127.2 ms | 4531.1 (35.6×) | 136.2 (**1.071**) |
| nf4 8192 | 71.5 ms | 72.9 (1.018) | 57.6 (0.806) |
| nf4 4096 | 35.8 ms | 37.6 (1.052) | 47.1 (1.316) |

Once the layout is fixed **every arm obeys the law**, bf16 included — which is
the S2 hypothesis (the mechanism is bytes) confirmed by a route S2's own ratio
form could not deliver.

**A harness defect this exposes, stated plainly.** `t_host − t_gpu` is a
difference of two separately-timed loops carrying ~±15 ms of noise. That is
negligible against a 286 ms overhead (S1: 1.009 and 1.001 across runs) and
crippling against a 36–72 ms one. **S3 confirmed at 3.961 in run 1 and would
falsify at 4.970 in run 2 from identical code**, purely because its denominator
is a small noisy difference; S2's ratio form has the same defect (4.709 post-fix
against an exact 3.556). S1's interval was safe by luck of magnitude, not by
design. Any future prediction built on this harness must be scored on a
difference large relative to that floor, or measure the transfer directly rather
than by subtraction.

## Scoring

Results in `receipts-stream-20260725/`. Each prediction is marked
**confirmed / falsified / void** with the measured value beside the predicted
interval. Falsified entries stay in this document and are not edited to match
what happened.
