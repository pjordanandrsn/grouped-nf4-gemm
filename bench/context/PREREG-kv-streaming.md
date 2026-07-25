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

## Scoring

Results land in `receipts-stream-20260725/`. Each prediction is marked
**confirmed / falsified / void** with the measured value beside the predicted
interval. Falsified entries stay in this document and are not edited to match
what happened.
