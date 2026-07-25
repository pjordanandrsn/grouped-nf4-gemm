# PREREG — context as a tier, in TRAINING: recompute vs offload on a busy link

**Tier: CONFIRMATORY. Status: STAMPED before the harness was written.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `2b80737`, e4b @ `1ad6604`.
Both local, unpushed.

## Translating the idea, precisely

Everything measured so far is **decode**, where the per-token VRAM term is the
**KV cache** and the finding is that it moves to host and streams back for ~11%.
Training has no KV cache. Its per-token VRAM term is the **activation stack**
held for backward, and there are two ways to buy it back:

| | spends | buys |
|---|---|---|
| **recompute** (gradient checkpointing) | GPU compute — one extra forward | activation VRAM |
| **offload** (`save_on_cpu(pin_memory=True)`) | PCIe bytes | activation VRAM |

Recompute is the standard answer and is in every training stack. Offload is the
*same idea* this project has been applying to weights and KV: **turn a VRAM term
into a bandwidth term.** Which is right is not obvious, and it is not obvious
*specifically because of what this project already measured.*

## The reason this is a real question and not a re-run

When weights stream, the link is the bottleneck and **the GPU is idle waiting on
it.** Recompute spends the thing that is idle. Offload spends the thing that is
saturated. So the scout's weight-dominance result predicts that **recompute gets
cheaper exactly where offload gets more expensive** — the two techniques should
diverge as soon as weight streaming is switched on.

That is a *mechanism story*, and mechanism stories are the class of prediction
this document set has falsified over and over (the traffic model, the occupancy
model, prefetch-hiding-behind-dequant, the barrier diagnosis, the codebook
gather). It is registered here **as a mechanism, labelled as one**, so that if it
falls it falls on the record with the others.

## Fixture

OLMoE-1B-7B, LoRA r=8, one optimizer step, seq 8192 and 32768, batch 1, median
of 3, on the A100 the flagship run is already paying for. Same model and loader
as the scout so the numbers compose.

**3×2 arms** — activation policy × weight residence:

| activation policy | weights resident | weights streamed |
|---|---|---|
| `none` (store everything) | ✓ | ✓ |
| `recompute` (checkpointing) | ✓ | ✓ |
| `offload` (`save_on_cpu` pinned) | ✓ | ✓ |

## Predictions

- **T1a — the textbook number, as a sanity anchor.** `recompute`/`none` step time
  with weights **resident** ∈ **[1.15, 1.50]**. Backward is ~2× forward, so one
  extra forward on three units is ~+33%. *Falsified outside [1.05, 1.80]* — and
  if it falls outside, the harness is presumed wrong before the textbook is.
- **T1b — offload buys the same VRAM.** `offload` peak ≤ **1.25×** `recompute`
  peak at 32768. *Falsified above 1.6.* Both should collapse the activation
  stack; if offload does not, it is not a substitute.
- **T1c — THE MECHANISM, flagged as such.** Switching weights from resident to
  streamed makes recompute *relatively cheaper* and offload *relatively dearer*:
  `offload/recompute` at 32768 is **higher with weights streamed than with
  weights resident**, by ≥ 0.15. *Falsified if the gap is < 0.05 or inverts.*
- **T1d — additivity, the scout's result carried into training.** With weights
  streamed, the offload arm's extra time over recompute is within **±35%** of
  `activation_bytes / measured_link_rate`. *Falsified outside ±60%.* The scout
  found weights and KV add on one link; this asks whether activations do too,
  when the link is carrying *two* weight passes instead of one.

## Pre-committed decisions

- **T1c confirmed** → the honest guidance is that **the tier idea does not
  transfer to training on a streaming box**: context in training should be bought
  with recompute, and the README says so rather than implying the decode result
  generalizes. A capacity technique that is right for decode and wrong for
  training is exactly the kind of thing this project should state plainly.
- **T1c falsified** → offload holds up under a busy link, which would be the
  stronger and more surprising result, and earns its own rung on the roadmap.
- **T1a outside [1.05, 1.80]** → nothing else in this prereg is scored; the
  harness is debugged first.

## Known confounds, stated in advance

1. **One model, one geometry, one device.** OLMoE is small; its activation stack
   at 32768 is a few GB, not the tens a 70B would hold. The *ratio* is what
   transfers, not the absolute.
2. `save_on_cpu` offloads **every** saved tensor, not just layer boundaries — a
   heavier policy than a hand-tuned offload would use. The offload arm is
   therefore a **lower bound on offload's quality**, and the finding must say so
   rather than claiming offload is beaten in general.
3. LoRA means the optimizer state is tiny and the backward is cheaper than a
   full fine-tune's. A full fine-tune would shift the fwd:bwd ratio T1a assumes.

## Outcome — the idea does not transfer, and the reason is structural

OLMoE-1B-7B, LoRA r=8 (58.7M trainable), batch 1, median of 2, A100-SXM-80GB.

**Weights resident** — the half that completed:

| seq | policy | s/step | peak VRAM |
|---:|---|---:|---:|
| 8192 | `none` | 2.357 | 39.76 GiB |
| 8192 | `recompute` | 3.799 | 10.33 GiB |
| 8192 | `offload` | 6.637 | 10.79 GiB |
| 32768 | `none` | **OOM** | — |
| 32768 | `recompute` | **7.439** | 28.15 GiB |
| 32768 | `offload` | **20.882** | 30.01 GiB |

**Weights streamed** — did not run, and *that is the finding*:

```
RuntimeError: backward re-dequantization read an offload-evicted expert
(0-element placeholder). Offloaded training requires gradient checkpointing
(use_reentrant=False) so the recompute re-stages the layer before its backward
runs — non-checkpointed offload training is unsupported
```

| prediction | predicted | measured | verdict |
|---|---|---|---|
| T1a recompute/none @8192 | [1.15, 1.50] | **1.612** | **outside interval** |
| T1b offload peak ≤ 1.25× recompute | ≤1.25 | **1.066** | **CONFIRMED** |
| T1c gap widens when weights stream | ≥0.15 | — | **VOID** |
| T1d activation bytes / link | ±35% | — | **VOID** |

### Offload never wins — not even where it was supposed to

T1c's whole premise was that offload would be *competitive on an idle link* and
lose once weight streaming saturated it. That premise is wrong at the root. With
weights **resident** — nothing else on PCIe, offload's most favourable case — it
is **2.81× slower than recompute at 32768** (20.882 s vs 7.439 s) and **saves no
memory** (30.01 vs 28.15 GiB, slightly *worse*). There was never a regime for the
busy link to take away.

### And with weights streamed, recompute is not better — it is mandatory

The guard above is e4b's own, and it is structural rather than incidental: a
streamed expert is **evicted after its forward**, so a backward that was not
re-staged by a checkpoint recompute has nothing to dequantize from. Checkpointing
is what re-stages it. So on a streaming box, `none` and `offload` are not slow
options — **they are not options.**

**The pre-committed decision fires, but not on T1c.** T1c as specified is VOID:
its arms never ran. The decision it gated — *the tier idea does not transfer to
training; context in training is bought with recompute* — fires on the two
results above, which are stronger than the timing comparison would have been.
Recording it this way rather than back-filling T1c as "confirmed", because a
prediction whose arms did not execute has not been tested.

### Two errors of mine, recorded

1. **The harness caught only `OutOfMemoryError`.** A `RuntimeError` therefore
   killed the process and took the streamed half with it. `none`-at-32768 OOMing
   was anticipated and handled; the guard that actually fired was not. That
   defect is what voided T1c and T1d, and it cost the run's second half.
2. **T1a's textbook anchor missed high** (1.612 against [1.15, 1.50]). Confound
   #3 named this in advance — LoRA changes the fwd:bwd ratio the +33% rule of
   thumb assumes — so the interval was set from a rule that this fixture was
   already known not to satisfy. Registering a confound and then predicting as
   though it did not apply is a specification error, not a discovery.

**Confound #2 still stands and limits the claim:** `save_on_cpu` offloads *every*
saved tensor, so these numbers are a **lower bound on offload's quality**. A
boundary-only offload would move far less. What is established is that the
*naive* tier translation loses badly and the streamed path forbids it outright —
not that no offload scheme could ever win.

**Open, unmeasured:** what recompute costs *when the link is busy* — the one arm
that both runs and matters on a streaming box. It was launched and lost to the
teardown clock.
