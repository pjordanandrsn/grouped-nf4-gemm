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
