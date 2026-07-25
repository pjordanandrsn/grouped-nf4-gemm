# PREREG — the flagship, with context: 235B at 32K on a consumer-sized working set

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `3496676`, e4b @ `1ad6604`.
Both local, unpushed.

## The claim under test

The flagship is **Qwen3-235B-A22B decoding on ≤16 GB of VRAM** — weights
NF4-packed in pinned host RAM, streamed per token over PCIe — measured at
**4.3–4.4 tok/s at seq-512**, and re-labelled today as *"at ~5K context"*
because KV was never in the figure.

Findings #13–#18 made KV a controllable term, and the composition scout measured
weights and KV **adding** on one link (additivity 0.962 at 32K, weights
dominating 6×). This run asks whether the composed configuration is real at
flagship scale:

> **235B, 32K of context, KV entirely off-device, working set ≤ 16 GB.**

## What this run can and cannot do — stated before it runs

**The 2×2 the scout used is impossible here.** Its `neither` and `KV-only` arms
need weights *resident*, and 235B in NF4 is ~123 GiB — larger than any single
GPU on offer. **Additivity therefore cannot be decomposed at 235B**, and this
run does not attempt it. That is precisely why the scout existed, and it is the
reason its result carries the mechanism while this one carries the claim.

Feasible arms, all weights-streamed:

| arm | ctx | KV |
|---|---:|---|
| `short` | 512 | streamed | reproduces the flagship's measured point
| `long-resident-KV` | 32768 | resident |
| `long` | 32768 | **streamed** | the claim

## Predictions

Grounded in the scout's measurements, not in a mechanism — which is the one
pattern that has held all day.

- **N1a — the headline.** `long` arm peak VRAM ≤ **16.0 GB**. *Falsified above
  18.0.* The stamped flagship working set is 15.2 GB at seq-512; streaming the
  KV should hold that roughly flat while resident NF4 KV at 32K would add
  1.65 GB and break it.
- **N1b — the cost of context.** `long` tok/s / `short` tok/s ≥ **0.85**, i.e.
  64× the context for ≤15%. The scout measured **11%** on a model with a
  *worse* KV ratio (GQA 8:1 against 235B's 16:1). *Falsified below 0.75.*
- **N1c — gate.** `long` reports **0 bytes** of KV on device, and greedy decode
  is deterministic across repeats. *Either failing voids N1a/N1b* — a
  configuration that quietly keeps KV resident would satisfy N1b for the wrong
  reason.
- **N1d.** KV's share of the composed step ∈ **[6%, 25%]**, bracketing the
  scout's 11% and the byte-model's ~12%. *Falsified outside.*

## Cost, risk, and the stop rule

2×A100-80GB (**for host RAM, not GPUs** — ~123 GiB of pinned NF4 experts needs
headroom a 1× slice may not have), ~$2.78/hr, 438 GB checkpoint download, hard
cap **2 hours** on a delete-only backstop. Expected **$4–6**, an order of
magnitude above anything else today.

**Staged stop rule, to avoid paying for a run that cannot finish:** host RAM is
checked **before** the download starts, and the pod is terminated immediately if
it cannot hold the expert stack. That costs ~$0.15 to learn instead of $5.

## Pre-committed decisions

- **N1a and N1c hold** → the flagship heading changes from *"≤16 GB (at ~5K
  context)"* to a figure that carries 32K, and the README says so with the
  tok/s cost beside it.
- **N1a falsified** → the composed configuration is a 24 GB story; the heading
  keeps its ~5K qualifier and the 32K claim moves to a larger card.
- **N1b falsified** → context is expensive at flagship scale in a way the scout
  did not predict, and the scout's additivity result gets a scale caveat it
  currently does not have.
