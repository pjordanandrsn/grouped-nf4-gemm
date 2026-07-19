# PREREG — hot-residency hybrid vs llama.cpp, gpt-oss-120b, matched VRAM

**Tier: EXPLORATORY.** Stamp-ready; NOT stamped (operator's key). Written
**before any tok/s exists**: the first hybrid run crashed on a tokenizer bug
(`load_moe_4bit_streaming` returns `(model, config)` not `(model, tok)`) before
producing a number, and the llama arm has not yet run. Predictions below are
committed before the fixed re-run.

## Setup

Same DO H200 (143 GB VRAM, 235 GB RAM, EPYC host, ~45–55 GB/s link). Model:
gpt-oss-120b (E=128, k=4, L=36, ~59 GB NF4 / 60 GB MXFP4). Two arms at matched
~7 GB resident-expert VRAM:

- **Hybrid** (ours): K=16 hot experts/layer resident (fused NF4 kernel, zero
  transfer), cold tail **streamed** over PCIe per token, gpt-oss clamped-GLU.
  Original weights freed → constrained-card footprint. Unoptimized reference
  implementation (per-layer python gather + synchronous H2D).
- **llama.cpp** `--n-cpu-moe 32` (layer-granular; cold experts **compute on the
  EPYC CPU** from DDR4) + an all-resident anchor.

## Derived constants (banked measurements)

- per-expert NF4 = 13.1 MB; held-out per-layer top-16 hot capture ≈ 30%
  (`RESULTS-specstream-reanalysis`), so ~2.8 of k=4 routed experts/layer are
  cold → streamed.
- **cold bytes/token ≈ 1.29 GB** → transfer-only 23–29 ms (55→45 GB/s).
- serialization t_s (Addendum-1, on the 94-layer 235B pipeline) 53–86 ms; the
  36-layer gpt-oss pipeline should be lower but is **unmeasured** here.
- llama ncmoe interpolation from the H100 A/B/C (ncmoe24=36.8, ncmoe36=25.5) →
  ncmoe32 ≈ 29 tok/s; RESIDENT anchor 234 tok/s.

## Predictions (falsifiers stated)

- **P1 correctness** — hybrid next-token-logit `b_rel` < 3e-2 vs the reference
  gpt-oss forward (validated 0.009 on synthetic). *Falsify:* ≥ 3e-2 ⇒ the
  gpt-oss hot/cold path is wrong; the run is void, not a loss.
- **P2 direction (headline)** — **llama > hybrid.** I predict our own approach
  **loses** on this box. *Falsify:* hybrid ≥ llama ⇒ PCIe-stream + fused kernel
  beats CPU-compute even on a fast host — reopens pod-viability.
- **P3 hybrid absolute** — hybrid ∈ **[1, 8] tok/s** (overhead/serialization-
  bound: the transfer-only ceiling is ~17–35 tok/s, but the unoptimized
  per-layer python path + t_s dominate). *Falsify:* outside the band.
- **P4 llama ncmoe32** — **[24, 34] tok/s.** *Falsify:* outside.
- **P5 llama resident anchor** — **[210, 250] tok/s** (fits VRAM here).
- **P6 ratio** — llama/hybrid ∈ **[4×, 30×].**

## The nuance being registered

The **transfer-only** term (23–29 ms for the 70% cold experts) is *comparable*
to llama's whole per-token time (~34 ms at ncmoe32). So at the byte level the
approaches are close — the hybrid's predicted loss is **implementation overhead
+ serialization, not a fundamental disadvantage of streaming the cold tail.**
Pre-registered corollary: a *pipelined* hybrid (async cold prefetch, no
per-layer python) could close most of the gap on this box, and would **win** on
a weak-CPU box where llama's CPU-compute term balloons while the hybrid's PCIe
term is unchanged. **That crossover is NOT tested here** and is registered only
as the mechanism, not a claim.

## Standing prediction, plainly

We predict our hybrid loses to llama on this fast-CPU pod (P2), by a large but
not order-of-2-magnitude ratio (P6), with the loss attributable to unoptimized
serialization rather than the streaming strategy per se (the nuance above).
Either outcome is informative; both are pre-registered as of this commit.
