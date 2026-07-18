# PREREG — gpt-oss-120b A/B/C serving on one A2000 + Xeon (EXPLORATORY)

**Stamped before the throughput data (see `.ots`).** At stamp time: the A-arm
is mid-load (1.6 GB of a ~65 GB requant store written, **zero tokens timed**);
B0/B/C have **not** run (llama.cpp CUDA build still compiling). The skew and
PCIe numbers below are already-measured **inputs**, not predictions, and are
labeled as such. This is a single-box exploratory benchmark — EXPLORATORY
tier, r=1 per bench cell, one prompt for the A-arm; it is not a confirmatory
and does not enter any law.

**Box.** QNAP TVS-h1688X, Xeon W-1250 (6C/12T), 128 GB DDR4, NVIDIA RTX A2000
12 GB (sm_86, ~8.9 GB free beside co-tenant services). Model:
`openai/gpt-oss-120b` — MXFP4 GGUF for B/C, requantized to bnb-NF4 for A.
Everything thread-capped `-t 6` to protect HA/Plex/arr.

## Frozen inputs (measured, cited)

- **PCIe H2D on this slot** (measured tonight): 5.83 GB/s pinned, 5.57 pageable.
- **Expert-usage skew** (from the banked 512-tok capture, 221,184 records,
  `router_probe` topk_set): **global cv 0.18** (near-uniform; OLMoE 0.51,
  Qwen3-30B 1.41), but **per-layer top-16 experts capture mean 41.7% of hits**
  (min 33%, max 62%; all 36 layers ≥25%) — layer-specialization. This is an
  INPUT to the fusion decision rule below, already banked, not re-tested here.
- **DDR4 streaming bandwidth: spec-estimated ~40 GB/s** (Xeon W-1250
  dual-channel DDR4), **NOT cleanly measured** — a naive single-thread torch
  microbench read 3.7 GB/s, which is unrepresentative of llama.cpp's
  6-thread AVX streaming and is explicitly not used as the ceiling. The B/C
  predictions carry this as an estimated input.
- **Active params gpt-oss-120b ≈ 5.1 B/token** (E=128, k=4).

## Arms (frozen; the exact things that run)

- **A — selective NF4 port**: `gptoss120b_selective.py`
  sha256 `f2430970f08d397622f215b2f2dc2d06592c3e20d5b2efa80b7a12392f241ee7`.
  e4b 0.5.0 offload load (NF4 store, mmap on the M.2 pool), then each expert
  module's forward replaced with a k-selective gather (only the routed
  experts' packed bytes → pinned staging → `gemm_4bit_grouped`, fused NF4
  decode + fp32 accum) plus gpt-oss's exact clamped-GLU/biases. Phase A of
  the same script also times the **stock layer-streaming baseline** (all 128
  experts/layer over PCIe) as an internal floor. Metric: median s/token over
  16 greedy decode tokens (Phase B) and 3 (Phase A), one prompt.
- **B0 — all-CPU llama.cpp** (prebuilt, `-ngl 0`): floor including CPU attention.
- **B — GPU-attention + all-experts-CPU**: CUDA `llama-bench -ngl 99
  --n-cpu-moe 36`. Experts computed on the Xeon from DDR4; attention on GPU.
- **C — layer-residency hybrid**: `--n-cpu-moe {34,32,30}` — 2/4/6 full
  expert-layers held resident on the A2000, rest CPU, until VRAM caps out.
  B0/B/C via `bc_bench.sh`
  sha256 `d5e7d15840aeababdac935b60e481526006be62f7d984d169d560c29efc58e4c`.
  Metric: `llama-bench` tg (token-generation) tok/s, `-n 32 -p 8 -r 1`.

All arms: same box, same model, same `-t 6`, run sequentially (A frees the
card before B/C). Not bit-comparable (NF4 vs MXFP4 storage); the comparison is
**serving throughput**, and each arm's coherence is sanity-checked by eye.

## Predictions (directional, before data)

Primary hypothesis **H1: B > A** — the experts sit in DDR4, ~7× wider than
this PCIe link, so computing them in place beats streaming them to the GPU.

Magnitude bands (falsify a band if the measured tok/s lands outside it):

| arm | predicted tok/s | falsifier |
|---|---|---|
| A (selective) | **0.8 – 2.5** | outside ⇒ our PCIe+fused model is wrong for this box |
| A (stock stream, internal floor) | 0.05 – 0.2 | — |
| B0 (all-CPU) | 3 – 10 | — |
| B (GPU-attn + CPU-experts) | **4 – 12** | — |
| C (residency) | **B to 1.3×B** | >1.4×B ⇒ residency exploits more than the layer-granular-null predicts |

Ordering prediction: **B ≈ C > B0**, and **{B,C} > A**.

## What each surprise would mean (pre-committed reading)

- **A > B** — PCIe+fused beats CPU/DDR4 compute ⇒ the Xeon is severely
  flop-bound on 6 threads; would flip the "keep experts on the card" call.
- **C > 1.4×B** — layer-granular residency helps more than expected ⇒ the
  per-layer skew is reachable even at layer granularity; revisit the null.
- **B0 ≈ B** — GPU attention offload buys nothing ⇒ time is in the experts,
  as assumed (confirms the expert path is the target).

## Fusion follow-on — decision rule, pre-registered NOW

The hot-expert fusion (expert-granular residency: pin each layer's hot-16 to
the A2000 via **our** kernel, cold-112 on the Xeon; llama.cpp `-ot` cannot do
sub-layer placement, so this needs the A-arm machinery, not B/C) will be built
**only if BOTH**:

1. **B or C wins decisively over A** (≥1.5×) — confirming CPU/DDR4 locality is
   the right primary, so adding an idle-GPU hot tier is the marginal move; AND
2. **Held-out subset-stability passes**: choose each layer's hot-16 on prompt
   set P1, measure hit-capture on a **disjoint** prompt set P2. Require
   **per-layer mean ≥ 30%** out-of-sample (in-sample was 41.7%). Below 30% ⇒
   the resident subset doesn't generalize ⇒ do not build.

If H1 is refuted (A wins), the fusion question is moot — the whole model
already belongs on the card, and the port is the answer. Both outcomes are
informative and pre-registered as of this stamp.

## Limits (stated, not hidden)

Single box, single GPU, r=1 bench cells, one prompt for A — EXPLORATORY, not
confirmatory. NF4↔MXFP4 storage differs between arms (throughput comparison,
not a fidelity claim). DDR4 bandwidth is spec-estimated. `--n-cpu-moe`
semantics (which layers land on GPU) are llama.cpp-version-specific (b10068);
the achieved VRAM residency is reported from the run, not assumed.

---

## Amendment 1 (2026-07-18) — joules/token column (forward-only, still pre-data)

**Stamped before any tok/s** (A-arm still mid-load, 0 tokens timed; B0/B/C not
started). The body above is unchanged; this adds the energy axis — the
program's signature metric.

**Instrument, stated honestly.** GPU-rail power is **measured** (nvidia-smi
`power.draw`, integrated over each arm's wall-clock generation window by an
external sampler so the frozen harnesses are untouched — `abc_power_sampler.py`
sha256 `a2270dfb7f1f3777018ac095956c4db92c201cd000372ceea2bac54393157c6e`).
CPU-package energy is **ESTIMATED**, not measured: this QNAP kernel exposes no
intel-rapl powercap (checked host + container) and no UPS/smart-plug wall meter
is available (checked NUT + HA). CPU joules = ∫ (busy-fraction × 80 W) dt, with
busy-fraction from `/proc/stat` and 80 W the Xeon W-1250 package limit — a
model, tier ESTIMATED. Reported as two columns: **GPU J/tok (measured)** and
**total J/tok (GPU measured + CPU estimated)**.

`J/token = (∫ power dt over the timed window) / tokens_generated` — aggregate
over each arm's generation window (16 tokens for A Phase B, 32 for B0/B/C),
not per-token, since `llama-bench` reports aggregate tg.

**Predictions (before data):**

| arm | GPU J/tok (measured) | total J/tok (GPU meas + CPU est) | falsifier (total) |
|---|---|---|---|
| A (selective) | 15 – 35 | **20 – 45** | outside ⇒ energy model wrong for this box |
| B0 (all-CPU) | ~2 (idle GPU) | 12 – 28 | — |
| B (GPU-attn + CPU-experts) | ~3 (brief attn) | **8 – 18** | — |
| C (residency) | 4 – 12 | **8 – 16** | >1.3×B ⇒ residency costs energy it shouldn't |

Energy ordering prediction: **B ≈ C < B0 < A** — same order as speed.

**The non-obvious, pre-committed nuance (H2):** A's total-energy gap to B is
**narrower than its speed gap.** A is bottlenecked on PCIe transfer (~346 ms/tok
of DMA) during which the GPU mostly *idles* at low power, so A draws less power
but for longer; B pegs the CPU at high power but briefly. Predicted: speed
margin B/A ≈ 3–6×, but energy margin B/A only ≈ 1.5–2.5×. If the measured
energy margin instead **matches** the speed margin (≈ the same multiple), H2 is
refuted — the streaming GPU is *not* idling as modeled (it's power-hungry while
waiting), and the fusion's premise (an idle GPU worth filling) weakens.

This H2 is exactly why the hot-expert fusion is interesting on energy grounds:
it puts the idle-during-transfer GPU to work on resident experts, so the
fusion's energy win (if built) should exceed its speed win. Registered here,
pre-data, as the energy rationale for the follow-on gated in the body.

---

## Correction 1 (2026-07-18) — UPS provenance (forward-only; stamped text above untouched)

Amendment 1 states "no UPS/smart-plug wall meter is available (checked NUT +
HA)." That is **imprecise and is corrected here**: a **PhxTec-A1000** UPS
(~1000 VA) *is* connected and configured in QNAP (`/etc/config/ups/ups.conf`
section `[qnapups]`, `UPS Type = PhxTec-A1000`). What Amendment 1 got right is
that its telemetry is **not reachable from the shell**: the NUT stack is not
live (no `usbhid-ups` driver process, empty `/var/state/ups/`, `upsd` not
listening on 3493), QNAP reads load% only through its proprietary web daemon,
`getsysinfo` exposes no UPS field, and the CP210x USB-serial device on this box
is **not** the UPS — it is the Thread border-router radio held by `otbr-agent`
(the FP300 Matter dongle), which was deliberately not touched.

So a real whole-box wall reading exists **in the QNAP GUI** but not on any
scriptable path. Operator decision (2026-07-18): **keep the TDP-model CPU
estimate** as registered in Amendment 1 — the joules column stays
"GPU measured + CPU estimated," and the total-J/tok numbers remain
ESTIMATED-tier. The GUI UPS load% is available for a future manual upgrade of
that column if a run is ever repeated with the value read off by hand; it is
not wired in for this exploratory pass. No prediction changes.
