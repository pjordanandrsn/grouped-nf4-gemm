# Cold Engine — Phase 0 premise test (QNAP measurements DONE 2026-07-19; roofline pipe term pending)

Status: desk items + the QNAP-box measurements below (receipts:
`bench/cold-engine/receipts-*-qnap.json`; tool = torch copy/triad, NOT
certified STREAM — named per the script header). Remaining Phase-0 items:
the QNAP pipe term (from the stamped QNAP slip) and the fat-host DDR column,
then the roofline table closes.

## 0.1/0.2 QNAP measurements (Xeon W-1250, 2ch DDR4-2666, 12 threads, AVX2-ONLY — no AVX-512 on Comet Lake)

| arm | result |
|---|---|
| memcpy ceiling (torch copy) | **~12 GB/s** plateau, saturates at 2 threads (1t: 8.8); triad peaks 15.3 |
| naive torch NF4 dequant floor | **0.067 GB/s packed** = 0.6% of ceiling (139 ms/expert on [5760,2880]); +GEMV adds ~4% (decode dominates utterly) |
| bnb 0.49.2 CPU dequantize_4bit | **0.041 GB/s** — SLOWER than naive: the AVX512 kernel cannot engage (no AVX-512 on this CPU); bnb silently falls back to its reference path |

**Verdict on the free floor: REFUTED on the target box.** The 0.3 scout's bnb
finding stands only for AVX-512 fleets; the QNAP cell gets no free lunch —
the **ggml-skeleton AVX2 port (Phase 2) is mandatory, not a fallback**, and
its grade is the fraction of ~12 GB/s it achieves. ggml's AVX2 Q4 paths (the
whole consumer llama.cpp base) are the existence proof at this ISA level.

**GO/NO-GO reading:** the premise is ALIVE but unpriced. The naive floor
being 0.6% of ceiling does not kill it (the directive anticipated writing the
kernel); what decides the QNAP cell is the ROOFLINE: the CPU tier pays iff a
bandwidth-bound AVX2 kernel (realistically 50–80% of ceiling ≈ 6–10 GB/s
packed) is comparable to or larger than the box's PCIe pipe term. Pull the
pipe number from the stamped QNAP slip before Phase-1 work is scheduled.

## 0.3 Scout — existing CPU 4-bit work (R6, with links)

- **bitsandbytes multi-backend CPU — VERIFIED in installed source (0.49.2,
  dev box)**: `bitsandbytes/backends/cpu/ops.py` registers
  `bitsandbytes::dequantize_4bit` for CPU on the **standard packed layout**
  (packed u8 + absmax + blocksize + quant_type), with an **AVX512
  implementation active at blocksize 64** (its fallback note applies only to
  blocksize ≥ 2048 fp16/fp32). No repack — arena bytes decode IN PLACE. The
  "AVX512 weight repacking" in the transformers docs is the Linear-layer
  gemv prepack path only; that path stays off-limits (second format), but
  the functional dequant is a free, law-compliant A-stock: bnb CPU dequant +
  torch matmul is the Phase-1 floor arm. Phase-2's job narrows to FUSING
  dequant+GEMV (bnb's fused CPU gemv on unpacked layout: verify at Phase-1
  entry).
  https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/bitsandbytes.md
  https://github.com/bitsandbytes-foundation/bitsandbytes/releases
- **ggml Q4-class microkernels** (MIT): the structural prior art for Phase 2
  (block layout, AVX2/AVX-512/VNNI paths, bandwidth-bound). Port the skeleton
  with credit (directive rail). llama.cpp's measured 45.34 tok/s ncmoe32@t24
  on the S3 box is the existence proof this class of kernel approaches DDR.
- **T-MAC** (table-lookup low-bit CPU kernels, MSR): relevant alternative
  shape (LUT-centric, avoids dequant multiply); heavier port, not the first
  choice — noted for Phase-2 if the ggml skeleton under-delivers.
  https://arxiv.org/pdf/2407.00088

## 0.4 Three-tier roofline (from stamped receipts; QNAP terms pending)

Inputs (receipts: `bench/homelab/RESULTS-pipelined-ladder.md`, stamped):
- Transfer law: `t ≈ 12.5 ms + cold_bytes / 45.2 GB/s` (S2 re-measure).
- Per-token cold bytes at K=0: panel median **1.61 GiB/tok** (E4; 10× route
  spread 0.13–1.28 GiB/tok at K=16).
- llama CPU-expert existence proof: thread ladder 3.98 / 7.65 / 13.93 / 24.25
  / 35.36 / **45.34** tok/s (E6, near-linear to 16t) and true fat-host best
  **ncmoe28 ≈ 49.8–51.6 tok/s @ ~13 GB** (S4 steelman).

Model: token time ≈ max(overhead, warm_bytes/pipe, cpu_bytes/DDR_eff) with
GPU-resident hot compute far under both. Optimal split puts
warm:cpu = pipe:DDR_eff, giving ceiling ≈ total_cold_bytes/(pipe + DDR_eff).

| box | pipe (meas.) | DDR terms | two-tier K=0 ceiling | three-tier K=0 ceiling | gain |
|---|---|---|---|---|---|
| S-box (H200 host) | 45.2 GB/s (ladder) | fat DDR5 (llama existence proof: 45 tok/s CPU-experts) | ~28 tok/s transfer-bound (measured 18.7–20.8 w/ overhead) | CPU tier adds little where pipe is this fat — NOT the target | ~1× |
| **QNAP (W-1250, AVX2-only)** | **5.83 GB/s pinned / 5.57 pageable** (stamped ABC slip, measured on-slot) | spec ~40; torch-copy ceiling **12** (measured 0.1); kernel-achievable est. 6–10 (50–80% of ceiling) | 1.61 GiB/tok ÷ 5.83 ≈ 296 ms ≈ **~3.4 tok/s** | 1.73 GB/tok ÷ (5.83 + 6…10) ≈ 109–146 ms ≈ **~6.8–9.2 tok/s** | **~2.0–2.7×** |

Reading: on the target cell the three-tier dial is worth ~2–2.7× at K=0 IFF
the Phase-2 AVX2 kernel lands at 50–80% of the 12 GB/s ceiling — the kernel
grade is the load-bearing unknown, and the naive/bnb floors (0.6%/0.3%)
prove nothing ships without it. On fat-pipe boxes (S-box) the tier is ~moot;
this engine is a thin-link instrument, exactly as the directive framed it.

## ADDENDUM-1 upgrades (2026-07-19)

**A1 theorem form — QNAP constants (measured, READ-SHAPED per A6):**
L = 5.83 GB/s pinned (stamped slip; mixed-bench solo 6.04 agrees).
Read ceiling (sum-reduce, 1× bytes — the kernel's shape) peaks **~20–23
GB/s** at 2–4 threads (copy's ~12 was a floor, as A6.1 said; noisy rows are
shared-box neighbors). C-projected = 6–12 (kernel grade band; decode ALU
likely binds before the read ceiling on 6 AVX2 cores — Phase 2 measures).
D_eff at the operating mix (read-shaped CPU + live DMA) ≈ **13–16.5 GB/s**
aggregate. **Candidate ceilings: L+C ≈ 11.8–17.8 vs D_eff ≈ 13–16.5 — they
CROSS in-band; whichever binds, the fused cold-tail ceiling is ~12–16 GB/s**
→ vs two-tier's 5.83: **~2.0–2.8×** at K=0. Phase 5's prediction form
(fraction of min(L+C, D_eff)) absorbs the ambiguity by construction.

**A2 mixed bench, read-shaped (receipts `receipts-mixed_qnap.json`):** DMA
holds 5.66–5.95 GB/s under 1–8 concurrent read-shaped CPU threads (≤ −6% vs
solo) — controller contention NEGLIGIBLE at this box's operating point (two
consumers ≲17 GB/s against a ~40 GB/s-spec 2ch controller). The predator
will bite on boxes where L+C approaches the controller limit; measure per
box, per the addendum.

**A3 registered:** Phase 4 delivers `plan_placement()` (load-time L/C/D_eff
probes → s* + tier assignment → printed self-grading receipt). Phase 5's
organizing prediction: achieved fused cold throughput as a FRACTION of
min(L+C, D_eff), band from the two-tier engine's achieved-fraction history
(house precedent: 90.7% of transfer floor at K=0), two-sided falsifiers on
the fraction. QNAP reframe stays mechanism-only; the standing bet slip
grades first under its original arms.

## 0.5 Graph-interleave — dependency analysis + recommendation

The hard constraint the four options must respect: layer L+1's input INCLUDES
layer L's expert contributions — a cold expert's output cannot be deferred a
layer without changing the model. True intra-token lead-ahead of COMPUTE is
therefore impossible without route+activation speculation; option (d) as
written buys weight/lookahead prep, not compute, unless it speculates.

- (a) two graphs + CPU join per layer: reintroduces per-layer host syncs —
  the B3 predator (94 syncs/token post-mortem). Dead for decode.
- (b) combine outside the graph: same per-layer sync, fragmented capture;
  loses the graphs-arm gains (ladder: graphs vs eager up to +150% at K=128).
- (c) pinned buffer + external-semaphore wait node: keeps one graph; needs
  driver-API graph surgery torch does not expose; fleet-fragility risk high.
- (d) lead-ahead: sound for PREFETCH, unsound for compute without the input
  activation; with route speculation it degrades to (c) plus a mispredict
  path (have-skip covers the fetch side only).

**Recommendation (per-box, matching where the engine is aimed):**
1. **Phase 1 lands EAGER with per-layer joins** (llama's own structure — its
   45 tok/s proves per-layer H2D/D2H joins of 5.7 KB vectors are affordable
   when compute dominates). On the QNAP cell the two-tier bottleneck is the
   LINK, not launch overhead — eager's ~12.5 ms overhead is noise against
   50–500 ms/tok transfer times there. The target cell does not need graphs
   to flip.
2. Graph-compatible dispatch is a **Phase-3 problem gated on Phase-1 physics**:
   evaluate (c) semaphore-node vs a bounded (d)-speculative hybrid ONLY if a
   graphs-class box shows a three-tier win in the roofline. Do not spend the
   B3/B4 budget before the QNAP answer exists.

## GO/NO-GO (unchanged from the directive)

NO-GO if measured naive/microkernel CPU throughput is a small fraction of the
box's memcpy ceiling with no ggml-class remedy, or if the roofline puts the
three-tier ceiling within noise of two-tier on every target box. A stamped
negative files as a finding.
