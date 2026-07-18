# COMPETITIVE.md — incumbent register (R9: name the incumbent)

**STATUS: DRAFT for operator review — UNCOMMITTED, private-fork only until
reviewed.** Plan-mandated deliverable (AGENT-PLAN-mxfp4.md §5, rule R9),
drafted 2026-07-17 at mxfp4-lane activation. The operator decides at review
whether this file (or a subset) ports to the public remote.

**Rule being enforced (R9/B2):** any public comparative or "first/only" claim
must cite its entry here, in the same breath. **No entry, no claim.**
Tier discipline (R3/R6): every number below is either **VERIFIED** (citation or
first-party receipt named) or **RECORDED-UNVERIFIED** (from the 2026-07-15
adversarial audit; must be verified before any public sentence leans on it).

---

## 1. Unsloth — gpt-oss QLoRA path

- **RECORDED-UNVERIFIED (audit 2026-07-15):** gpt-oss-20b QLoRA at
  **12.8–14 GB** via MXFP4→NF4/bnb conversion; gpt-oss-120b at **~65 GB**.
  MXFP4 backward (`W_TRANSPOSE` or equivalent) unimplemented.
- **Bounded re-check 2026-07-17** (`gh api search/commits
  q=repo:unslothai/unsloth+mxfp4`): commit 2026-06-22 (#6563) — *"loader:
  gpt-oss MXFP4 default-4bit takes the MXFP4 dtype path, not BnB."* Unsloth's
  **loader** now keeps native MXFP4 at load for gpt-oss. The stale part of the
  audit line is therefore the conversion mechanism, not (yet) the training
  claim: **whether their training path trains over native MXFP4 or dequantizes
  at forward is unverified** — that is WATCHLIST trigger 1's verification
  target. Do not repeat "Unsloth converts MXFP4→NF4 to fine-tune" publicly
  until re-verified against their current loader.
- **Consequence lock (B1):** gpt-oss-20b is a dev target ONLY. Any claim
  shaped like "fine-tune gpt-oss on a 16 GB card" is PROHIBITED — Unsloth
  ships that user outcome. 20b appears publicly only as the provenance-receipt
  proof-of-method.

## 2. Unsloth MoE-4bit correctness gaps — VERIFIED, operator-authored

Filed by the operator (pjordanandrsn) 2026-07-02, both OPEN as of 2026-07-17:

- **unsloth-zoo #849** — `preprocess_weight` silently transposes expert
  weights when `2*moe_intermediate == hidden_size` (square gate_up carries no
  layout signal → guesses "already correct" → trains on transposed weights).
  OLMoE's exact shape triggers it; the FP8 path shares the surface.
  **PR #913** (BardiaKoopah, 2026-07-18, OPEN) fixes #849 only —
  disambiguates via the sibling projection; sound by code-read, **not locally
  verified** (operator decision pending).
- **unsloth-zoo #850** — `load_in_4bit` quantizes fused experts generically
  but dequant-at-forward routing is a per-arch list; any arch off the list
  (OLMoE) loads, VRAM drops, then **crashes on first forward**. Gist repro
  in-thread; no maintainer reply in 15 days. **UNTOUCHED by #913.**
- **Failure-mode taxonomy** (usable in public writing, each entry cited):
  bnb #1849 = fail-silent-too-little → OOM; zoo #850 = fail-silent-too-much →
  crash; experts4bit-qlora = quantized storage WITH faithful forward and a
  fail-closed-loud loader. Provenance: both issues filed before e4b PR #24
  existed.

## 3. unsloth #4032 (bnb-4bit MoE support) — VERIFIED via gh API

**CLOSED, stateReason=COMPLETED, 2026-06-18** (checked 2026-07-15 and
2026-07-16). What shipped is **dequant-path MoE support** (OLMoE / Qwen3-MoE /
Gemma-4 text; tracks bnb#1849) — **not** a fused grouped kernel. The
bnb-4bit-MoE "unsupported" wedge has collapsed; the provenance niche (B1) is
the selected pivot. WATCHLIST trigger 4 (fused grouped bnb-NF4 MoE GEMM) has
NOT fired.

## 4. Marlin / Machete + vLLM fused Marlin MoE

**RECORDED-UNVERIFIED (audit 2026-07-15):** fused 4-bit MoE GEMM for
GPTQ/AWQ/MXFP4/NVFP4; Ampere/Hopper lean; known **K=N=2880 alignment failure
on gpt-oss shapes**. Needs the vLLM issue/PR citation before public use — as a
coverage difference, never a dunk (B4).

**Kernel phrasing lock (R7-ext/B4), restated so claim-authors see it here:**
sanctioned description is *"fused grouped GEMM on bitsandbytes NF4-packed
weights with host streaming, on consumer GPUs."* Bare "fused 4-bit MoE GEMM"
is prohibited as a novelty claim; Marlin/Machete are acknowledged by name
wherever kernel novelty is asserted.

## 5. llama.cpp / ik_llama.cpp / ktransformers

- **RECORDED-UNVERIFIED (audit 2026-07-15):** Qwen3-235B ~**7.4 tok/s**
  community benchmark (3090 + 128 GB, IQ3_K — quant-precision caveat vs NF4);
  gpt-oss-120b CPU-MoE offload ~**28–30 tok/s** on high-end desktops.
- **R7 lock:** no inference-superiority claims for gpt-oss-class sparsity —
  gpt-oss (~5.1B active/tok) is llama.cpp's best case and its GGUFs preserve
  native MXFP4 blocks. Any public gpt-oss comparison acknowledges llama.cpp
  explicitly. **B3:** no throughput-leadership claims for Qwen3-235B;
  sanctioned differentiators are exact-checkpoint-bytes fidelity
  (quality-per-bit, precision caveat stated), energy-per-token where measured,
  and the receipts methodology itself.

## 6. bitsandbytes

- **#1849 — VERIFIED (operator-filed; reproduced in-lane):** stock
  `load_in_4bit` walks `nn.Linear` only; fused 3-D expert stacks silently
  stay bf16 (OLMoE "4-bit" at ~9.55 GiB OOMs a 12 GB A2000). First-party
  receipts: experts4bit-qlora loader + router-probe Phase-1 captures.
- **Release state (checked 2026-07-17, `gh release list`):** latest stable
  **0.49.2** (2026-02-16); v0.50.0 = WATCHLIST trigger 3 (audit contents for
  grouped/MoE 4-bit kernels; fires the PR #1965 comment per its gate).
- Multiarch rows (already VERIFIED in-repo, citations in
  `docs/PORTABILITY.md`): ROCm backend preview / PyPI wheel CUDA-only; XPU
  preview-grade; Triton-on-RDNA4 ~30–50% behind hand-HIP; Strix Halo
  ~212–215 GB/s measured.

## 7. transformers (HF)

- **VERIFIED in-lane:** the MXFP4 path **dequantizes to bf16** to train —
  the memory-blowing incumbent baseline for the provenance headline
  (`convert_moe_packed_tensors`). It is also the dequant **ground truth**: e4b
  `mxfp4.py` matches it bit-identically on real gpt-oss-20b/120b shapes
  (receipt: experts4bit-qlora PR #24 identity gate,
  `tests/test_mxfp4_dequant.py` on real released shards — **merged `7e8dd12`
  and SHIPPED in experts4bit-qlora 0.4.0 on PyPI, 2026-07-18**).
- transformers ≥5 ships dequant-path MoE-4bit support for three architectures
  (see entry 3); gpt_oss expert stacks on disk remain MXFP4 blocks/scales.

---

*Verification debt (do before any public claim citing entries 1, 4, 5):
primary-source citations for Unsloth VRAM numbers + current training path;
the Marlin K=N=2880 issue link; the ik_llama.cpp 7.4 tok/s thread. R6: until
then those numbers are silence in public.*
