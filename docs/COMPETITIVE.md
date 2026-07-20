# COMPETITIVE.md — R9 enforcement artifact (no entry, no claim)

Every public comparative or "first/only" claim must cite an entry here.
Entries carry a verification status and as-checked date; **"reported" entries
may NOT back a public claim until upgraded to "verified."** Moves to the
public repo at the gate.

## Unsloth (gpt-oss fine-tuning) — verified 2026-07-19

- Fine-tunes gpt-oss by loading the native MXFP4 checkpoint and **converting
  expert weights to bnb-linear/NF4 for QLoRA** ("these parameters were
  converted into nn.Linear layers"); 20b trains in under ~24 GB (their docs;
  widely reported 12.8–14 GB). Their own docs state **"currently no
  framework supports fp4 or MXFP4 training"** — i.e., the backward pass runs
  on CONVERTED weights, and conversion destroys byte identity with the
  shipped checkpoint.
  - https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune
  - https://unsloth.ai/blog/gpt-oss
- Consequence for our claims: the **user outcome** "fine-tune gpt-oss-20b on
  a 16 GB card" is THEIRS (never our headline); the **byte-provenance
  receipt** (native bytes, hash-identical pre/post) is not producible by a
  conversion path. 120b "~65 GB" figure: **reported, not re-verified** — do
  not quote without checking their current docs on gate day.

## Marlin / Machete + vLLM fused MoE — verified 2026-07-19

- vLLM's Marlin MoE kernel family provides fused 4-bit MoE GEMM
  (GPTQ/AWQ/MXFP4 classes) — the prior art the B4 phrasing lock
  acknowledges by name.
- **Concrete coverage difference (cite as difference, not dunk):** the
  Marlin MoE GEMM fails on gpt-oss shapes — K=N=2880 is not 128-aligned, so
  the kernel's thread-config lookup finds no valid configuration
  ([vllm#38022](https://github.com/vllm-project/vllm/issues/38022),
  MXFP4-quantized gpt-oss-20b). Note honestly: on Blackwell, vLLM routes
  MXFP4 MoE to FlashInfer/TensorRT-LLM kernels with native MXFP4 tensor
  cores (https://blog.vllm.ai/2025/08/05/gpt-oss.html) — the gap is a
  consumer/Ampere-class coverage gap, not a universal one.

## llama.cpp CPU-MoE offload (gpt-oss) — verified 2026-07-19

- gpt-oss-120b on high-end desktops with `--n-cpu-moe`: **~28–30 tok/s tg**
  (RTX 5090 + fast DDR5 ≈ 30 t/s at zero context; partial-CPU configs bench
  tg=28 t/s; RAM speed is decisive — XMP off→on tripled tg 10→30).
  - https://github.com/ggml-org/llama.cpp/discussions/15396
  - https://carteakey.dev/blog/optimizing-gpt-oss-120b-local-inference/
  - https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/
- This is the R7 baseline every public gpt-oss comparison must acknowledge:
  for pure inference, gpt-oss is llama.cpp's best case and its GGUFs
  preserve the native MXFP4 blocks. Our differentiator is TRAINING (native
  bytes + provenance), never inference supremacy.
- House-measured anchors (own receipts, S3/S4 ladder): llama ncmoe32@t24 =
  45.34 ± 0.18 tok/s on the 24-vCPU S-box; true best ncmoe28 ≈ 49.8–51.6
  @ ~13 GB (bench/homelab/RESULTS-pipelined-ladder.md, stamped).

## bitsandbytes — verified 2026-07-19

- Stable = 0.49.2; v0.50.0 unreleased (dev builds only on a third-party
  Windows/ROCm fork). Watchlist trigger 3 not fired.
  https://github.com/bitsandbytes-foundation/bitsandbytes/releases
- CPU backend ships `dequantize_4bit` for the standard packed layout with an
  AVX-512 kernel (verified in installed 0.49.2 source); on AVX2-only hosts
  it falls back below naive torch (own receipts,
  bench/cold-engine/receipts-*-qnap.json — the cold-engine lane's finding;
  upstream issue parked post-gate).

## ik_llama.cpp / ktransformers (Qwen3-235B) — reported, NOT verified

- Community figure ~7.4 tok/s (3090 + 128 GB, IQ3_K) — **do not use in any
  public claim without verification**; quant-precision caveat (IQ3_K ≈ 3-bit
  vs our NF4 4-bit) mandatory if ever cited. B3 already bans 235B
  throughput-leadership claims regardless.

## Watchlist tie-in (checked 2026-07-19, both clear)

- Trigger 1 (Unsloth ships MXFP4 backward): NOT fired — see Unsloth entry.
- Trigger 3 (bnb 0.50.0): NOT fired — see bitsandbytes entry.
