# Portability risk register — grouped-nf4-gemm

**Phase 1 deliverable (multiarch plan P1.1/P1.3).** What must be verified or
changed to run the kernel outside CUDA/sm_86. No claim here is a port result —
this is the pre-port hazard list. Tier language (R3): every non-CUDA row is
`port target`, not "supported," until a confirmatory passes there.

## Kernel-code hazards (single kernel source; the risks are backend semantics)

| # | construct | CUDA (sm_86, shipped) | HIP / CDNA + RDNA | XPU / Intel | action |
|---|---|---|---|---|---|
| K1 | **warp / sub-group width** | 32 | **64 on CDNA**, 32 on RDNA3/4 | 16 or 32 (sub-group) | `num_warps=2/4/8` maps to different thread counts; occupancy + tile assumptions shift. Re-autotune per backend, don't port the constants. |
| K2 | **`tl.dot` accumulate dtype** | TF32 inputs, fp32 acc — the *fidelity edge* | no TF32; MFMA (CDNA) / WMMA (RDNA) with fp32 acc | XMX with fp32 acc | The "fused is more accurate than dequant" claim is **TF32-specific**. On MFMA/WMMA/XMX fp32-acc it likely holds, but it is a NEW fidelity measurement per backend, not an inherited claim. |
| K3 | **`tl.gather` (v6 register-LUT)** | present (triton 3.4) | availability varies by triton-ROCm version | availability varies by intel-xpu-triton | already fenced: `prefill_variant` auto-falls-back to the V0 loop when `tl.gather` absent. Verify the fallback fires, don't assume gather. |
| K4 | **SMEM / LDS ceiling** | BLOCK_K=128 (GROUPS=2) already dies on sm_86 (181 KB) | LDS typically 64 KB (CDNA/RDNA) — the M-tile 128×128 config may not fit | SLM budget differs | tile configs are per-backend; the SMEM-overflow failure mode is expected on tighter-LDS parts. |
| K5 | **bf16 epilogue / `tl.bfloat16`** | native | native (CDNA/RDNA3+) | native (recent) | low risk; confirm on older RDNA. |
| K6 | **`_sm_count` device API** | `torch.cuda.get_device_properties().multi_processor_count` | ROCm torch aliases `.cuda` (works); CU count returned | needs `torch.xpu` | **fixed in CI** to be CPU/XPU-safe; `backends/` centralizes detection. |

## Dependency status (P1.3 — verified 2026-07-15, cited; ranges where fluid)

- **bitsandbytes NF4 on ROCm:** multi-backend is **preview**; all features
  reportedly work on RDNA + CDNA, buildable ROCm 6.2–7.2.3 **from source (HIP,
  CMake, `-DGPU_TARGETS=gfx…`)**. The **PyPI wheel ships CUDA-only precompiled
  binaries** — so a ROCm run needs a source build or a fork wheel, not
  `pip install bitsandbytes`. (bnb issue #1608; ROCm `rocm_enabled_multi_backend`
  branch; HF bnb install docs.)
- **bitsandbytes on Intel XPU:** **preview-grade** (multi-backend alpha names
  Intel CPU+GPU; less mature than ROCm). Treat as the lowest-readiness lane.
- **Triton on RDNA4 (gfx1201):** runtime kernel-gen works; AOTriton 0.10b lists
  gfx1201 official — **but** competition/community findings put Triton GEMM at
  **~30–50% of hand-written HIP** on RDNA4, and gfx1201 has known table-gaps
  elsewhere (AITER silent FP32 fallback, ROCm/TransformerEngine #520). So RDNA4
  projection bands carry a **"Triton maturity"** widening note.
- **intel-xpu-backend-for-triton:** exists and is maintained; gfx-equivalent
  maturity for our exact ops unverified — flag, don't assume.

**Consequence for Phase-2 ordering (per the dependency audit above):** Intel drops **below** AMD
— bnb-XPU preview + XPU-triton unknowns make it the riskiest first confirmatory.
AMD MI300X (rented, bnb source-build, CDNA MFMA) is the cleanest first port.

## What Phase 1 does NOT change

Per R3, no README/claim language is altered by this file. Every non-CUDA entry
stays `port target`. The interpreter-mode contract suite (CI) validates
*semantics* on any host pre-silicon; correctness-on-hardware and any perf number
require a `RESULTS-<platform>.md` confirmatory under `PROTOCOL-multiarch.md`.

## Sources

bitsandbytes multi-backend / ROCm: github.com/bitsandbytes-foundation/bitsandbytes
issue #1608, discussion #1339, ROCm/bitsandbytes `rocm_enabled_multi_backend`,
HF bitsandbytes installation docs. Triton RDNA4: ROCm compatibility matrix,
apollo-mg RDNA4/ROCm 7.1 guide, ROCm/TransformerEngine #520, AOTriton 0.10b
notes. Intel: github.com/intel/intel-xpu-backend-for-triton. *Captured
2026-07-15; verify before a port session (R6).*
