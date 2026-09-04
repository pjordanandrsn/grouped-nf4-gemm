# How do I run a single-token INT4 decode GEMV over routed MoE experts, and pack calibrated (GPTQ) weights for it?

Use the int4-b32 lane of `grouped-nf4-gemm`: `int4_pack_ref.pack_int4_b32` packs a weight onto a uniform symmetric int4 grid with one fp16 scale per 32 elements, `int4_b32.gemv_int4_b32` runs the decode GEMV on int8-quantised activations with exact integer accumulation and split-K partials, and `gptq_pack.gptq_pack_int4_b32` chooses grid points against calibration activations while emitting byte-identical format.

## Symptoms

- At batch-1 decode the expert and attention projections are GEMVs (one row each), bound by weight bytes and launch count; a codebook gather per element (NF4's LUT) is the wrong inner loop for that regime.
- You want an INT4 format that unpacks arithmetically so the inner loop is integer multiply-accumulate, exact in int32.
- Round-to-nearest int4 on attention projections fails your perplexity gate, and you want a calibrated pack (GPTQ-style) that keeps the same bytes, scales and kernels.
- You need the same format under CUDA-graph capture for batched decode.

## Why it happens

NF4 is a non-uniform grid, so a GEMV pays a gather per weight before it can multiply. A uniform grid decodes as `(nibble - 8) * scale`, and with int8 activations the block dot product is an exact int32 sum; only the fp32 scale products and the final bf16 rounding are inexact. Separately, rounding each weight to its nearest grid point minimises weight error, which is not what a perplexity gate measures; GPTQ minimises activation-weighted output error under `H = 2 X Xᵀ`, pushing each column's residual into the columns that follow (`kernel/gptq_pack.py` docstring).

## Which project solves it

`grouped-nf4-gemm` owns the format, both packers and their reference decode, the GEMV, the grouped M-tile int4 GEMM against prebuilt tiles, and the split-K reduce. [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) ([PyPI](https://pypi.org/project/experts4bit-qlora/)) decides which projections use the format, runs the perplexity gate, and serves.

## Install

```bash
pip install grouped-nf4-gemm
```

Linux, NVIDIA GPU sm_80 or newer (the lane was tuned on sm_120, the primary serving target), `triton>=3.4` (Linux-only distribution), `torch>=2.8` (pre-releases accepted); CI tests Python 3.11. `int4_pack_ref` and `gptq_pack` are pure torch; `int4_b32` imports triton at module level. Through the consumer: `pip install "experts4bit-qlora[fast]"`.

## Smallest correct example

CPU-only: the plain and the calibrated pack, both decoded by the reference.

```python
# CPU-only (pure torch; no GPU, no triton)
import torch
from int4_pack_ref import BLOCK, pack_int4_b32, dequant_int4_ref
from gptq_pack import HessianAccumulator, gptq_pack_int4_b32

N, K = 64, 256                                    # K % 32 == 0
w = torch.randn(N, K) * 0.1
packed, scales = pack_int4_b32(w)                 # [N, K//2] u8, [N, K//32] fp16
deq = dequant_int4_ref(packed, scales, N, K)      # [N, K] fp32
# round-to-nearest lands within half a grid step (scale rounded to fp16)
assert ((w - deq).abs() <= 0.51 * scales.float().repeat_interleave(BLOCK, dim=1)).all()

acc = HessianAccumulator(K)                       # accumulates H = 2 X X^T
x = torch.randn(1024, K) * torch.logspace(-1, 1, K)   # skewed input channels
acc.add(x)
p_cal, s_cal = gptq_pack_int4_b32(w, acc.H)       # same byte format, different grid points
assert p_cal.shape == packed.shape and p_cal.dtype == packed.dtype
assert s_cal.shape == scales.shape and s_cal.dtype == scales.dtype
d_cal = dequant_int4_ref(p_cal, s_cal, N, K)
err_rtn = (x @ (deq - w).t()).pow(2).mean().item()
err_cal = (x @ (d_cal - w).t()).pow(2).mean().item()
print("output MSE  rtn:", err_rtn, " calibrated:", err_cal)   # calibrated is expected lower
```

GPU: the grouped decode GEMV against the fp32 reference of the same int4 values and int8 activations.

```python
# GPU (sm_80+) + triton
import torch
from int4_pack_ref import BLOCK, pack_int4_b32, dequant_int4_ref
from int4_b32 import quant_x_rows, gemv_int4_b32

E, N, K, R = 8, 256, 2048, 4
W = torch.randn(E, N, K) * 0.1
pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(E)])
packed = torch.stack(pk).cuda().contiguous()      # [E, N, K//2] u8
scales = torch.stack(sc).cuda().contiguous()      # [E, N, K//32] fp16
eids = torch.tensor([5, 0, 5, 2], dtype=torch.int32, device="cuda")   # expert per row
x = (torch.randn(R, K) * 0.2).cuda().to(torch.bfloat16)

xq, xs = quant_x_rows(x)                          # int8 rows + fp32 per-32 scales
out = gemv_int4_b32(xq, xs, packed, scales, eids, N, K)   # [R, N] bf16

ref = torch.stack([
    (dequant_int4_ref(packed[int(e)].cpu(), scales[int(e)].cpu(), N, K).cuda()
     * (xq[i].float() * xs[i].repeat_interleave(BLOCK))[None, :]).sum(-1)
    for i, e in enumerate(eids)])
assert (out.float() - ref).abs().max() <= ref.abs().max() * 2 ** -7   # bf16 output rounding only
```

## Expected result

The CPU block passes its assertions and prints two output errors; on skewed channels the calibrated one is expected lower (the in-tree test asserts a margin over three regimes). The GPU block returns `[R, N]` bf16 whose only deviation from the fp32 reference is output rounding, because the integer accumulation is exact.

## Supported scope

- Format: symmetric grid, levels -8..7 stored offset-binary, even `k` in the low nibble; `packed [E, N, K//2]` uint8, `scales [E, N, K//32]` fp16; `K % 32 == 0`.
- `int4_b32.quant_x_rows(x [R, K]) -> (int8 [R, K], fp32 [R, K//32])`; `gemv_int4_b32(..., part=None)` accepts a preallocated partials buffer under capture; `int4_b32.reduce_partials(part, sk, R, N)` is the split-K reduce plus bf16 cast in one launch.
- Batched decode: `int4_b32.gemm_int4_b32_grouped_captured(aq_sorted, as_sorted, packed, scales, t_row0, t_rows, t_group)` over tiles from `nf4_grouped.build_group_tiles_device`, legal inside CUDA-graph capture; rows return sorted and the caller scatters by the inverse of `order`.
- Calibration: `gptq_pack.HessianAccumulator(in_features, device=None)` computes each batch's Gram where the activations are and can keep the Hessian off-device (`device="cpu"`); `gptq_pack_int4_b32(w, hessian, damp=0.01, blocksize=128)`.
- The decode glue kernels (`rmsnorm_rows`, `rope_norm_heads`, `router_epilogue`, `swiglu_rows`, `combine_rows`, and the rest) also live in `int4_b32`: [`fp8-paged-attention-for-moe-serving.md`](fp8-paged-attention-for-moe-serving.md).

## Limitations

- Decode-only: rows are single tokens. Prefill and training stay on the NF4 and MXFP4 paths.
- Measured refusals, kept: the format stays off the `lm_head`, and dense attention shapes lose to bf16 in both the GEMV and small-M GEMM regimes. Not a registered claim: this is recorded in the `notes` field of claim `gnf4.serve.int4-b32-gemv` (measured-private), not in that claim's measured statement.
- Calibrated packing is a Qwen3-30B-A3B result; the consumer's gate refused it on Mixtral and OLMoE. Not a registered claim: this is recorded in the `notes` field of claim `gnf4.serve.gptq-pack-int4-b32` (measured-private), not in that claim's measured statement.
- Pack from source weights only; quantising onto an already-quantised grid compounds the cost (`int4_pack_ref` docstring).
- Time under CUDA-graph replay, never eager: eager sweeps anti-select split-K. Split-K on the NF4 decode GEMV was refuted and ships dormant as the evidence ([`STATUS.md`](../STATUS.md)).
- `int4_b32` is not importable without triton; no ROCm or XPU ([`PORTABILITY.md`](../PORTABILITY.md)).
- This lane's throughput and quality numbers are measured-private: real runs whose receipts live in a private audit tree, not checkable here.

## Related

[`STATUS.md`](../STATUS.md) · [`claims.json`](../claims.json) · [`nf4-grouped-gemm-without-bf16-materialization.md`](nf4-grouped-gemm-without-bf16-materialization.md) · [`native-mxfp4-moe-inference.md`](native-mxfp4-moe-inference.md) (`gemv_mxfp4_b32` shares this GEMV's plan) · `kernel/test_int4_b32.py` · `kernel/test_gptq_pack.py`

## Evidence

Register: [`claims.json`](../claims.json). Measured-private, labelled as such: claim `gnf4.serve.int4-b32-gemv` (dense and grouped decode cells, plus the M-tile GEMM), claim `gnf4.serve.gptq-pack-int4-b32` (calibrated attention packs on Qwen3-30B-A3B). Nothing on this page is at the confirmed tier. What is checkable here is correctness: `kernel/test_int4_b32.py` pins the GEMV and grouped GEMM against `dequant_int4_ref`; `kernel/test_gptq_pack.py` pins format identity and the calibrated-versus-rounding margin on CPU.
