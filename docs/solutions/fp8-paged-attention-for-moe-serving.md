# How do I run paged decode attention over an FP8 KV cache for a quantized MoE serving path, with sliding windows, attention sinks and a custom scale?
<!-- summary: fp8_paged_decode_attention runs paged flash-decode over an E4M3 KV cache with windows, sinks and a custom scale; the fp8 compute path is supported on sm_89+, the f32 path is open under #319. -->

Use `fp8_paged_attn.fp8_paged_decode_attention` from `grouped-nf4-gemm`: a Triton flash-decode kernel over paged E4M3 K/V blocks that dequantizes in registers, takes `window`, `sinks`, `sm_scale` and per-layer row strides, and is checked against the pure-torch `fp8_paged_attn.paged_attn_ref`. The KV quantize/pack primitives live in `fp8_kv`; the decode glue kernels (fused RMSNorm, residual fold, rotary, router epilogue) live in `int4_b32`.

## Symptoms

- The 4-bit MoE keeps its expert weights small, but the bf16 KV cache and its attention kernel now dominate the decode step and the VRAM budget.
- You need one paged decode kernel for Granite (custom attention scale), Gemma-4 (sliding layers beside full layers, two KV geometries in one pool) and gpt-oss (attention sinks), not a kernel per model.
- The non-GEMM part of the step is a tail of small launches: RMSNorm, residual add, rotary, router softmax and top-k.

## Why it happens

At batch-1 decode nothing is compute-bound; every kernel is a launch plus a read. A per-query-head grid re-reads shared K/V heads under GQA, an unsplit grid leaves most SMs idle, and a materialized bf16 dequant of an fp8 cache adds the round trip the fp8 storage was meant to remove. The kernel's docstring records each decision as a measured fact: grid over (sequence, KV head, split), E4M3 decode as bit assembly, scales loaded at their natural shape.

## Which project solves it

`grouped-nf4-gemm` owns the paged attention kernel and its reference, the FP8 KV primitives, the writable KV row tier (`row_pool.RowPool`) and the decode glue. [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) ([PyPI](https://pypi.org/project/experts4bit-qlora/)) owns KV allocation, block tables, batching, model wiring and the served-path parity gates.

## Install

Kernel package (the minimum route):

```bash
pip install grouped-nf4-gemm
```

Linux, NVIDIA GPU sm_80 or newer, `triton>=3.4` (Linux-only distribution), `torch>=2.8` (pre-releases accepted); CI tests Python 3.11. The fp8 compute path needs sm_89 or newer; sm_80–sm_88 take the f32 path, which is open (below). `fp8_kv` and `paged_attn_ref` are pure torch. Through the model consumer:

```bash
pip install "experts4bit-qlora[fast]"
```

## Two compute paths, two support states

One kernel, two numerical paths, and [`capabilities.json`](../capabilities.json) carries them as two entries because a single status cannot be true of both.

| path | entry in `capabilities.json` | who takes it | requirements (the predicate is `fp8_paged_attn.fp8_compute_unsupported`; the default selector and the path's own asserts share it) | measured on | status |
|---|---|---|---|---|---|
| **fp8 compute** (`compute="fp8"`; the default where it can run) | `fp8-paged-attention-fp8-compute` | sm_89+ with `GNF4_ATTN_COMPUTE` unset and every constraint passing; any explicit fp8 request (never downgraded: the asserts refuse instead) | sm_89 or newer; `v_groups == 1`; `k_groups` in (1, 2, 4, 8, 16); `head_dim // k_groups >= 32`; `q` bf16/fp16; split branch: `ktile >= 32` when supplied; packed branch (`pack_heads=True`): `block_tokens * n_kv_heads >= 32` | RTX 5090 (sm_120) only — sm_89 and sm_90 meet the requirement, no registered cell was run there (claims `gnf4.serve.fp8-paged-attn-windows-sinks-scale`, `gnf4.serve.m3-defaults-on`, both measured) | supported |
| **f32 compute** (`compute="f32"`, split or packed kernel) | `fp8-paged-attention-f32-compute` | sm_80–sm_88 by default; sm_89+ whenever a constraint above fails with the env unset; every explicit `GNF4_ATTN_COMPUTE=f32` / `compute="f32"` on any card | sm_80 or newer | its reference tests were run on an RTX 5090 with the mode forced; no cell on sm_80–sm_88 | **open** — [#319](https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/319): on torch 2.8.0+cu128 / triton 3.4.0 the split and packed f32 modes miss their reference by up to 0.074 against a 0.02 tolerance on 10 of 35 tests, on unmodified `main` (claim `gnf4.open.f32-compute-modes-triton34`, status open, so it backs no capability); the capability entry is `unsupported` until the issue closes |

A kernel that imports and launches is not thereby numerically supported: the status of each path follows the claim and issue register, not the import. `compute_counts()` records which path actually ran, because an environment variable is a request, not an event.

## Smallest correct example

CPU-only: quantize and pack a paged KV pool, run the reference, and check it against plain attention over the dequantized values.

```python
# CPU-only (pure torch; no GPU, no triton)
import torch
from fp8_kv import quantize_kv_fp8, dequant_kv_fp8_ref, pack_kv_block, kv_block_bytes
from fp8_paged_attn import paged_attn_ref

BT, H_KV, H_Q, D, T = 16, 2, 4, 64, 40            # 16-token blocks, GQA 2:1, 40 live tokens
n_blk = (T + BT - 1) // BT
row = kv_block_bytes(BT, H_KV, D)                  # fp8 payload + fp32 scales per block
k_pool = torch.zeros(n_blk * row, dtype=torch.uint8)
v_pool = torch.zeros(n_blk * row, dtype=torch.uint8)
kt, vt = torch.randn(n_blk * BT, H_KV, D), torch.randn(n_blk * BT, H_KV, D)
table = torch.tensor([[2, 0, 1]], dtype=torch.int32)   # block table, permuted on purpose
kd, vd = [], []
for i in range(n_blk):
    r = int(table[0, i])
    qk, sk = quantize_kv_fp8(kt[i * BT:(i + 1) * BT]); pack_kv_block(qk, sk, k_pool[r * row:(r + 1) * row])
    qv, sv = quantize_kv_fp8(vt[i * BT:(i + 1) * BT]); pack_kv_block(qv, sv, v_pool[r * row:(r + 1) * row])
    kd.append(dequant_kv_fp8_ref(qk, sk, dtype=torch.float32))
    vd.append(dequant_kv_fp8_ref(qv, sv, dtype=torch.float32))
q = torch.randn(1, H_Q, D).to(torch.bfloat16)      # one decode token for one sequence
lens = torch.tensor([T], dtype=torch.int32)
out = paged_attn_ref(q, k_pool, v_pool, table, lens, n_kv_heads=H_KV, head_dim=D)  # [1, H_Q, D]

k = torch.cat(kd)[:T].permute(1, 0, 2); v = torch.cat(vd)[:T].permute(1, 0, 2)      # [H_KV, T, D]
qq = q[0].float().view(H_KV, H_Q // H_KV, D)
w = torch.softmax(torch.einsum("hgd,htd->hgt", qq, k) * D ** -0.5, dim=-1)
want = torch.einsum("hgt,htd->hgd", w, v).reshape(1, H_Q, D)
torch.testing.assert_close(out.float(), want, rtol=1e-2, atol=1e-2)
```

GPU: the kernel against the reference, recording which compute mode actually ran.

```python
# GPU (sm_89+ for the fp8 compute default) + triton; sm_80-sm_88 take the f32 path,
# open under #319 on triton 3.4 (claim gnf4.open.f32-compute-modes-triton34)
import torch
from fp8_paged_attn import (fp8_paged_decode_attention, paged_attn_ref,
                            paged_attn_available, compute_counts)
# build q, k_pool, v_pool, table, lens, H_KV, D exactly as in the CPU block
assert paged_attn_available()
before = compute_counts()
got = fp8_paged_decode_attention(q.cuda(), k_pool.cuda(), v_pool.cuda(),
                                 table.cuda(), lens.cuda(),
                                 n_kv_heads=H_KV, head_dim=D, window=0, sinks=None)
mode = "fp8" if compute_counts()["fp8"] > before["fp8"] else "f32"   # what ran, not what was asked
want = paged_attn_ref(q, k_pool, v_pool, table, lens, n_kv_heads=H_KV, head_dim=D)
tol = 1.5e-1 if mode == "fp8" else 2e-2          # the suite's per-mode bounds
torch.testing.assert_close(got.cpu().float(), want.float(), rtol=tol, atol=tol)
print("compute mode:", mode)
```

## Expected result

The CPU block passes: the packed pool, read back through the reference, reproduces attention over the dequantized K/V to bf16 output precision. The GPU block prints the compute mode that ran and passes its bound. `window=W` restricts each query to the last `W` keys; `sinks` (an `[H_q]` fp32 tensor) adds one logit per query head to the softmax denominator with no value, the gpt-oss `s_aux` convention; `sm_scale` replaces `D ** -0.5`. `paged_attn_ref` accepts the same three.

## Supported scope

- `q [B, H_q, D]` bf16/fp16, one decode token per sequence; flat uint8 `k_pool`/`v_pool` of 16-token packed rows in the `fp8_kv.pack_kv_block` layout (`"tokens"` or `"heads"`); `block_table [B, MAX_BLOCKS]` int32; `seq_lens [B]` int32.
- Options: `window`, `sinks`, `sm_scale`, `k_groups` in (1, 2, 4, 8, 16) sub-row key scales, `v_groups`, `k_row_bytes`/`v_row_bytes` for a pool whose stride is wider than this layer's row, `compute="f32"|"fp8"`, `pack_heads`, `layout`.
- Compute-mode policy: an unset `GNF4_ATTN_COMPUTE` picks fp8 where `fp8_paged_attn.fp8_compute_unsupported(...)` returns `None`, else f32; an explicit request is never downgraded; `compute_counts()` records what ran.
- KV write side: `fp8_kv.quantize_kv_fp8(x, group=None)` (per-token-per-head E4M3 with an fp32 scale, or sub-row groups), `fp8_kv.fp8_kv_append_t1(...)` for a one-launch, capture-safe T=1 append, `row_pool.RowPool` for the device/pinned-host KV row tier.
- Decode glue in `int4_b32`: `rmsnorm_rows`, `rmsnorm_resid_rows` (with a residual multiplier), `scaled_resid_add_rows`, `rope_norm_heads`, `rope_heads`, `router_epilogue` (softmax-then-top-k, or top-k on logits with an optional bias for gpt-oss and GraniteMoe), `swiglu_rows`, `combine_rows`, `reduce_partials`.

## Limitations

- Open: the f32 compute modes (split, packed) miss their reference on torch 2.8.0+cu128 / triton 3.4.0 on sm_120, on unmodified `main` (`#319`, claim `gnf4.open.f32-compute-modes-triton34`: up to 0.074 against a 0.02 tolerance, 10 of 35 tests); the fp8 modes pass. Gate a lane there with `-k "f8dot or pf8"` ([`STATUS.md`](../STATUS.md)). On a pre-Ada card the default is f32, so the GPU block above can fail on that torch/triton pair; an explicit `compute="f32"` on any card is on the same open path.
- The fp8 compute path requires sm_89+, `v_groups == 1`, `k_groups` in (1, 2, 4, 8, 16), `head_dim // k_groups >= 32`, bf16/fp16 `q`, `ktile >= 32` when supplied (split branch) and `block_tokens * n_kv_heads >= 32` (packed branch); it adds one e4m3 rounding on `q` and on `p`, hence its wider tolerance. Its registered cells are RTX 5090 only; the sm_89+ requirement is the kernel's precondition, not a measurement on Ada or Hopper.
- `pack_heads=True` falls back to the split kernel where its shared-memory demand exceeds the card, with a one-time `RuntimeWarning`.
- Served-path fidelity (Granite, gpt-oss, Gemma-4) is measured in experts4bit-qlora's receipts, which are private; the evidence here is kernel-level parity against `paged_attn_ref`. The glue-kernel composition is measured-private (claim `gnf4.serve.decode-glue-kernels`).
- Not a serving engine; CUDA + Triton only; `paged_attn_ref` is slow, for test sizes.

## Related

[`STATUS.md`](../STATUS.md) · [`claims.json`](../claims.json) · [`context-budgets.md`](../context-budgets.md) (KB/token, rung one only) · [`int4-decode-gemv.md`](int4-decode-gemv.md) · [`nf4-grouped-gemm-without-bf16-materialization.md`](nf4-grouped-gemm-without-bf16-materialization.md) · `kernel/test_fp8_paged_attn.py` · `kernel/test_fp8_kv.py`

## Evidence

Register: [`claims.json`](../claims.json). Measured, fp8 compute path: claim `gnf4.serve.fp8-paged-attn-windows-sinks-scale` (windows, sinks, scale and stride overrides; fp8-mode GPU suite on an RTX 5090), claim `gnf4.serve.m3-defaults-on` (both decode knobs on by default, capability-conditional, with a paired perplexity check). Open, f32 compute path: claim `gnf4.open.f32-compute-modes-triton34` (backs nothing). Measured-private, labelled: claim `gnf4.serve.decode-glue-kernels` (its own capability entry, `decode-glue-kernels`). The single-stream decode anchor, claim `gnf4.serve.decode-anchor-5090`, is a consumer-measured anchor of the serving class on a knob-off basis, not a measurement of this kernel, and is not cited as its evidence. Receipts: `kernel/RESULTS-m3-default-on.md`, `kernel/RESULTS-k8-fp8-compute-attn.md`, `kernel/RESULTS-m2-anchor-recert.md`.
