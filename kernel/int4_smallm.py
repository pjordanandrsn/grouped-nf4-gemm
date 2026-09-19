# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Small-M int4-b32 GEMM for the attention projections (lane K16, PREREG-k16-smallm-int4-gemm.md).

``y[M, N] = x[M, K] @ dequant(packed[N, K//2], scales[N, K//32]).T`` for ``M <= 16``, ONE launch.

Why a second int4 GEMM exists beside ``int4_b32._gemm_int4_b32_grouped`` (K14): that kernel does exact
integer MMA with ONE ``tl.dot`` per 32-wide k-block, because its per-(row, k-block) fp32 scale product
must be applied after each dot -- at M=16 on a single projection that is 64 tiny dots per program and no
expert-count parallelism, 22% of the streaming ceiling on the 5090 (K14). This kernel does what Marlin
does (K15): dequantise the int4 tile IN REGISTERS, scale it per 32-block inside the tile, and run bf16
tensor-core MMA over a fat K-chunk (KC = 128..256), with split-K across programs and the reduction fused
into the same launch (the last program to finish a column block sums the fp32 partials in split order).
Activations stay bf16 -- no int8 activation quantisation on this path -- so the arithmetic is the
dequant-then-GEMM path's, without materialising the bf16 weight.

Contract (registered before any perf number): deterministic for a fixed config; within one bf16 output
ulp of ``x @ dequant_int4_ref(packed, scales).T`` on every shape; within one bf16 ulp across SK.
"""
import torch

from _triton_shim import triton, tl

BLOCK = 32          # the int4-b32 scale block, as int4_pack_ref.BLOCK
_SUPPORTED_KC = (256, 128, 64, 32)


@triton.jit
def _gemm_int4_b32_smallm(x_ptr, w_ptr, ws_ptr, part_ptr, cnt_ptr, out_ptr,
                          M, N, K: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          KC: tl.constexpr, SK: tl.constexpr):
    """Grid ``(cdiv(N, BLOCK_N), SK)``. Program (pid_n, pid_k) accumulates rows [0, M) x its N block
    over its K span in fp32, then either stores (SK == 1) or writes an fp32 partial and lets the LAST
    arriving program of the column block reduce all SK partials in split order and store bf16."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    KB: tl.constexpr = K // 32                    # scale blocks along K
    NKC: tl.constexpr = K // KC                   # K chunks
    PER_SPLIT: tl.constexpr = NKC // SK           # chunks per program
    NSC: tl.constexpr = KC // 32                  # scale blocks per chunk
    offs_kc = tl.arange(0, KC)
    offs_kh = tl.arange(0, KC // 2)               # packed bytes per row per chunk
    offs_sc = tl.arange(0, NSC)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c in range(pid_k * PER_SPLIT, (pid_k + 1) * PER_SPLIT):
        k0 = c * KC
        a = tl.load(x_ptr + offs_m[:, None] * K + k0 + offs_kc[None, :],
                    mask=m_mask[:, None], other=0.0).to(tl.bfloat16)                 # [BM, KC]
        wb = tl.load(w_ptr + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kh[None, :],
                     mask=n_mask[:, None], other=0).to(tl.int32)                     # [BN, KC/2] bytes
        lo = ((wb & 0xF) - 8).to(tl.float32)                                         # even k = LOW nibble
        hi = (((wb >> 4) & 0xF) - 8).to(tl.float32)                                  # odd k = HIGH nibble
        w = tl.interleave(lo, hi)                                                    # [BN, KC] in k order
        sc = tl.load(ws_ptr + offs_n[:, None] * KB + (k0 // 32) + offs_sc[None, :],
                     mask=n_mask[:, None], other=0.0).to(tl.float32)                 # [BN, NSC]
        w3 = tl.reshape(w, (BLOCK_N, NSC, 32)) * sc[:, :, None]                     # scale per 32-block, in-tile
        wsc = tl.reshape(w3, (BLOCK_N, KC)).to(tl.bfloat16)
        acc += tl.dot(a, tl.trans(wsc), out_dtype=tl.float32)                        # [BM, BN] bf16 MMA
    ooff = offs_m[:, None] * N + offs_n[None, :]
    omask = m_mask[:, None] & n_mask[None, :]
    if SK == 1:
        tl.store(out_ptr + ooff, acc.to(tl.bfloat16), mask=omask)
    else:
        # fp32 partial [SK, BLOCK_M, N]; then the last arriver reduces IN SPLIT ORDER (deterministic)
        tl.store(part_ptr + pid_k * (BLOCK_M * N) + ooff, acc, mask=omask)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
        if prev == SK - 1:
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in range(0, SK):
                total += tl.load(part_ptr + s * (BLOCK_M * N) + ooff, mask=omask, other=0.0)
            tl.store(out_ptr + ooff, total.to(tl.bfloat16), mask=omask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)                                       # armed for the next launch


def plan_smallm(N: int, K: int, *, block_n: int = 64, kc: int = 128, sk: int = 4):
    """Legalise a config for (N, K): KC must divide K and be a multiple of 32; SK must divide K // KC.
    Falls back downward (never silently to a different arithmetic). Returns (block_n, kc, sk)."""
    if K % 32:
        raise ValueError(f"K={K} is not a multiple of the int4-b32 scale block ({BLOCK})")
    kc = next((c for c in _SUPPORTED_KC if c <= kc and K % c == 0), None)
    if kc is None:
        raise ValueError(f"no supported K chunk divides K={K} (supported {_SUPPORTED_KC})")
    nkc = K // kc
    while sk > 1 and nkc % sk:
        sk //= 2
    return int(block_n), int(kc), int(max(1, sk))


def smallm_workspace(N: int, *, block_m: int = 16, block_n: int = 64, sk: int = 4, device="cuda"):
    """Preallocated (part, cnt) so the launch is capture-legal: ``part [SK, 16, N] fp32``, ``cnt [cdiv(N, BN)] int32``
    zeroed once; the kernel leaves ``cnt`` zeroed after every launch."""
    part = torch.empty(sk, block_m, N, dtype=torch.float32, device=device)
    cnt = torch.zeros(triton.cdiv(N, block_n), dtype=torch.int32, device=device)
    return part, cnt


def gemm_int4_b32_smallm(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, *,
                         block_n: int = 64, kc: int = 128, sk: int = 4, warps: int = 4, stages: int = 2,
                         workspace=None) -> torch.Tensor:
    """``x [M, K]`` (bf16/fp16/fp32; M <= 16), ``packed [N, K//2] uint8``, ``scales [N, K//32]`` (fp16/bf16/fp32)
    in the int4-b32 layout (``int4_pack_ref.pack_int4_b32``). Returns ``[M, N]`` bf16. One launch."""
    M, K = x.shape
    N, kh = packed.shape
    if M > 16:
        raise ValueError(f"gemm_int4_b32_smallm serves M <= 16 rows (got {M}); the grouped M-tile kernel serves more")
    if kh * 2 != K or tuple(scales.shape) != (N, K // 32):
        raise ValueError(f"layout mismatch: x K={K}, packed {tuple(packed.shape)}, scales {tuple(scales.shape)}")
    block_n, kc, sk = plan_smallm(N, K, block_n=block_n, kc=kc, sk=sk)
    if workspace is None:
        part, cnt = smallm_workspace(N, block_n=block_n, sk=sk, device=x.device)
    else:
        part, cnt = workspace
        if part.shape[0] < sk or part.shape[2] != N or cnt.numel() < triton.cdiv(N, block_n):
            raise ValueError("workspace does not fit this plan")
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    xc = x.contiguous()
    _gemm_int4_b32_smallm[(triton.cdiv(N, block_n), sk)](
        xc, packed, scales, part, cnt, out, M, N, K=K,
        BLOCK_M=16, BLOCK_N=block_n, KC=kc, SK=sk, num_warps=warps, num_stages=stages)
    return out
