# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Grouped W4A16 GEMM over fused NF4 expert stacks — dequant inside the mainloop.

Computes, in ONE launch, ``out[t, :] = a[t, :] @ dequant_nf4(B[e(t)]).T`` for
tokens grouped by expert (KERNEL_CONTRACT.md: `bitsandbytes::gemm_4bit_grouped`
= the #1949 conventions + (group_offsets, expert_ids) + expert-major B/absmax).
The bf16 weight is never materialized in global memory: packed nibbles are
LUT-decoded to fp32 in registers, scaled by the fp32 blockwise absmax, and fed
to tensor-core ``tl.dot`` (TF32 on sm_86: 10-bit-mantissa inputs — *less*
rounding than the dequant path's bf16 materialization, which is the P-fid
mechanism) with fp32 accumulation and a single bf16 downcast at the epilogue.

Jagged grouping: the host expands groups into fixed-size M-tiles and passes
three small int32 descriptor arrays (tile→row0, tile→valid-rows, tile→expert);
the grid is (m_tiles, N/BLOCK_N). Empty groups never reach the kernel (the
caller drops them — a grouped GEMM never launches a 0-row tile). BLOCK_K == the
quant blocksize (64), so each (n, k-step) needs exactly one absmax scalar.

v1 scope per the contract: plain fp32 absmax (nested/`compress_statistics`
states are de-nested on the host at repack), no bias, nf4 only.
"""

from __future__ import annotations

import os

import torch
# ``triton`` is a Linux-only dependency (pyproject marks it
# ``platform_system == 'Linux'``), so a supported macOS install has none and a
# bare ``import triton`` here made the whole module unimportable there. The shim
# binds the real thing when it exists — this file is unchanged below in that
# case — and otherwise lets the kernels still DEFINE while a launch raises.
from _triton_shim import tl, triton  # noqa: F401  (re-exported names)

#: ``tl.gather`` arrived in triton 3.3. Bind it ONCE here rather than naming it
#: inside a kernel, because triton's JIT walks the whole kernel AST to build its
#: cache key and calls ``getattr`` on every attribute it sees — including inside
#: a ``tl.constexpr`` branch that codegen would prune. So merely *mentioning*
#: ``tl.gather`` in the source raises AttributeError on triton < 3.3, even when
#: the variant that uses it was never selected. That made the whole fused path
#: unusable on triton 3.2, which is what torch 2.6 pins. Resolving the name to
#: None here keeps the walker happy; the pruned branch never calls it.
_TL_GATHER = getattr(tl, "gather", None)
HAS_TL_GATHER = _TL_GATHER is not None
#: Escape hatch for debugging the v5 loop; never for real results.
_ALLOW_UNVERIFIED_V5 = os.environ.get("GNF4_ALLOW_UNVERIFIED_V5") == "1"

# The NF4 codebook (QLoRA appendix / bitsandbytes source). Code 7 decodes to
# exactly 0.0 — the zero-decode byte 0x77 the e4b mask fix relies on. The
# property suite asserts EXACT agreement (values and nibble order) against the
# installed bitsandbytes' dequantize_4bit, so a drift there fails loudly.
NF4_LUT = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
BLOCKSIZE = 64  # quant blocksize; K % 64 == 0 enforced (locked e4b design)


def repack_from_bnb(packed_list, states, N: int, K: int):
    """bnb per-expert quantize_4bit output -> the contract's expert-major tensors.

    Returns ``B [E, N, K//2] uint8`` and ``absmax [E, N, K//64] fp32``. bnb's
    packed tensor is the row-major flat [N*K/2, 1]; its absmax is flat
    [N*K/64] over the same row-major order, so both reshape cleanly when
    K % 64 == 0. Nested (compress_statistics) states are de-nested here —
    v1 of the kernel takes plain fp32 absmax per the contract."""
    assert K % BLOCKSIZE == 0, f"K={K} not a multiple of blocksize {BLOCKSIZE}"
    E = len(packed_list)
    dev = packed_list[0].device
    B = torch.empty(E, N, K // 2, dtype=torch.uint8, device=dev)
    A = torch.empty(E, N, K // BLOCKSIZE, dtype=torch.float32, device=dev)
    for e in range(E):
        B[e] = packed_list[e].reshape(N, K // 2)
        st = states[e]
        am = st.absmax
        if getattr(st, "nested", False):
            from bitsandbytes import functional as F

            am = F.dequantize_blockwise(st.absmax, st.state2) + st.offset
        A[e] = am.to(torch.float32).reshape(N, K // BLOCKSIZE)
    return B, A


class _PinnedIndexArena:
    """A persistent pinned staging arena for the small int32 index tensors the
    grouped kernels need (tile tables, expert ids, LoRA group sizes).

    WHY AN ARENA AND NOT JUST `.pin_memory()` PER CALL. Measured on an RTX A2000
    (torch 2.8.0+cu128), each construct captured in its own process against a
    trivial kernel:

        torch.tensor(list, device='cuda')          FAIL
        torch.tensor(list).pin_memory().to(cuda)   FAIL   <- pinned, still fails
        <pre-allocated pinned>.to(cuda, nb=True)   OK
        dev.copy_(<pre-allocated pinned>, nb=True) OK
        dev.copy_(<pre-allocated pinned>, nb=False) FAIL  <- nb=True is required
        torch.arange(..., device='cuda')           OK
        .pin_memory() with the result discarded    OK

    So "use pinned memory" is not the fix on its own: pinned memory ALLOCATED
    INSIDE the capture region still fails. The staging buffer has to already
    exist when capture starts, which is what this arena is for.

    SLICES TAKEN DURING A CAPTURE ARE PERMANENT: `off` is monotone and never
    rewinds. A captured graph's H2D nodes RE-READ their pinned host slices on
    EVERY replay, for as long as the graph lives -- so reusing those bytes for
    anything else would silently corrupt later replays. (The pre-0.13.1 design
    rewound on an event fence once transfers completed; that fence proves the
    COPY finished, not that no graph still reads the bytes -- a latent replay
    hazard, found while fixing the Bugbot report on PR #88, closed by
    permanence.) Since 0.13.1 `take` runs only under capture, so everything
    ever taken is capture-owned and permanence costs exactly the bytes the
    process's captures consumed: with the 1 Mi-int default, hundreds of
    captured whole-model steps.

    Growth keeps the old arena alive rather than freeing it -- retired buffers
    may still be read by live graphs.

    THE ARENA MUST NOT GROW DURING A CAPTURE, and that is why the default is
    generous. Inside a capture nothing completes, so the bump pointer never
    rewinds and one capture consumes the SUM of every call in it -- a whole-model
    step is hundreds of calls, not the handful a single-projection cell makes.
    Growing would allocate pinned memory inside the region, which is measured to
    break capture (the `pin_inside_then_to` row above). So growth is refused
    while capturing, with a named error, rather than producing the opaque
    "previous error during capture" this whole change exists to eliminate.

    The default holds 1 Mi int32 = 4 MiB of pinned host memory, allocated once
    per device on first use. That is enough for a very large model step; raise it
    with ``GNF4_PIN_ARENA_INTS`` if a capture ever reports exhaustion.
    """

    DEFAULT_INTS = int(os.environ.get("GNF4_PIN_ARENA_INTS", 1 << 20))

    def __init__(self, device, n: int | None = None):
        self.device = device
        self.host = torch.empty(n or self.DEFAULT_INTS, dtype=torch.int32,
                                pin_memory=True)
        self.off = 0
        self.evt = torch.cuda.Event()
        self._retired: list = []

    def _capturing(self) -> bool:
        try:
            return torch.cuda.is_current_stream_capturing()
        except Exception:
            return False

    def take(self, n: int):
        """A pinned int32 host view of length ``n``, permanently owned by the
        capture that takes it. Never rewinds -- see the class docstring."""
        capturing = self._capturing()
        if self.off + n > self.host.numel():
            if capturing:
                raise RuntimeError(
                    "grouped-nf4: the pinned index arena (%d int32, %d already "
                    "owned by this process's captures) cannot hold %d more ints, "
                    "and growing it would allocate pinned memory inside the "
                    "capture region, which is not capturable — so this refuses "
                    "instead of failing later with an opaque 'previous error "
                    "during capture'. Slices taken under capture are permanent "
                    "(replays re-read them), so set GNF4_PIN_ARENA_INTS to at "
                    "least %d (int32: %.1f MiB pinned) and re-run, or call "
                    "reserve() outside capture."
                    % (self.host.numel(), self.off, n, 2 * (self.off + n),
                       2 * (self.off + n) * 4 / 2**20))
            self._grow(self.off + n)
        view = self.host[self.off:self.off + n]
        self.off += n
        return view, capturing

    def _grow(self, need: int):
        """Replace the staging buffer with a larger one, OUTSIDE capture only.
        The old buffer is retired, never freed eagerly: live graphs may still
        read slices of it on replay, and `off` keeps counting in the new buffer
        from where the old one left off (owned bytes stay owned)."""
        self._retired.append(self.host)
        new_n = max(2 * self.host.numel(), need)
        fresh = torch.empty(new_n, dtype=torch.int32, pin_memory=True)
        # Owned prefix carries over so pre-capture reservation composes with
        # slices already owned by earlier captures in this process.
        fresh[: self.off] = self.host[: self.off]
        self.host = fresh

    def reserve(self, n_more: int):
        """Ensure ``n_more`` ints fit above the current watermark. Call OUTSIDE
        capture (raises inside one); this is how a caller sizes the arena for an
        upcoming capture instead of trusting the default."""
        if self._capturing():
            raise RuntimeError("reserve() inside a capture would allocate "
                               "pinned memory in the region; call it before "
                               "capturing.")
        if self.off + n_more > self.host.numel():
            self._grow(self.off + n_more)

    def mark(self):
        """Vestigial since permanence (nothing rewinds, so nothing waits on the
        fence). Kept as a no-op-shaped hook so call sites need not churn; never
        records inside a capture, where an event record becomes a graph node."""
        if not self._capturing():
            self.evt.record()


_ARENAS: dict = {}


def _arena(device) -> _PinnedIndexArena:
    key = str(device)
    if key not in _ARENAS:
        _ARENAS[key] = _PinnedIndexArena(device)
    return _ARENAS[key]


def to_device_i32(seqs, device):
    """Small host-side integer sequences to device, in one transfer whose KIND
    depends on whether the stream is capturing. Returns one int32 device view
    per input, in order — identical values on every path.

    UNDER CAPTURE: one pinned, async transfer staged from the persistent arena
    (a pageable build syncs, and a sync invalidates the capture). OTHERWISE:
    the plain pageable build the pre-capturability code used — measured 1.8%
    cheaper at the median on the host-bound e2e step (§11), because outside a
    capture the sync costs nothing while the extra host work does.

    Why this exists, and why it is batched
    --------------------------------------
    ``torch.tensor(list, device='cuda')`` builds a PAGEABLE host tensor and
    copies it with ``cudaMemcpyAsync`` followed by ``cudaStreamSynchronize``.
    That sync is illegal inside a CUDA graph capture, and it is why the fused
    training path could not be captured while the dequant-on-forward baseline
    could -- the asymmetry recorded in
    ``bench/phase1/results/dequant_forward/FINDING-host-bound-small-batch.md``.
    A copy whose source is PINNED needs no such sync, so it is capturable, and it
    is also the cheaper transfer.

    The bisection (``bench/phase1/probe_capture_bisect.py``, RTX A2000, each
    attempt in its own process) found FIVE such sites in the fused training step,
    only two of which had been named as candidates. ``build_group_tiles`` alone
    issued three per call and is called twice per step -- forward M-tile path and
    backward dgrad -- so a single fused training step made **eight** separate
    syncing transfers of a few hundred bytes each. Batching makes it one per call
    site.

    What this does NOT buy
    ----------------------
    A REPLAYED graph re-reads the staging buffer, so the routing metadata baked
    into a capture is whatever it held at capture time. MoE routing changes every
    step, so a captured fused graph is not usable without a padding or bucketing
    scheme, which is a separate question with its own registration
    (``kernel/prereg_capturability_scope.json``). **Capturability is a
    precondition, not a speedup.**
    """
    lens = [len(s) for s in seqs]
    total = sum(lens)
    dev = torch.device(device)
    # The staging arena needs a CUDA context; the interpreter/CPU contract path
    # has none, and there is nothing to make capturable there either.
    cuda = dev.type == "cuda" and torch.cuda.is_available()
    if total == 0:
        return [torch.empty(0, dtype=torch.int32, device=dev) for _ in seqs]
    # CAPTURE-CONDITIONAL, and this conditional is the whole repair for a
    # measured regression (RESULTS-capturability.md §11). The pinned-arena
    # transfer exists to be legal INSIDE a CUDA graph capture; outside one it is
    # strictly extra host work — staging writes, event queries, arena
    # bookkeeping — bought to remove syncs that cost nothing while the GPU is
    # idle. Measured on a whole-machine 3060 Ti (instrument clean to 2.2%): the
    # unconditional arena path priced the host-bound e2e step 1.8% down at the
    # median. So: the arena serves capture, the pre-change pageable build serves
    # everything else, and the branch below is the pre-change path restored by
    # construction.
    capturing = False
    if cuda:
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = False
        # The arena must EXIST before capture starts — pinned allocation inside
        # a capture region is the measured-illegal construct this whole file is
        # about. Touching it here, on every CUDA call including the pageable
        # ones, guarantees any later capture finds it already allocated (the
        # capture discipline always runs warm-up steps uncaptured first).
        ar = _arena(dev)
        if capturing and ar.off + total > ar.host.numel():
            raise RuntimeError(
                "grouped-nf4: the pinned index arena (%d int32, %d owned by "
                "prior captures) cannot hold this call's %d ints, and growing "
                "inside a capture is not capturable. Set GNF4_PIN_ARENA_INTS "
                "to at least %d, or reserve() the arena before capturing."
                % (ar.host.numel(), ar.off, total, 2 * (ar.off + total)))
    if not capturing:
        # Not capturing (or not CUDA): the PRE-CHANGE transfer. A pageable
        # build syncs, and on the host-bound paths that reach here the sync is
        # free — the pipeline it would drain is idle.
        packed = torch.tensor([int(v) for s in seqs for v in s],
                              dtype=torch.int32, device=dev)
    else:
        host, _ = ar.take(total)
        off = 0
        for s, n in zip(seqs, lens):
            if n:
                # torch.as_tensor handles both host sequences and host TENSORS
                # (a CPU tensor is host data and takes this path too — Bugbot,
                # PR #85). Either way the write into `host` is a CPU->CPU copy
                # into pinned memory, no CUDA call, so it stays capture-legal.
                host[off:off + n] = torch.as_tensor(s, dtype=torch.int32).reshape(-1)
            off += n
        # non_blocking is REQUIRED, not an optimisation: the blocking form is
        # `cudaMemcpyAsync` + `cudaStreamSynchronize`, and the sync is what is
        # illegal under capture (measured — see _PinnedIndexArena).
        packed = host.to(dev, non_blocking=True)
        ar.mark()
    out, off = [], 0
    for n in lens:
        out.append(packed[off:off + n])
        off += n
    return out


def build_group_tiles(sizes, block_m: int, device):
    """Expand jagged group sizes into fixed M-tiles: (row0, valid_rows, group_idx).

    ``sizes`` is READ-ONLY here, and `int(m)` is what keeps it that way. When
    `sizes` is a tensor, `enumerate` yields 0-dim VIEWS into it, so a bare
    `left = m` followed by `left -= take` subtracts *in place* and zeroes the
    caller's tensor. Every caller until now passed a Python list (where `m` is
    an int and `-=` rebinds) or called once and discarded the tensor, so the
    aliasing never surfaced; the first caller to run two GEMMs off one `sizes`
    tensor — gate/up/down within a single MoE layer — saw `[0, 0]` on the
    second call.
    """
    t_row0, t_rows, t_group = [], [], []
    row = 0
    for g, m in enumerate(sizes):
        m = int(m)
        left = m
        while left > 0:
            take = min(block_m, left)
            t_row0.append(row + (m - left))
            t_rows.append(take)
            t_group.append(g)
            left -= take
        row += m
    # ONE pinned transfer for all three, not three pageable ones. Same values,
    # same dtype, same shapes -- see to_device_i32 for why the transfer kind
    # matters.
    a, b, c = to_device_i32((t_row0, t_rows, t_group), device)
    return a, b, c


@triton.jit
def _gemm_nf4_grouped(
    a_ptr,
    b_ptr,
    amax_ptr,
    out_ptr,
    lut_ptr,
    t_row0_ptr,
    t_rows_ptr,
    t_group_ptr,
    expert_ids_ptr,
    K,
    N,
    stride_be,
    stride_bn,  # B strides (bytes dim contiguous)
    stride_ae,
    stride_an,  # absmax strides (block dim contiguous)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUPS: tl.constexpr,   # quant groups per K-step (BLOCK_K // 64)
    VARIANT: tl.constexpr,  # 0 = v5 mainloop; 1 = register-LUT tl.gather;
                            # 3 = OPT-IN bf16 MMA (documented looser P-fid)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    grp = tl.load(t_group_ptr + pid_m)
    eid = tl.load(expert_ids_ptr + grp)
    # int64 BEFORE any stride product: eid * stride_be overflows signed int32
    # the moment the packed stack passes 2^31 bytes — measured exactly at the
    # boundary (256 x 8 MiB passes, 257 faults; 128 x 16 MiB predicted, hit).
    eid = eid.to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < rows
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + (row0 + offs_m)[:, None] * K
    if VARIANT == 1:
        lut_reg = tl.load(lut_ptr + tl.arange(0, 16))  # codebook in registers

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(
            tl.int32
        )
        # bnb packs element 2j into the HIGH nibble, 2j+1 into the LOW nibble
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        if VARIANT == 1:
            # register-resident codebook: shuffle-gather, no per-element L1 LDG
            w = tl.reshape(
                _TL_GATHER(lut_reg, tl.reshape(nib, [BLOCK_N * BLOCK_K]), 0),
                [BLOCK_N, BLOCK_K],
            )
        else:
            w = tl.load(lut_ptr + nib)  # [BN, BK] fp32 codebook gather
        g0 = k0 // 64
        if GROUPS == 1:
            am = tl.load(
                amax_ptr + eid * stride_ae + offs_n * stride_an + g0,
                mask=n_mask,
                other=0.0,
            )
            scale = am[:, None]
        else:  # two quant groups per K-step (wrapper guarantees K % BLOCK_K == 0)
            am0 = tl.load(
                amax_ptr + eid * stride_ae + offs_n * stride_an + g0,
                mask=n_mask,
                other=0.0,
            )
            am1 = tl.load(
                amax_ptr + eid * stride_ae + offs_n * stride_an + (g0 + 1),
                mask=n_mask,
                other=0.0,
            )
            scale = tl.where(offs_k[None, :] < 64, am0[:, None], am1[:, None])
        if VARIANT == 3:
            # OPT-IN bf16 MMA: weight rounding matches the dequant baseline
            # (P-fid parity, not the fp32/TF32 edge); full-rate HMMA on sm_86.
            wb = (w * scale).to(tl.bfloat16)
            a = tl.load(a_base + kk[None, :], mask=m_mask[:, None], other=0.0)
            acc += tl.dot(a, tl.trans(wb))
        else:
            w = w * scale
            a = tl.load(a_base + kk[None, :], mask=m_mask[:, None], other=0.0).to(
                tl.float32
            )
            acc += tl.dot(a, tl.trans(w))  # TF32 tensor cores on sm_86, fp32 acc

    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemv_nf4_grouped(
    a_ptr,
    b_ptr,
    amax_ptr,
    out_ptr,
    lut_ptr,
    expert_ids_ptr,
    K,
    N,
    stride_be,
    stride_bn,
    stride_ae,
    stride_an,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode-specialized path: one token per group (M==1 everywhere), so a
    tensor-core M-tile would waste 15/16 of its lanes. Straight reduction:
    program (g, n-tile) accumulates out[g, n] = sum_k a[g,k] * w[n,k]."""
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(expert_ids_ptr + g)
    # int64 before the stride product — see the boundary note in the M-tile
    # kernel; same signed-int32 overflow, same fix.
    eid = eid.to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(
            tl.int32
        )
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.load(lut_ptr + nib)
        am = tl.load(
            amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // BLOCK_K),
            mask=n_mask,
            other=0.0,
        )
        a = tl.load(a_base + kk).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1) * am

    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _gemv_nf4_grouped_splitk(
    a_ptr,
    b_ptr,
    amax_ptr,
    ws_ptr,
    lut_ptr,
    expert_ids_ptr,
    K,
    N,
    T,
    KBLOCKS_PER_SPLIT,
    stride_be,
    stride_bn,
    stride_ae,
    stride_an,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split-K variant of the decode reduction for occupancy-starved grids
    (few groups x few n-tiles, e.g. top_k=1): program (g, n-tile, k-split)
    accumulates a PARTIAL fp32 sum over its span of whole absmax blocks into
    ``ws[k_split, g, n]``; the host reduces ``ws.sum(0)`` (deterministic
    two-pass, no atomics) and downcasts once. Decode math is identical to
    ``_gemv_nf4_grouped``. A split whose span starts past K stores zeros."""
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    eid = tl.load(expert_ids_ptr + g)
    # int64 before the stride product — see the boundary note in the M-tile
    # kernel; same signed-int32 overflow, same fix.
    eid = eid.to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K

    k_lo = pid_k * KBLOCKS_PER_SPLIT * BLOCK_K
    k_hi = tl.minimum(k_lo + KBLOCKS_PER_SPLIT * BLOCK_K, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(
            tl.int32
        )
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.load(lut_ptr + nib)
        am = tl.load(
            amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // BLOCK_K),
            mask=n_mask,
            other=0.0,
        )
        a = tl.load(a_base + kk).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1) * am

    tl.store(ws_ptr + pid_k * (T * N) + g * N + offs_n, acc, mask=n_mask)


_LUT_CACHE: dict = {}
_SM_CACHE: dict = {}


def _sm_count(device) -> int:
    key = str(device)
    if key not in _SM_CACHE:
        # CPU / interpreter mode (correctness testing, no GPU) can't query cuda;
        # the count only feeds split-K planning, so a nominal value is safe there.
        # On a real cuda device the true count is still used (no perf-path change).
        if torch.cuda.is_available() and "cuda" in key:
            _SM_CACHE[key] = torch.cuda.get_device_properties(device).multi_processor_count
        else:
            _SM_CACHE[key] = 64
    return _SM_CACHE[key]


# v4 dispatch constants (each carries its measured basis; see the prereg/results
# docs for the runs behind the numbers):
# - DECODE_MIN_FUSED_BYTES: below this per-call weight+absmax traffic the fused
#   launch loses OUTRIGHT to the dequant path (v3 blind: Switch-Base cells at
#   1.3-2.7 MB ran 0.24-0.35x speed and 4-7x energy on both devices, while
#   granite down at 3.5 MB kept a 1.13-1.97x win). Product integrations should
#   route below-floor calls to the dequant path via decode_dispatch().
# - SPLITK_MIN_BLOCKS: a split must own at least this many absmax blocks —
#   v3 blind showed the starvation-only trigger splitting 12-block cells hurt
#   (Switch gu paired 0.655 on the A5000) while >=32-block splits helped
#   (Scout down 1.46x, Hunyuan down 1.18x paired).
DECODE_MIN_FUSED_BYTES = 3_000_000
SPLITK_MIN_BLOCKS = 32


def _decode_plan(N: int, K: int, T: int, sm_count: int):
    """Decode launch plan: (BLOCK_N, num_warps, split_k).

    Config is the universal constant (64, 2) — dense 2-device (N, K, T)
    sweeps put it at median regret 1.000 on both grids
    (bench/phase2/sweeps/); the v3 confirmatory showed the v2-era A2000
    preference for 128/4 did not reproduce (config deltas on the 26-SM card
    are run-context noise), so the SM-conditional branch is reverted.

    Split-K engages only for truly starved grids (programs < 2*SM — census
    cells never split) AND only when each split owns >= SPLITK_MIN_BLOCKS
    absmax blocks (v3: splitting tiny-K cells hurt). fp32 partials are
    host-reduced; power-of-2, capped at 8."""
    # PREREG-m1-decode-config (K1) ablation override: dead unless
    # exported. Shape-keyed so an A/B arm can select PER-SHAPE winners
    # (gate_up and down want different configs) without code edits:
    #   GNF4_DECODE_PLAN="1536,2048=64,4,8;2048,768=128,2,4"
    # maps (N, K) -> (bn, warps, split_k); shapes not listed fall
    # through to the plan below. Unset => byte-identical behavior.
    plan_env = os.environ.get("GNF4_DECODE_PLAN")
    if plan_env:
        for entry in plan_env.split(";"):
            shape, _, cfg = entry.partition("=")
            n_s, _, k_s = shape.partition(",")
            if int(n_s) == N and int(k_s) == K:
                bn_s, w_s, sk_s = cfg.split(",")
                return int(bn_s), int(w_s), int(sk_s)
    bn, warps = 64, 2
    programs = T * -(-N // bn)
    split_k = 1
    if programs < 2 * sm_count:
        want = -(-(4 * sm_count) // programs)
        while split_k < want and split_k < 8:
            split_k *= 2
        kblocks = max(K // BLOCKSIZE, 1)
        while split_k > 1 and kblocks // split_k < SPLITK_MIN_BLOCKS:
            split_k //= 2
    return bn, warps, split_k


def decode_dispatch(N: int, K: int, T: int, sm_count: int):
    """Product-layer path choice for one decode call: ``("dequant",)`` when
    the call is below the fused floor (tiny cells belong to the dequant
    path — v3 measured them losing outright), else
    ``("fused", BLOCK_N, num_warps, split_k)``.

    The op itself (gemm_4bit_grouped) always runs fused — an op that
    silently ran a different algorithm would be a contract violation — so
    integrations consult this helper and call the dequant path themselves
    for below-floor cells. The benchmark's ``fused_routed`` backend does
    exactly that."""
    traffic = T * N * (K // 2 + K // 16)  # packed nibbles + fp32 absmax bytes
    if traffic < DECODE_MIN_FUSED_BYTES:
        return ("dequant",)
    return ("fused", *_decode_plan(N, K, T, sm_count))


def _lut(device):
    key = str(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = torch.tensor(NF4_LUT, dtype=torch.float32, device=device)
    return _LUT_CACHE[key]


_SHARED_LIMIT: dict[int, int] = {}


def _device_shared_limit(dev) -> int:
    """Per-device shared-memory (LDS) cap in bytes, cached; 0 if unqueryable.

    NVIDIA SMs expose 100-228 KB; CDNA3 (MI300X, gfx942) exposes 64 KB. The
    prefill M-tile config is tuned for the NVIDIA budget, so on a smaller-LDS
    device it must be trimmed to fit (see the fit-down in ``gemm_4bit_grouped``).
    Returning 0 (unqueryable — no CUDA/HIP device, interpreter mode, or a
    driver that doesn't expose the property) makes the fit-down a no-op, so the
    CPU/TRITON_INTERPRET path is never perturbed. Any failure -> 0.
    """
    try:
        idx = getattr(dev, "index", None)
        if idx is None:
            if not torch.cuda.is_available():
                return 0
            idx = torch.cuda.current_device()
        if idx not in _SHARED_LIMIT:
            props = triton.runtime.driver.active.utils.get_device_properties(idx)
            _SHARED_LIMIT[idx] = int(props["max_shared_mem"])
        return _SHARED_LIMIT[idx]
    except Exception:
        return 0


def _prefill_block_m(max_rows: int) -> int:
    """Group-size-keyed M-tile height (sweep basis in the wrapper comment)."""
    if max_rows <= 16:
        return 16
    if max_rows <= 32:
        return 32
    if max_rows <= 64:
        return 64
    return 128


def gemm_4bit_grouped(
    a_cat,
    B,
    absmax,
    sizes,
    expert_ids,
    block_m: int | None = None,
    decode_config: tuple | None = None,
    split_k: int | None = None,
    prefill_config: tuple | None = None,
    prefill_variant: int | None = None,
    prefill_groups: int = 1,
):
    """Single-launch grouped NF4 GEMM. ``a_cat [T,K]`` bf16/fp16 in group-sorted
    order, ``B [E,N,K//2]`` uint8, ``absmax [E,N,K//64]`` fp32, ``sizes`` the
    per-group token counts (all > 0), ``expert_ids [G]`` int32/list. Returns
    ``[T, N]`` bf16 in the same group order. ``decode_config`` overrides the
    decode path's (BLOCK_N, num_warps); ``split_k`` overrides the decode
    split-K factor (None = plan, 1 = off); ``prefill_config`` overrides the
    M-tile path's (BLOCK_N, num_warps, num_stages) — benchmark/ablation
    support only. ``prefill_variant``: None = auto (register-LUT mainloop
    when triton has ``tl.gather``, else the v5 loop), 0 = force v5 loop,
    1 = register-LUT tl.gather (the v6 default: fidelity-identical, kills
    the per-element L1 codebook gather), 3 = OPT-IN bf16 MMA (P-fid parity
    with the dequant baseline, not the fp32/TF32 edge — measured slower
    than variant 1 everywhere in the v6 matrix; see RESULTS-phase2-v1.1 and
    bench/phase2/v6_prefill_matrix.py). ``prefill_groups``: quant groups
    per K-step (2 = BLOCK_K 128: dead on sm_86 — SMEM blowout; kept for
    ablation only)."""
    E, N, _ = B.shape
    T, K = a_cat.shape
    assert sum(sizes) == T, (sum(sizes), T)
    dev = a_cat.device
    # CUDA-only in real use; TRITON_INTERPRET=1 runs the kernel on CPU tensors
    # (the interpreter-contract suite), so exempt that path from the guard.
    if dev.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError(
            f"gemm_4bit_grouped runs the fused Triton kernel and requires CUDA tensors "
            f"(got device '{dev.type}'). For a CPU-checkable decode of the same NF4 bytes, "
            f"use dequant_ref(packed, absmax, N, K) — the pure-torch reference the property "
            f"suite pins the kernel against."
        )
    # A CUDA tensor passes straight through; anything HOST — a list, or a CPU
    # tensor (Bugbot, PR #85: `torch.tensor([...])` with no device= used to work
    # via the per-element path and must keep working) — is converted ONCE here,
    # at the boundary, through a pinned transfer (to_device_i32). It used to be
    # a pageable `torch.tensor(list, device=dev)`, which syncs and is not
    # capturable; a CPU tensor handed to the launch below would not work at all.
    eids = (
        expert_ids
        if torch.is_tensor(expert_ids) and expert_ids.is_cuda
        else to_device_i32((expert_ids,), dev)[0]
    ).to(torch.int32)
    out = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
    if max(sizes) == 1:
        # decode: every group is one token; the reduction path skips the M-tile.
        # Launch plan (_decode_plan): SM-conditional constant + split-K for
        # starved grids. Each part carries its measured basis in the plan's
        # docstring; the ablation kwargs let a harness force either off.
        bn, warps, sk = _decode_plan(N, K, T, _sm_count(dev))
        if decode_config is not None:
            bn, warps = decode_config
        if split_k is not None:
            sk = split_k
        if sk <= 1:
            grid = (T, triton.cdiv(N, bn))
            _gemv_nf4_grouped[grid](
                a_cat,
                B,
                absmax,
                out,
                _lut(dev),
                eids,
                K,
                N,
                B.stride(0),
                B.stride(1),
                absmax.stride(0),
                absmax.stride(1),
                BLOCK_N=bn,
                BLOCK_K=BLOCKSIZE,
                num_warps=warps,
                num_stages=3,
            )
            return out
        kblocks = -(-K // BLOCKSIZE)
        span = -(-kblocks // sk)
        ws = torch.empty(sk, T, N, dtype=torch.float32, device=dev)
        grid = (T, triton.cdiv(N, bn), sk)
        _gemv_nf4_grouped_splitk[grid](
            a_cat,
            B,
            absmax,
            ws,
            _lut(dev),
            eids,
            K,
            N,
            T,
            span,
            B.stride(0),
            B.stride(1),
            absmax.stride(0),
            absmax.stride(1),
            BLOCK_N=bn,
            BLOCK_K=BLOCKSIZE,
            num_warps=warps,
            num_stages=3,
        )
        out.copy_(ws.sum(dim=0))  # fp32 partial reduce, single bf16 downcast
        return out
    if block_m is None:
        block_m = _prefill_block_m(max(sizes))
    if prefill_variant is None:
        prefill_variant = 1 if HAS_TL_GATHER else 0
    if prefill_config is not None:
        block_n, warps, stages = prefill_config
    elif prefill_variant == 1:
        # v6 register-LUT mainloop rule (bench/phase2/v6_prefill_matrix.py,
        # A5000): bn=128/w4/s3 with the group-size-keyed BLOCK_M is the
        # per-cell oracle on 6/8 census prefill cells, worst regret 1.034.
        # Under it the M-tile path runs 1.20-2.88x the dequant baseline on
        # every census cell except OLMoE gate_up (0.62x, the remaining
        # known loser; was 0.38x on the v5 loop).
        block_n = 128
        warps = 4
        stages = 3
    else:
        # v4 group-size-keyed rule for the v5 (VARIANT=0) loop
        # (bench/phase2/sweeps/v4_prefill_*.json): 128/128/w8/s3 for m >= 128
        # groups, 64-and-below groups want the narrower 64-row tile at w4/s2.
        # Rule regret vs per-cell oracle: worst 1.058, 13/16 cells at 1.00-1.02.
        block_n = 128
        warps = 8 if block_m >= 128 else 4
        stages = 3 if block_m >= 128 else 2
    if prefill_variant == 1 and not HAS_TL_GATHER:
        raise RuntimeError("prefill_variant=1 needs triton with tl.gather")
    if prefill_variant == 0 and not HAS_TL_GATHER and not _ALLOW_UNVERIFIED_V5:
        # The v5 loop is the automatic fallback on triton < 3.3, and it is
        # WRONG there. Measured on triton 3.2.0 against an fp32 ground truth
        # built from the pre-quantization weights: the NF4 baseline lands at
        # 1.7e-01 relative (ordinary 4-bit error) while this path lands at
        # 1.7e+00 — not a rounding difference, a different answer.
        #
        # It went unnoticed because CI runs on triton >= 3.3, where variant 1
        # is always selected, so nothing ever exercised this loop; and on older
        # triton the kernel failed to compile at all, so the wrong numbers were
        # masked by a crash. Refusing here keeps that property: a user on old
        # triton still cannot get silently wrong logits, but now they get a
        # sentence explaining why instead of an AttributeError from the JIT.
        raise RuntimeError(
            "grouped-nf4 prefill: this triton (%s) lacks tl.gather, and the v5 "
            "fallback loop is numerically WRONG here (measured 1.7e+00 relative "
            "vs fp32 truth, against 1.7e-01 for the NF4 reference). Upgrade to "
            "triton >= 3.3, or set GNF4_ALLOW_UNVERIFIED_V5=1 to run it anyway "
            "for debugging — do not use it to produce results."
            % getattr(triton, "__version__", "?"))
    block_k = BLOCKSIZE * prefill_groups
    if prefill_groups != 1:
        assert prefill_groups == 2 and K % block_k == 0, (prefill_groups, K)
    # --- fit the M-tile pipeline to the device's shared-memory (LDS) budget ---
    # The M-tile mainloop stages, per pipeline stage, the activation tile plus
    # (variant 0) a dequantized-B tile or (variant 1) the packed-B tile. The
    # tuned (bm=128, stages=3) config needs 3*(128*64*2 + 128*64*2)=98304 B for
    # variant 0 — fine on NVIDIA (100-228 KB LDS), but over CDNA3's 64 KB. Step
    # stages (down to 2) then block_m (down to 64) then stages (to 1) until the
    # estimate fits with ~8 KB headroom for the compiler's own scratch. No-op
    # where the config already fits (every NVIDIA cell) or the limit is
    # unqueryable. Only correctness-preserving knobs (tiling/pipelining) move.
    _smem_cap = _device_shared_limit(dev)
    if _smem_cap and prefill_config is None:
        _hr = 8192

        def _prefill_smem(bm: int, st: int) -> int:
            a = bm * block_k * 2
            b = block_n * block_k * 2 if prefill_variant == 0 else block_n * (block_k // 2)
            return st * (a + b)

        while stages > 2 and _prefill_smem(block_m, stages) > _smem_cap - _hr:
            stages -= 1
        while block_m > 64 and _prefill_smem(block_m, stages) > _smem_cap - _hr:
            block_m //= 2
        while stages > 1 and _prefill_smem(block_m, stages) > _smem_cap - _hr:
            stages -= 1
    t_row0, t_rows, t_group = build_group_tiles(sizes, block_m, dev)
    grid = (t_row0.numel(), triton.cdiv(N, block_n))
    _gemm_nf4_grouped[grid](
        a_cat,
        B,
        absmax,
        out,
        _lut(dev),
        t_row0,
        t_rows,
        t_group,
        eids,
        K,
        N,
        B.stride(0),
        B.stride(1),
        absmax.stride(0),
        absmax.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUPS=prefill_groups,
        VARIANT=prefill_variant,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def dequant_ref(packed_row_major: torch.Tensor, absmax: torch.Tensor, N: int, K: int):
    """Pure-torch reference decode (same LUT + nibble order as the kernel) —
    the property suite asserts this matches bnb's dequantize_4bit EXACTLY,
    which pins both the codebook values and the high-nibble-first order.
    Runs on CPU (no CUDA/Triton), so it is the checkable oracle for the kernel.

    Example:
        >>> from nf4_pack_ref import quantize_pack_nf4
        >>> from nf4_grouped import dequant_ref
        >>> packed, absmax = quantize_pack_nf4(torch.randn(128, 256))
        >>> w = dequant_ref(packed, absmax, 128, 256)      # [128, 256] fp32
    """
    lut = _lut(packed_row_major.device)
    flat = packed_row_major.reshape(-1).to(torch.int32)
    hi = (flat >> 4) & 0xF
    lo = flat & 0xF
    codes = torch.stack([hi, lo], dim=1).reshape(-1)  # element 2j = high nibble
    vals = lut[codes]
    am = absmax.to(torch.float32).reshape(-1).repeat_interleave(BLOCKSIZE)
    return (vals * am).reshape(N, K)


# ---------------------------------------------------------------------------
# dgrad: the backward of `gemm_4bit_grouped`.
#
# `gemm_4bit_grouped` computes out = a @ dequant(B).T, contracting over K. Its
# backward needs grad_a = grad_out @ dequant(B), contracting over N — identical
# FLOPs, identical packed-byte traffic, different axis. Until this existed there
# was no backward kernel at all: `nf4_qlora.FusedGroupedNf4.backward` looped the
# active experts in Python, one `dequant_ref` + matmul each. At 256 experts over
# 40 layers that is ~10k decode+matmul pairs per step, and it measured 78-84% of
# an experts4bit-qlora training step.
#
# The transposed contraction costs nothing structurally. The weight tile is
# [BLOCK_N, BLOCK_K] here exactly as in the forward, from the same pointer
# arithmetic; what changes is that the mainloop walks N instead of K, `a` becomes
# grad_out (strided by N), the output tile is [BLOCK_M, BLOCK_K], and `tl.dot`
# needs no transpose. With BLOCK_K dividing 64 the whole output tile sits inside
# one quant group, so the absmax column index is a compile-time-constant scalar
# and the load is the forward's [BLOCK_N] read rather than a per-element gather.
# ---------------------------------------------------------------------------


@triton.jit
def _dgrad_nf4_grouped(
    g_ptr,
    b_ptr,
    amax_ptr,
    out_ptr,
    lut_ptr,
    t_row0_ptr,
    t_rows_ptr,
    t_group_ptr,
    expert_ids_ptr,
    K,
    N,
    stride_be,
    stride_bn,
    stride_ae,
    stride_an,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    grp = tl.load(t_group_ptr + pid_m)
    eid = tl.load(expert_ids_ptr + grp)
    # int64 BEFORE any stride product: eid * stride_be overflows signed int32
    # the moment the packed stack passes 2^31 bytes — measured exactly at the
    # boundary (256 x 8 MiB passes, 257 faults; 128 x 16 MiB predicted, hit).
    eid = eid.to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < rows
    k_mask = offs_k < K

    lut_reg = tl.load(lut_ptr + tl.arange(0, 16))  # codebook in registers
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    g_base = g_ptr + (row0 + offs_m)[:, None] * N
    # BLOCK_K divides 64, so every element of this output tile shares one quant
    # group and the absmax column is a scalar — not a gather. The 64 is the
    # literal `BLOCKSIZE`, spelled out because a @triton.jit body cannot read a
    # module global that is not a tl.constexpr; the forward mainloop does the same.
    g0 = (pid_k * BLOCK_K) // 64

    for n0 in range(0, N, BLOCK_N):
        nn = n0 + offs_n
        n_mask = nn < N
        bytes_ = tl.load(
            b_ptr + eid * stride_be + nn[:, None] * stride_bn + (offs_k[None, :] // 2),
            mask=n_mask[:, None] & k_mask[None, :],
            other=0,
        ).to(tl.int32)
        # bnb packs element 2j into the HIGH nibble, 2j+1 into the LOW nibble
        nib = tl.where((offs_k[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.reshape(
            tl.gather(lut_reg, tl.reshape(nib, [BLOCK_N * BLOCK_K]), 0),
            [BLOCK_N, BLOCK_K],
        )
        am = tl.load(amax_ptr + eid * stride_ae + nn * stride_an + g0, mask=n_mask, other=0.0)
        w = w * am[:, None]
        g = tl.load(
            g_base + nn[None, :], mask=m_mask[:, None] & n_mask[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.dot(g, w)  # [BM,BN] @ [BN,BK] — no transpose, unlike the forward

    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * K + offs_k[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & k_mask[None, :])


# Measured on an RTX A2000 by sweeping (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) over
# the E=256 gate_up shape (N=1536, K=512, T_cat=4096): this config ran 3.29 ms,
# 0.91x of the FORWARD kernel's time on the same problem — i.e. dgrad reaches the
# forward's ceiling rather than landing above it. Every config in the sweep
# produced bit-identical output, so this is a speed choice, not a fidelity one.
_DGRAD_DEFAULT = (32, 64, 64, 2)  # BLOCK_M, BLOCK_N, BLOCK_K, num_warps


def dgrad_eligible(grad_out, B, absmax, block_k: int = 64):
    """None if this problem can take the dgrad kernel, else the reason it cannot.

    Separate from the launch so callers can decide *before* committing — the
    training path falls back to its per-expert loop rather than failing.
    """
    if grad_out.dtype != torch.bfloat16:
        # The kernel's epilogue stores bf16. fp16 callers would get a silent
        # dtype change, so they keep the reference path.
        return f"grad_out dtype {grad_out.dtype} is not bfloat16"
    if BLOCKSIZE % block_k:
        return f"BLOCK_K {block_k} does not divide the quant blocksize {BLOCKSIZE}"
    K = B.shape[2] * 2
    if K % block_k:
        return f"K {K} not divisible by BLOCK_K {block_k}"
    if absmax.shape[2] * BLOCKSIZE != K:
        return f"absmax blocks {absmax.shape[2]} do not tile K {K} at blocksize {BLOCKSIZE}"
    if B.numel() == 0:
        return "packed storage is empty (evicted?)"
    return None


def dgrad_4bit_grouped(grad_out, B, absmax, sizes, expert_ids, config=None):
    """``grad_a[t, :] = grad_out[t, :] @ dequant_nf4(B[e(t)])`` in one launch.

    The backward companion to :func:`gemm_4bit_grouped`, taking the same
    group-sorted layout: ``grad_out [T, N]`` bf16, ``B [E, N, K//2]`` uint8,
    ``absmax [E, N, K//64]`` fp32, ``sizes`` the per-group row counts,
    ``expert_ids [G]``. Returns ``[T, K]`` bf16 in the same group order.

    Materializes nothing — the decode happens in registers inside the GEMM,
    exactly as the forward does, so this preserves the "packed bytes are the only
    residency" property that a whole-stack dequantize would spend.

    ``config`` overrides ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps)``; see
    ``_DGRAD_DEFAULT`` for where the default came from. Benchmark support — the
    output does not depend on it.
    """
    E, N, half = B.shape
    K = half * 2
    T = grad_out.shape[0]
    assert sum(sizes) == T, (sum(sizes), T)
    dev = grad_out.device
    if dev.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError(
            f"dgrad_4bit_grouped runs the fused Triton kernel and requires CUDA tensors "
            f"(got device '{dev.type}'). The CPU-checkable equivalent is "
            f"grad_out @ dequant_ref(packed, absmax, N, K) per group."
        )
    block_m, block_n, block_k, warps = config or _DGRAD_DEFAULT
    why = dgrad_eligible(grad_out, B, absmax, block_k)
    if why is not None:
        raise ValueError(
            f"dgrad_4bit_grouped cannot run this problem: {why}. Callers that can "
            "fall back should check dgrad_eligible() first rather than catching this."
        )
    t_row0, t_rows, t_group = build_group_tiles(sizes, block_m, dev)
    # Same boundary conversion as the forward's: a CUDA tensor passes through,
    # anything host (list or CPU tensor) converts once through a pinned
    # transfer. `ctx` hands this whatever the caller supplied, unchanged.
    eids = (
        expert_ids
        if torch.is_tensor(expert_ids) and expert_ids.is_cuda
        else to_device_i32((expert_ids,), dev)[0]
    ).to(torch.int32)
    out = torch.empty(T, K, dtype=torch.bfloat16, device=dev)
    _dgrad_nf4_grouped[(t_row0.numel(), triton.cdiv(K, block_k))](
        grad_out.contiguous(),
        B,
        absmax,
        out,
        _lut(dev),
        t_row0,
        t_rows,
        t_group,
        eids,
        K,
        N,
        B.stride(0),
        B.stride(1),
        absmax.stride(0),
        absmax.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
    )
    return out
