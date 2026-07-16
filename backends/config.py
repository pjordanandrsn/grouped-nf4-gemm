# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-backend autotune search spaces (P1.2).

`port target` placeholders — the CUDA space is the shipped, confirmed one; the
hip/xpu spaces are STARTING points a confirmatory must actually tune (R3: not
"supported" until measured). They differ where the K1/K4 hazards bite: warp
counts scale to wavefront width on CDNA; tile heights shrink where LDS is
tighter. Consumed by the sweep harness, never by the shipped kernel path.
"""

# (BLOCK_N, num_warps) for the decode gemv; (BLOCK_M, BLOCK_N, warps, stages)
# for the M-tile. CUDA values are the v5/v6 confirmed constants.
_CUDA = {
    "decode": [(64, 2), (64, 4), (128, 4)],
    "prefill": [(64, 128, 4, 3), (128, 128, 4, 3), (128, 128, 8, 3)],
}

# CDNA: wavefront 64 -> a "warp" is 2x the threads, so halve warp counts as the
# starting guess; LDS ~64KB -> keep BLOCK_M<=128. STARTING POINT, unconfirmed.
_HIP_CDNA = {
    "decode": [(64, 1), (64, 2), (128, 2)],
    "prefill": [(64, 128, 2, 2), (128, 128, 2, 2), (128, 64, 4, 2)],
}
# RDNA: wavefront 32 like NVIDIA -> reuse the CUDA space as the starting guess.
_HIP_RDNA = dict(_CUDA)
# XPU: sub-group 16/32, distinct occupancy model -> conservative starting space.
_XPU = {
    "decode": [(64, 2), (128, 2)],
    "prefill": [(64, 64, 4, 2), (128, 128, 4, 2)],
}


def _space(backend: str, arch: str, key: str):
    if backend == "hip":
        a = arch.lower()
        table = _HIP_CDNA if (a.startswith("gfx9")) else _HIP_RDNA
    elif backend == "xpu":
        table = _XPU
    else:
        table = _CUDA
    return list(table[key])


def decode_search_space(backend: str, arch: str = ""):
    return _space(backend, arch, "decode")


def prefill_search_space(backend: str, arch: str = ""):
    return _space(backend, arch, "prefill")
