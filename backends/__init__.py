# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Backend detection + per-arch config (Phase 1 P1.2/P1.5 scaffolding).

Single kernel source; this package centralizes the parts that MUST differ per
vendor — device detection, SM/CU count, warp width, and the autotune search
space — so `nf4_grouped.py` never grows a `torch.cuda`-only assumption again
(the K6 hazard the CI already caught). Nothing here claims a port works; it is
the seam a port fills. Per-arch search spaces are `port target` placeholders
until a confirmatory tunes them.
"""
from .detect import detect_backend, device_fingerprint, sm_or_cu_count, warp_width
from .config import decode_search_space, prefill_search_space

__all__ = [
    "detect_backend", "device_fingerprint", "sm_or_cu_count", "warp_width",
    "decode_search_space", "prefill_search_space",
]
