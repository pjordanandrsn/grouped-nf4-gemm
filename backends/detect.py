# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Vendor-agnostic device detection + fingerprint (P1.2/P1.5).

Backends: 'cuda' (NVIDIA), 'hip' (AMD ROCm — torch aliases torch.cuda, so
is_hip distinguishes via torch.version.hip), 'xpu' (Intel), 'cpu' (interpreter
/ CI). Everything reads MEASURED facts (R6) — SM/CU count from device props,
PCIe link from lspci/sysfs where present — never assumes.
"""
from __future__ import annotations

import subprocess


def detect_backend() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "hip"          # ROCm torch reports a hip version and still exposes torch.cuda
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def warp_width(backend: str, arch: str = "") -> int:
    """Threads per warp/wavefront/sub-group. CDNA wavefront=64; RDNA/NVIDIA=32;
    Intel sub-group commonly 16/32 (kernel-selectable). arch lets CDNA (gfx9xx)
    be told apart from RDNA (gfx10xx/11xx/12xx)."""
    if backend == "hip":
        a = arch.lower()
        return 64 if (a.startswith("gfx9") or a.startswith("gfx94") or a.startswith("gfx95")) else 32
    if backend == "xpu":
        return 32           # sub-group; verify per device (K1) — not load-bearing yet
    return 32               # cuda / cpu


def sm_or_cu_count(device=None) -> int:
    """Compute-unit count, CPU-safe (the K6 fix). 0 when unknown (interpreter)."""
    try:
        import torch
    except Exception:
        return 0
    b = detect_backend()
    try:
        if b in ("cuda", "hip"):
            return torch.cuda.get_device_properties(device or 0).multi_processor_count
        if b == "xpu":
            return getattr(torch.xpu.get_device_properties(device or 0),
                           "gpu_subslice_count", 0) or 0
    except Exception:
        return 0
    return 0


def _measured_pcie() -> str:
    """Negotiated PCIe gen x width, MEASURED (R6). Best-effort; '' if unreadable."""
    try:
        out = subprocess.run(["lspci", "-vv"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return ""
    for ln in out.splitlines():
        if "LnkSta:" in ln and "Speed" in ln:
            return ln.strip().split("LnkSta:")[-1].strip()[:80]
    return ""


def device_fingerprint() -> dict:
    """Full environment fingerprint for a results-JSONL header (PROTOCOL-multiarch)."""
    fp = {"backend": detect_backend()}
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["cuda"] = getattr(torch.version, "cuda", None)
        fp["hip"] = getattr(torch.version, "hip", None)
        if fp["backend"] in ("cuda", "hip"):
            p = torch.cuda.get_device_properties(0)
            fp["device"] = p.name
            fp["cu_count"] = p.multi_processor_count
            fp["arch"] = getattr(p, "gcnArchName", getattr(p, "major", ""))
    except Exception as e:
        fp["torch_error"] = str(e)[:100]
    for mod in ("triton", "bitsandbytes"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = None
    fp["warp_width"] = warp_width(fp["backend"], str(fp.get("arch", "")))
    fp["pcie_linksta"] = _measured_pcie()
    return fp


if __name__ == "__main__":
    import json
    print(json.dumps(device_fingerprint(), indent=1))
