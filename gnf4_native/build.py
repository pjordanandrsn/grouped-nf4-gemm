# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Compile-at-first-use loader for cpu_kernels.c.

Why runtime compilation and not a prebuilt extension: the wheel stays a
pure-python artifact installable everywhere (the pyproject comment block
exists to keep macOS/Windows installable), `-march=native` dispatch is a
runtime-CPU decision anyway, and the kernel contract requires the binary to
match the measured box. The cache key is (source sha256, compiler id,
flags), so editing the C source or changing boxes rebuilds automatically.

Failure is soft: `available()` returns False and the torch-facing wrappers
in kernel/cpu_grouped.py fall back to the exact (slow) reference path.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "cpu_kernels.c"
_LIB = None
_ERR = None


def _cache_dir() -> Path:
    root = os.environ.get("GNF4_NATIVE_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "gnf4-native",
    )
    return Path(root)


def _compiler():
    for cc in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cc and shutil.which(cc):
            return cc
    return None


def _build() -> tuple[ctypes.CDLL | None, str | None]:
    if not _SRC.exists():
        return None, f"source missing: {_SRC}"
    cc = _compiler()
    if cc is None:
        return None, "no C compiler on PATH"
    src = _SRC.read_bytes()
    # -ffp-contract=off is load-bearing: GCC's GNU-mode default contracts
    # mul+add into FMA, silently breaking the locked two-rounding summation
    # tree (caught by test_cpu_grouped exact-parity on first run). The
    # kernel is bandwidth-bound; the extra rounding op costs nothing.
    base_flags = ["-O3", "-march=native", "-ffp-contract=off", "-shared",
                  "-fPIC", "-lm"]
    for flags in (base_flags + ["-fopenmp"], base_flags):
        key = hashlib.sha256(
            src + cc.encode() + " ".join(flags).encode()
        ).hexdigest()[:16]
        out = _cache_dir() / f"cpu_kernels-{key}.so"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                dir=out.parent, suffix=".so", delete=False
            )
            tmp.close()
            r = subprocess.run(
                [cc, "-o", tmp.name, str(_SRC)] + flags,
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                os.unlink(tmp.name)
                err = r.stderr[-400:]
                continue_reason = err
                if "openmp" in err.lower() or "-fopenmp" in err:
                    continue          # retry without OpenMP
                return None, f"compile failed ({cc}): {continue_reason}"
            os.replace(tmp.name, out)   # atomic publish
        try:
            return ctypes.CDLL(str(out)), None
        except OSError as e:
            return None, f"dlopen failed: {e}"
    return None, "compile failed with and without -fopenmp"


def _typed(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    lib.gnf4_cpu_features.restype = c.c_int
    lib.gnf4_gemv_nf4_grouped.restype = c.c_int
    lib.gnf4_gemv_nf4_grouped.argtypes = [
        c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p,
        c.c_int, c.c_int64, c.c_int64, c.c_void_p, c.c_int,
    ]
    lib.gnf4_gemv_mxfp4_grouped.restype = c.c_int
    lib.gnf4_gemv_mxfp4_grouped.argtypes = lib.gnf4_gemv_nf4_grouped.argtypes
    lib.gnf4_route_epilogue_bf16.restype = None
    lib.gnf4_route_epilogue_bf16.argtypes = [
        c.c_void_p, c.c_int64, c.c_int64, c.c_int64, c.c_int, c.c_int,
        c.c_void_p, c.c_int64, c.c_void_p, c.c_int64,
    ]
    lib.gnf4_dense_gemv_f32.restype = None
    lib.gnf4_dense_gemv_f32.argtypes = [
        c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_int64, c.c_int64,
    ]
    lib.gnf4_pool_start.restype = c.c_int
    lib.gnf4_pool_start.argtypes = [c.c_int]
    lib.gnf4_pool_stop.restype = None
    lib.gnf4_pool_stop.argtypes = []
    lib.gnf4_pool_size.restype = c.c_int
    lib.gnf4_pool_size.argtypes = []
    return lib


def load() -> ctypes.CDLL:
    """The compiled library, building it on first call. Raises on failure —
    use `available()` to probe without raising."""
    global _LIB, _ERR
    if _LIB is not None:
        return _LIB
    if _ERR is not None:
        raise RuntimeError(f"gnf4_native unavailable: {_ERR}")
    lib, err = _build()
    if lib is None:
        _ERR = err
        raise RuntimeError(f"gnf4_native unavailable: {err}")
    _LIB = _typed(lib)
    return _LIB


def available() -> bool:
    try:
        load()
        return True
    except (RuntimeError, OSError):
        return False


def features() -> dict:
    """Decoded gnf4_cpu_features() bits (empty when unavailable)."""
    if not available():
        return {}
    f = load().gnf4_cpu_features()
    return {
        "avx2": bool(f & 1), "avx512f": bool(f & 2),
        "avx512vbmi": bool(f & 4), "avx512vnni": bool(f & 8),
        "compiled_avx512": bool(f & 16), "openmp": bool(f & 32),
    }
