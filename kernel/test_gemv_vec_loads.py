# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K2 tripwire: the vectorized dual-nibble mainloop must be BITWISE-
equal to the legacy path at any fixed config. Runs the comparison in a
SUBPROCESS with TRITON_INTERPRET=1 in a fresh interpreter, so it
always executes regardless of the surrounding session's triton mode
(setting the env in-process is too late once triton imported compiled
-- hit live on the K2 box; and gating on a preset env made the test
never run anywhere -- Bugbot gnf4#243). Interp mode guards the
ALGEBRA; the on-box CUDA bitwise assert in bench/k2_vecnib_bench.py
is the codegen gate."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton", reason="interp-mode kernels need triton")

_INNER = """
import os
import torch
import nf4_grouped

torch.manual_seed(9)
E, N, K, T = 4, 128, 256, 4
packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
absmax = torch.rand(E, N, K // 64) + 0.5
a = torch.randn(T, K, dtype=torch.bfloat16)
eids = torch.arange(E, dtype=torch.int32)[:T]
sizes = [1] * T
for sk in (1, 4):
    for env in (None, "GNF4_GEMV_VEC_LOADS", "GNF4_GEMV_WIDE_LOADS"):
        os.environ.pop("GNF4_GEMV_VEC_LOADS", None)
        os.environ.pop("GNF4_GEMV_WIDE_LOADS", None)
        if env is None:
            legacy = nf4_grouped.gemm_4bit_grouped(
                a, packed, absmax, sizes, eids, decode_config=(64, 2),
                split_k=sk)
        else:
            os.environ[env] = "1"
            other = nf4_grouped.gemm_4bit_grouped(
                a, packed, absmax, sizes, eids, decode_config=(64, 2),
                split_k=sk)
            assert torch.equal(legacy, other), f"mismatch {env} sk={sk}"
print("BITWISE-OK")
"""


def test_vec_loads_bitwise_vs_legacy_interp_subprocess():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env.pop("GNF4_GEMV_VEC_LOADS", None)
    r = subprocess.run(
        [sys.executable, "-c", _INNER], env=env, capture_output=True,
        text=True, cwd=str(Path(__file__).resolve().parent), timeout=600)
    assert r.returncode == 0 and "BITWISE-OK" in r.stdout, (
        r.stdout[-800:] + r.stderr[-800:])
