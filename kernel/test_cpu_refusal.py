"""Loud death is doctrine; doctrine gets a test.

`gemm_4bit_grouped` is CUDA-only. Called on CPU it must fail with a message
that (a) names the CUDA requirement and (b) points at `dequant_ref` as the
CPU-checkable path — not a raw Triton "0 active drivers" error. This pins the
taught guard added in 0.2.2.
"""
import re

import pytest
import torch

from nf4_grouped import gemm_4bit_grouped
from nf4_pack_ref import make_stack


def test_cpu_call_raises_taught_message(monkeypatch):
    # The guard is exempted under TRITON_INTERPRET=1 (interpreter mode runs the
    # kernel on CPU by design). A sibling module (test_mxfp4_interp) sets that
    # env at import, so a full-suite collection would leak it here — delete it
    # for this test so the refusal assertion is order-independent.
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    # valid shapes, wrong device — read the true signature, no arity fumbles:
    # (a_cat [T,K] bf16, B [E,N,K//2] u8, absmax [E,N,K//64] f32, sizes, expert_ids)
    E, N, K = 2, 128, 128
    B, absmax = make_stack(E, N, K, device="cpu")
    a_cat = torch.randn(2, K, dtype=torch.bfloat16)          # 2 tokens, one per group
    with pytest.raises(Exception) as ei:
        gemm_4bit_grouped(a_cat, B, absmax, [1, 1], [0, 1])
    msg = str(ei.value)
    assert re.search(r"cuda", msg, re.I), f"message must name CUDA: {msg!r}"
    assert "dequant_ref" in msg, f"message must point at dequant_ref: {msg!r}"
