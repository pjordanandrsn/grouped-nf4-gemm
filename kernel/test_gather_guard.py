# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""The prefill path must fail LOUDLY on a triton without ``tl.gather``.

Both grouped GEMMs used to do this::

    prefill_variant = 1 if hasattr(tl, "gather") else 0

which reads like a graceful fallback and is not one. ``_gemm_nf4_grouped``'s
*source* contains ``tl.gather`` inside ``if VARIANT == 1:``, and triton resolves
module attributes while walking the AST — even for a branch a ``constexpr``
makes dead. So selecting variant 0 does not avoid it: the launch dies with a
bare ``AttributeError: module 'triton.language' has no attribute 'gather'``.

Measured on the shipped code (finding #47):

    triton 3.4:  PREFILL OK
    triton 3.0:  PREFILL FAILED -- AttributeError ... no attribute 'gather'

``pyproject.toml`` declares ``triton>=3.4``, so this is out of declared support
and no correct install is affected. The defect was that three files advertised
a fallback that could not work. These tests keep it from coming back.

The source-contract tests read the files rather than importing, so they run on
CPU CI where no GPU or triton build is present -- which is exactly where a
refactor would reintroduce the pattern.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_KERNEL = Path(__file__).resolve().parent
_TARGETS = ("nf4_grouped.py", "mxfp4_grouped.py")


@pytest.mark.parametrize("fname", _TARGETS)
def test_no_silent_gather_fallback(fname):
    """The `1 if hasattr(...) else 0` auto-fallback must not return."""
    src = (_KERNEL / fname).read_text()
    assert 'hasattr(tl, "gather") else' not in src, (
        f"{fname} reintroduced the silent gather fallback. Selecting "
        "prefill_variant=0 does NOT make the kernel compile on a triton "
        "without tl.gather -- the source still contains it and triton "
        "resolves it during the AST walk. Raise a clear error instead."
    )


@pytest.mark.parametrize("fname", _TARGETS)
def test_prefill_guard_present_and_actionable(fname):
    """A bare `hasattr` check is not enough; the message must explain."""
    src = (_KERNEL / fname).read_text()
    assert 'if not hasattr(tl, "gather"):' in src, f"{fname} lost the guard"
    assert "prefill_variant=0 does NOT work around this" in src, (
        f"{fname}'s guard message no longer explains why the obvious "
        "workaround fails; that explanation is the whole point of the guard."
    )
    assert "DECODE path" in src, (
        f"{fname}'s guard must say decode still works, or a user on an old "
        "triton will assume the package is unusable when it is not."
    )


@pytest.mark.parametrize("fname", _TARGETS)
def test_guard_runs_after_the_decode_early_return(fname):
    """Decode has no tl.gather dependency and MUST keep working on old triton.

    Guarding too early would break a path that is verified working on triton
    3.0 (the A100 runs behind findings #43/#44 used it).
    """
    src = (_KERNEL / fname).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name.startswith("gemm_"))
    body_src = ast.get_source_segment(src, fn) or ""
    guard_at = body_src.index('if not hasattr(tl, "gather"):')
    # every `return` belonging to the decode branch precedes the guard
    returns = [m.start() for m in re.finditer(r"\n        return out\b", body_src)]
    assert returns, f"{fname}: no decode early-return found"
    assert max(returns) < guard_at, (
        f"{fname}: the gather guard sits BEFORE a decode early-return, so it "
        "would reject decode calls that work fine without tl.gather."
    )


def test_runtime_guard_raises_clearly():
    """With tl.gather removed, prefill must raise RuntimeError, not AttributeError."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to reach the launch path")
    import triton.language as tl

    from nf4_grouped import gemm_4bit_grouped

    if not hasattr(tl, "gather"):
        pytest.skip("this triton already lacks tl.gather; guard is live anyway")

    dev = "cuda"
    E, N, K = 2, 128, 128
    B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    am = torch.rand(E, N, K // 64, device=dev).float() + 0.05
    a = (torch.randn(8, K, device=dev) * 0.1).bfloat16()
    eids = torch.tensor([0, 1], dtype=torch.int32, device=dev)

    saved = tl.gather
    try:
        del tl.gather
        with pytest.raises(RuntimeError, match=r"tl\.gather|triton>=3\.4"):
            gemm_4bit_grouped(a, B, am, [4, 4], eids)   # prefill: sizes > 1
    finally:
        tl.gather = saved
