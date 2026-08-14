# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The triton stand-in must MISDIRECT NOBODY.

`_triton_shim` is shared by every module that defines `@triton.jit` kernels, so
a single hard-coded "use dequant_ref" would tell an MXFP4 caller — or a
`host_gather` caller, which has no CPU path at all — to reach for an NF4-only
API. That is a worse failure than the `ModuleNotFoundError` the shim replaced:
an import error is at least honestly unhelpful, where a confidently wrong
pointer costs the reader a debugging session.

These run on EVERY platform, including Linux CI where triton imports fine. The
stand-ins are defined unconditionally and only *bound* when triton is missing,
precisely so this routing is checkable where CI actually runs — logic that only
exists on the platforms CI skips is logic nothing checks.
"""
import pathlib
import re

import pytest

import _triton_shim as shim

_KERNEL_DIR = pathlib.Path(__file__).resolve().parent


def _shim_consumers():
    """Modules that import the shim, read off disk rather than hard-coded."""
    out = set()
    for p in _KERNEL_DIR.glob("*.py"):
        if p.name.startswith("test_"):
            continue
        if re.search(r"^from _triton_shim import", p.read_text(), re.M):
            out.add(p.stem)
    return out


def _launch_failure(module_name: str) -> str:
    def kernel():
        pass

    kernel.__module__ = module_name
    with pytest.raises(RuntimeError) as ei:
        shim._UnlaunchableKernel(kernel)[(1,)]()
    return str(ei.value)


def test_every_shim_consumer_has_a_cpu_path():
    """The drift guard: adding a triton consumer forces a fallback decision.

    Without this, a new module gets the generic message by silent default and
    nobody decides what its CPU story actually is.
    """
    missing = sorted(_shim_consumers() - set(shim._CPU_PATH))
    assert not missing, (
        f"these modules import _triton_shim but have no _CPU_PATH entry: "
        f"{missing}. Add one naming that module's CPU-checkable path — or "
        f"saying plainly that it has none, as host_gather does."
    )


def test_consumers_are_actually_found():
    """Positive control for the scan above.

    A typo in the regex would make `_shim_consumers()` return the empty set and
    the drift guard would pass vacuously forever.
    """
    found = _shim_consumers()
    assert {"nf4_grouped", "mxfp4_grouped", "host_gather"} <= found, found


@pytest.mark.parametrize(
    "module, expect, forbid",
    [
        ("nf4_grouped", "dequant_ref(packed, absmax, N, K)", "dequant_mxfp4"),
        ("mxfp4_grouped", "mxfp4_pack_ref.dequant_mxfp4", "dequant_ref("),
        ("host_gather", "no CPU equivalent", "dequant_ref("),
    ],
)
def test_launch_failure_names_the_right_cpu_path(module, expect, forbid):
    msg = _launch_failure(module)
    assert expect in msg, f"{module}: expected {expect!r} in {msg!r}"
    assert forbid not in msg, f"{module}: must not steer at {forbid!r}: {msg!r}"


def test_unknown_consumer_is_vague_but_never_wrong():
    msg = _launch_failure("some_module_added_later")
    assert "pure-torch reference, where it has one" in msg
    # the whole point: no specific API is named at a module we know nothing about
    assert "dequant_ref(" not in msg and "dequant_mxfp4" not in msg


def test_launch_form_is_subscripted_then_called():
    """`kernel[grid](...)` must fail on the CALL, not on the subscript."""
    def kernel():
        pass

    kernel.__module__ = "nf4_grouped"
    k = shim._UnlaunchableKernel(kernel)
    assert k[(1, 2)] is k, "subscript must return the kernel, not raise"
    with pytest.raises(RuntimeError):
        k()


def test_kernel_identity_survives_the_stub():
    """The failure names the kernel, so `__name__` has to come through."""
    def _gemm_nf4_grouped():
        pass

    _gemm_nf4_grouped.__module__ = "nf4_grouped"
    k = shim._UnlaunchableKernel(_gemm_nf4_grouped)
    assert k.__name__ == "_gemm_nf4_grouped"
    with pytest.raises(RuntimeError) as ei:
        k[(1,)]()
    assert str(ei.value).startswith("_gemm_nf4_grouped: ")


def test_unknown_attribute_raises_so_getattr_default_wins():
    """`getattr(tl, "gather", None)` must resolve to None, not to a stub.

    If `__getattr__` returned another stand-in, `HAS_TL_GATHER` would go True on
    a box with no triton and the register-LUT variant would be selected against
    a stub. Raising is what makes the probe answer correctly.
    """
    tl = shim._MissingModule("triton.language")
    assert getattr(tl, "gather", None) is None
    with pytest.raises(AttributeError):
        tl.gather


def test_jit_defines_but_does_not_launch():
    """Kernels are built at import, so decorating must succeed everywhere."""
    triton = shim._MissingModule("triton")

    @triton.jit
    def kernel():
        pass

    assert isinstance(kernel, shim._UnlaunchableKernel)

    @triton.jit(num_warps=4)          # the decorator-with-kwargs form
    def kernel_kw():
        pass

    assert isinstance(kernel_kw, shim._UnlaunchableKernel)


def test_real_triton_is_bound_when_present():
    """Where triton imports, the shim must hand back the real thing."""
    if not shim.HAS_TRITON:
        pytest.skip("no triton on this platform — the fallback path is under test above")
    assert shim.triton.__name__ == "triton"
    assert not isinstance(shim.triton, shim._MissingModule)
    assert not isinstance(shim.tl, shim._MissingModule)
