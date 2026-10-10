# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The vendor-first refusal map, on CPU with a mocked ROCm build.

A ROCm torch answers ``torch.cuda.get_device_capability`` with gfx numbers: MI300X (gfx942) is ``(9, 4)``, which passes
every sm-number rule written here (``>= (8, 9)`` for fp8). Each gate below must ask the vendor first and refuse in words
that name AMD. Every refusal is paired with the same call under an NVIDIA build at the same capability tuple, which must
NOT refuse on the vendor -- otherwise a test could pass because the gate refuses everything.
"""

import pytest
import torch

import _triton_shim as shim
import fp8_kv
import fp8_paged_attn as attn
import nf4_route

GFX942 = (9, 4)   # MI300X, CDNA3
GFX90A = (9, 0)   # MI200 reports the one capability torch._grouped_mm's auto route keys on


@pytest.fixture
def rocm(monkeypatch):
    """A ROCm build of torch on a gfx942 card, as far as every gate can see."""
    monkeypatch.setattr(torch.version, "hip", "6.4.43482", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: GFX942)
    return monkeypatch


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: GFX942)
    return monkeypatch


def test_rocm_refusal_names_amd_only_on_rocm(rocm):
    why = shim.rocm_refusal("the widget")
    assert why and "AMD" in why and "the widget" in why and "6.4.43482" in why
    assert "gfx number" in why
    rocm.setattr(torch.version, "hip", None)
    assert shim.rocm_refusal("the widget") is None and shim.rocm_version() is None


def test_fp8_attention_refuses_on_rocm_and_the_default_is_f32(rocm):
    q = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    why = attn.fp8_compute_unsupported(q, 128, 1, 1)
    assert why and "AMD" in why and "fp8" in why
    rocm.delenv("GNF4_ATTN_COMPUTE", raising=False)
    assert attn._compute_default(q, 128, 1, 1) == "f32"


def test_fp8_attention_control_nvidia_same_tuple_is_allowed(nvidia):
    """(9, 4) as an sm number passes the sm_89 rule: the AMD refusal above is the vendor, not the tuple."""
    q = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    assert attn.fp8_compute_unsupported(q, 128, 1, 1) is None
    nvidia.delenv("GNF4_ATTN_COMPUTE", raising=False)
    assert attn._compute_default(q, 128, 1, 1) == "fp8"


def test_fp8_attention_vendor_is_asked_before_the_call_shape(rocm):
    """Vendor first: a call that ALSO fails a shape rule is refused for being on AMD."""
    q = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    assert "AMD" in attn.fp8_compute_unsupported(q, 128, 1, 2)


@pytest.mark.parametrize("which", ["t1", "bt1"])
def test_fused_kv_append_refuses_on_rocm_before_launching(rocm, which):
    H, D = 4, 128
    pool = torch.zeros(4096, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match=r"AMD GPU \(ROCm .*\): the fused FP8 KV append") as err:
        if which == "t1":
            fp8_kv.fp8_kv_append_t1(torch.zeros(H, D), pool, torch.zeros(2, dtype=torch.int32),
                                    torch.zeros(1, dtype=torch.int32), 2048, 1024, 8, 1)
        else:
            fp8_kv.fp8_kv_append_bt1(torch.zeros(1, H, D), pool, torch.zeros(1, 2, dtype=torch.int32),
                                     torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
                                     2048, 1024, 8, 1)
    assert "quantize_kv_fp8" in str(err.value)


@pytest.mark.parametrize("which", ["t1", "bt1"])
def test_fused_kv_append_control_nvidia_reaches_the_availability_boundary(nvidia, which):
    """The same valid calls under an NVIDIA build: no vendor refusal; each stops at the CPU-tensor boundary as before."""
    H, D = 4, 128
    pool = torch.zeros(4096, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="needs triton and CUDA") as err:
        if which == "t1":
            fp8_kv.fp8_kv_append_t1(torch.zeros(H, D), pool, torch.zeros(2, dtype=torch.int32),
                                    torch.zeros(1, dtype=torch.int32), 2048, 1024, 8, 1)
        else:
            fp8_kv.fp8_kv_append_bt1(torch.zeros(1, H, D), pool, torch.zeros(1, 2, dtype=torch.int32),
                                     torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
                                     2048, 1024, 8, 1)
    assert "AMD" not in str(err.value)


def test_fused_kv_append_bad_arguments_still_fail_as_bad_arguments(rocm):
    with pytest.raises(ValueError, match="must divide head_dim"):
        fp8_kv.fp8_kv_append_t1(torch.zeros(4, 128), torch.zeros(4096, dtype=torch.uint8),
                                torch.zeros(2, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), 2048, 1024, 8, 3)


def test_route_never_auto_selects_grouped_mm_from_a_gfx_tuple():
    route, why = nf4_route.route_for(GFX90A, has_grouped_mm=True, vendor="amd")
    assert route == "fused" and "AMD" in why
    route, why = nf4_route.route_for(GFX90A, has_grouped_mm=True, vendor="amd", n_groups=4)
    assert route == "dense" and "AMD" in why


def test_route_control_nvidia_sm90_still_auto_selects_grouped_mm():
    assert nf4_route.route_for(GFX90A, has_grouped_mm=True, vendor="nvidia")[0] == "grouped_mm"


def test_route_refuses_explicit_grouped_mm_on_amd_in_words():
    route, why = nf4_route.route_for(GFX90A, has_grouped_mm=True, requested="grouped_mm", vendor="amd")
    assert route is None and "AMD" in why and "grouped_mm" in why


def test_route_explicit_kernels_are_used_as_given_on_amd():
    """Only the capability-number decisions change: a route the caller names is still the caller's."""
    for requested in ("fused", "dense", "decoded"):
        assert nf4_route.route_for(GFX942, has_grouped_mm=True, requested=requested, vendor="amd")[0] == requested


def test_route_vendor_defaults_to_the_running_torch(rocm):
    assert nf4_route.route_for(GFX90A, has_grouped_mm=True)[0] == "fused"
    rocm.setattr(torch.version, "hip", None)
    assert nf4_route.route_for(GFX90A, has_grouped_mm=True)[0] == "grouped_mm"


def test_route_does_not_judge_an_amd_tuple_by_the_sm_floor():
    """Vendor before capability: (7, 5) is below the sm_80 floor as an sm number, and means nothing as a gfx number."""
    assert nf4_route.route_for((7, 5), has_grouped_mm=True, vendor="nvidia")[0] is None
    route, why = nf4_route.route_for((7, 5), has_grouped_mm=True, vendor="amd")
    assert route == "fused" and "AMD" in why and "floor" not in why


def test_route_rejects_an_unknown_vendor():
    with pytest.raises(ValueError, match="vendor"):
        nf4_route.route_for(GFX90A, has_grouped_mm=True, vendor="intel")


def test_prebind_is_refused_on_rocm_even_when_forced(monkeypatch):
    if not shim.HAS_TRITON:
        pytest.skip("triton is not installed: prebind returns the kernel unchanged regardless of vendor")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setattr(shim, "_triton_version", lambda: shim.PREBIND_TRITON[0])
    triton, tl = shim.triton, shim.tl

    @triton.jit
    def _k(x_ptr, N: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, N), tl.zeros((N,), tl.float32))

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert shim.prebind_unsupported() is None
    assert isinstance(shim.prebind(_k, force=True), shim.Prebound), "control: the same kernel IS prebound on NVIDIA"
    monkeypatch.setattr(torch.version, "hip", "6.4.43482")
    why = shim.prebind_unsupported()
    assert why and "AMD" in why
    assert shim.prebind(_k, force=True) is _k
    monkeypatch.setenv("GNF4_TRITON_PREBIND", "1")
    assert shim.prebind(_k) is _k
