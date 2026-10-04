"""nf4_route.route_for: the training-route decision as a pure function of device facts. It must agree with what
train_gemm_route resolves on a live device, and it must answer without one -- that is the point of it."""
import pytest
import torch

import nf4_route
from nf4_route import route_for


@pytest.mark.parametrize("cap,has,want", [
    ((9, 0), True, "grouped_mm"),
    ((9, 0), False, "fused"),          # the card could, this torch cannot
    ((8, 6), True, "fused"),           # RTX A2000 / 3090
    ((8, 9), True, "fused"),           # L4 / 4090
    ((10, 0), True, "fused"),          # torch 2.8's _grouped_mm has no sm_100 kernel
    ((12, 0), True, "fused"),          # RTX 5090
    ((8, 0), True, "fused"),           # A100: exactly the floor
])
def test_auto(cap, has, want):
    route, why = route_for(cap, has_grouped_mm=has)
    assert route == want
    assert why.startswith("auto")


@pytest.mark.parametrize("n,want", [(1, "dense"), (16, "dense"), (17, "fused"), (128, "fused"), (0, "fused")])
def test_auto_takes_the_dense_route_for_few_groups_off_sm90(n, want):
    route, why = route_for((8, 6), has_grouped_mm=True, n_groups=n)
    assert route == want and str(n) in why
    assert route_for((9, 0), has_grouped_mm=True, n_groups=n)[0] == "grouped_mm"     # sm_90 keeps grouped_mm
    assert route_for((8, 6), has_grouped_mm=True, requested="fused", n_groups=n)[0] == "fused"   # explicit wins


def test_below_the_floor_has_no_route_and_says_why():
    route, why = route_for((7, 5), has_grouped_mm=True)
    assert route is None
    assert "7.5" in why and "8.0" in why


def test_no_cuda_device():
    route, why = route_for(None, has_grouped_mm=True)
    assert route is None and "CUDA" in why


def test_explicit_requests():
    assert route_for((8, 6), has_grouped_mm=True, requested="fused")[0] == "fused"
    assert route_for((8, 6), has_grouped_mm=False, requested="dense")[0] == "dense"     # any CUDA card at the floor
    assert route_for((7, 5), has_grouped_mm=True, requested="dense")[0] is None
    assert route_for((9, 0), has_grouped_mm=True, requested="grouped_mm")[0] == "grouped_mm"
    route, why = route_for((8, 6), has_grouped_mm=True, requested="grouped_mm")
    assert route is None and "9.0 only" in why and "8.6" in why
    route, why = route_for((9, 0), has_grouped_mm=False, requested="grouped_mm")
    assert route is None and "has none" in why


def test_requested_is_validated_like_the_env_var():
    with pytest.raises(ValueError):
        route_for((9, 0), has_grouped_mm=True, requested="cublas")
    assert route_for((9, 0), has_grouped_mm=True, requested=" Fused ")[0] == "fused"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares against the live device")
def test_agrees_with_the_live_resolution(monkeypatch):
    monkeypatch.delenv("GNF4_TRAIN_GEMM", raising=False)
    nf4_route._AUTO_ROUTE.clear()
    idx = torch.cuda.current_device()
    cap, gmm = torch.cuda.get_device_capability(idx), hasattr(torch, "_grouped_mm")
    for n in (None, 4, 16, 17, 128):
        want, _ = route_for(cap, has_grouped_mm=gmm, n_groups=n)
        assert nf4_route.train_gemm_route(None, n) == (want or "fused"), n
