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


# ------------------------------------------------------------------------------------- GNF4_TRAIN_GEMM=decoded: opt-in only
@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize("has", [True, False])
def test_decoded_is_taken_only_when_asked_for(cap, has):
    """An explicit `decoded` runs on any CUDA card at the floor, with or without torch._grouped_mm; `auto` never answers it, for
    any capability or group count -- the route ships opt-in, `auto` untouched (experts4bit-qlora RD1's decision)."""
    route, why = route_for(cap, has_grouped_mm=has, requested="decoded")
    assert route == "decoded" and "opt-in" in why
    for n in (None, 0, 1, 16, 17, 64, 128, 256):
        assert route_for(cap, has_grouped_mm=has, n_groups=n)[0] != "decoded", (cap, has, n)


def test_decoded_below_the_floor_or_without_cuda_has_no_route():
    assert route_for((7, 5), has_grouped_mm=True, requested="decoded")[0] is None
    assert route_for(None, has_grouped_mm=True, requested="decoded")[0] is None
    assert route_for((12, 0), has_grouped_mm=True, requested=" Decoded ")[0] == "decoded"


def test_decoded_env_value_is_accepted(monkeypatch):
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "decoded")
    assert nf4_route.train_gemm_route(torch.device("cpu")) == "decoded"      # an explicit value is used as given


@pytest.mark.parametrize("groups,N,K,cap,want", [
    (10, 1024, 2048, 256 * 2**20, [(0, 10)]),                        # 4 MiB each: one chunk
    (60, 2048, 2048, 256 * 2**20, [(0, 32), (32, 60)]),              # olmoe gate_up: 8 MiB each, 32 per chunk
    (8, 28672, 4096, 256 * 2**20, [(i, i + 1) for i in range(8)]),   # mixtral gate_up: 224 MiB each, one per chunk
    (3, 28672, 4096, 2**20, [(0, 1), (1, 2), (2, 3)]),               # a cap below one expert still decodes it whole
    (0, 64, 128, 2**20, []),
])
def test_decoded_chunks(groups, N, K, cap, want):
    assert nf4_route.decoded_chunks(groups, N, K, cap) == want
    for g0, g1 in want:
        assert (g1 - g0) * N * K * 2 <= cap or g1 - g0 == 1


def test_decoded_max_bytes(monkeypatch):
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    assert nf4_route.decoded_max_bytes() == nf4_route.DECODED_MAX_BYTES_DEFAULT == 256 * 2**20
    monkeypatch.setenv("GNF4_DECODED_MAX_BYTES", " 1048576 ")
    assert nf4_route.decoded_max_bytes() == 2**20
    for bad in ("0", "-5", "256MiB", "1e6"):
        monkeypatch.setenv("GNF4_DECODED_MAX_BYTES", bad)
        with pytest.raises(ValueError, match="GNF4_DECODED_MAX_BYTES"):
            nf4_route.decoded_max_bytes()


def test_decoded_tiles_split_on_mean_rows():
    split = nf4_route.DECODED_ROWS_SPLIT
    assert nf4_route.decoded_tiles(split * 4 - 1, 4) == nf4_route.DECODED_TILES_SMALL
    assert nf4_route.decoded_tiles(split * 4, 4) == nf4_route.DECODED_TILES_LARGE
    assert nf4_route.decoded_tiles(10, 0) == nf4_route.DECODED_TILES_SMALL
    # BLOCK_K is shared, so the two tilings accumulate each output over the same K slices
    assert nf4_route.DECODED_TILES_SMALL[2] == nf4_route.DECODED_TILES_LARGE[2]
