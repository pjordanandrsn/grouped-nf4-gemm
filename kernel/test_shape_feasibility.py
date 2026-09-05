# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Shape feasibility (gnf4#324): a tile that will not fit the device's shared
memory is refused or re-dispatched BEFORE the launch, with the numbers, never
surfaced as Triton's ``OutOfResources``.

Two layers, so the rules are checked where they can be:

  - the selection functions are pure Python -- ``nf4_grouped.prefill_fit``
    picks the largest feasible M-tile configuration or raises
    ``UnsupportedShapeError``; ``fp8_paged_attn.packed_unsupported`` decides
    from the calibrated ``packed_tile_smem_bytes`` model whether the packed
    fp8 attention tile fits -- and are driven here with MOCKED device limits
    (the RTX 5090's 101376 B as Triton reported it in gnf4#324, an H100's
    227 KB, CDNA3's 64 KB), so this half runs on any CPU with torch;
  - under ``TRITON_INTERPRET=1`` the NF4 M-tile fit-down is exercised end to
    end: a capped limit trims the configuration and the result still matches
    ``dequant_ref``; an impossible limit raises before any launch; an explicit
    ``prefill_config=`` is launched as given.

An ``_INTERP_FILES`` member (kernel/conftest.py): it sets ``TRITON_INTERPRET``
at import, so it runs in its own process.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import warnings  # noqa: E402

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

import fp8_paged_attn as FA  # noqa: E402
import nf4_grouped as NG  # noqa: E402
from _triton_shim import (HAS_TRITON, UnsupportedShapeError,  # noqa: E402
                          device_shared_mem_limit)

# Device limits, in bytes, as the drivers report them.
SM120 = 101376      # RTX 5090: Triton's "Hardware limit" in gnf4#324
H100 = 232448       # 227 KB opt-in per block
CDNA3 = 65536       # MI300X (gfx942) LDS
MEASURED_OVERFLOW = 148480   # Triton's "Required" at D=256, 8 kv heads (gnf4#324)


# ---------------------------------------------------------------- the error --
def test_unsupported_shape_error_carries_the_numbers():
    e = UnsupportedShapeError("k", {"D": 256, "H_kv": 8}, MEASURED_OVERFLOW,
                              SM120, hint="use pack_heads=False")
    assert isinstance(e, ValueError)
    assert (e.kernel, e.need_bytes, e.limit_bytes) == ("k", MEASURED_OVERFLOW, SM120)
    assert e.shape == {"D": 256, "H_kv": 8}
    msg = str(e)
    assert "D=256" in msg and "H_kv=8" in msg
    assert str(MEASURED_OVERFLOW) in msg and str(SM120) in msg
    assert "pack_heads=False" in msg


def test_device_limit_is_an_int_and_zero_means_unqueryable():
    lim = device_shared_mem_limit()
    assert isinstance(lim, int) and lim >= 0
    if not HAS_TRITON or not torch.cuda.is_available():
        assert lim == 0, "no triton or no device must read as unqueryable, not as a guess"
    else:
        assert lim >= 48 * 1024, lim   # every supported NVIDIA part exposes >= 48 KB


# ------------------------------------------------ packed fp8 attention tile --
def test_packed_model_reproduces_the_measured_overflow():
    """The one Triton report on record: 148480 bytes at Gemma-4's sliding
    geometry (head_dim 256, 8 kv heads, 16 tokens/block, 4 key groups, 3
    stages). The model is calibrated to reproduce it EXACTLY, not roughly."""
    assert FA.packed_tile_smem_bytes(256, 8, 16, 4, 3) == MEASURED_OVERFLOW


@pytest.mark.parametrize("d,kg,hkv", [(128, 4, 4), (512, 4, 2), (512, 8, 2),
                                      (512, 16, 2), (64, 1, 4), (64, 2, 4),
                                      (128, 4, 8)])
def test_packed_geometries_the_suite_runs_on_sm120_are_admitted(d, kg, hkv):
    """Every packed geometry kernel/test_fp8_paged_attn.py runs on the 101376
    B card must NOT be pre-refused: an over-eager model would silently move
    those cells to the split kernel."""
    assert FA.packed_unsupported(d, hkv, 16, kg, 3, compute="fp8",
                                 smem_limit=SM120) is None


@pytest.mark.parametrize("kg", [4, 8])
def test_issue_geometry_is_refused_on_sm120_and_admitted_on_h100(kg):
    why = FA.packed_unsupported(256, 8, 16, kg, 3, compute="fp8", smem_limit=SM120)
    assert why is not None
    assert str(FA.packed_tile_smem_bytes(256, 8, 16, kg, 3)) in why
    assert str(SM120) in why and "D 256" in why and "H_kv 8" in why
    assert FA.packed_unsupported(256, 8, 16, kg, 3, compute="fp8",
                                 smem_limit=H100) is None


def test_f32_packed_is_never_pre_refused():
    """No calibrated model for the f32 packed kernel: it is not refused on a
    model, its launch's own overflow falls back (the wrapper's catch)."""
    assert FA.packed_unsupported(256, 8, 16, 4, 3, compute="f32",
                                 smem_limit=SM120) is None


def test_unqueryable_limit_never_refuses():
    assert FA.packed_unsupported(256, 8, 16, 4, 3, compute="fp8", smem_limit=0) is None


def test_packed_model_scales_with_the_tile_and_the_pipeline():
    base = FA.packed_tile_smem_bytes(128, 8, 16, 4, 3)
    payload = 16 * 8 * 128
    # more heads or a taller block scale every term; a wider head scales the
    # payload and the sub-dot operand but not the per-column V scales
    assert FA.packed_tile_smem_bytes(128, 16, 16, 4, 3) == 2 * base
    assert FA.packed_tile_smem_bytes(128, 8, 32, 4, 3) == 2 * base
    assert (FA.packed_tile_smem_bytes(256, 8, 16, 4, 3) - base
            == 2 * (2 * payload + payload // 4))
    assert FA.packed_tile_smem_bytes(128, 8, 16, 4, 2) == base // 2
    assert FA.packed_tile_smem_bytes(128, 8, 16, 4, 1) == base // 2   # >= 1 buffer
    assert FA.packed_tile_smem_bytes(128, 8, 16, 8, 3) < base           # narrower sub-dot


def test_packed_pre_launch_dispatch_falls_back_before_the_launch(monkeypatch):
    """The wrapper consults the model before deriving the split grid: with
    the limit mocked to the 5090's, the issue geometry flips pack_heads off,
    warns once, remembers the geometry, and NEVER reaches a packed launch.
    Runs without a device: the fallback decision is host-side, and the split
    launch that follows is stubbed."""
    FA._PACKED_FALLBACK_WARNED.clear()
    FA._PACKED_UNFIT.clear()
    monkeypatch.setattr(FA, "device_shared_mem_limit", lambda dev=None: SM120)
    monkeypatch.setattr(FA, "paged_attn_available", lambda: True)
    if not FA._TRITON:
        # no triton on this platform: the wrapper's two host-side helpers
        import types
        monkeypatch.setattr(FA, "triton", types.SimpleNamespace(
            next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length(),
            cdiv=lambda a, b: -(-a // b)), raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev: type("P", (), {"multi_processor_count": 170,
                                                   "major": 12, "minor": 0})())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev=None: (12, 0))
    launched = []

    class _Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *a, **k: launched.append((self.name, grid))

    for name in ("_fp8_paged_decode_packed_f8", "_fp8_paged_decode_packed",
                 "_fp8_paged_decode_split_f8dot", "_fp8_paged_decode_split",
                 "_fp8_combine"):
        monkeypatch.setattr(FA, name, _Launch(name), raising=False)
    monkeypatch.setattr(FA, "_fuse_counters", lambda n, dev: torch.zeros(n, dtype=torch.int32))
    B, hq, hkv, d = 1, 16, 8, 256
    q = torch.zeros(B, hq, d, dtype=torch.bfloat16)
    from fp8_kv import kv_block_bytes
    k_row = kv_block_bytes(16, hkv, d) + 16 * hkv * 4 * 3
    v_row = kv_block_bytes(16, hkv, d)
    kp = torch.zeros(4 * k_row, dtype=torch.uint8)
    vp = torch.zeros(4 * v_row, dtype=torch.uint8)
    tab = torch.zeros(B, 4, dtype=torch.int32)
    lens = torch.tensor([64], dtype=torch.int32)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(2):
            FA.fp8_paged_decode_attention(q, kp, vp, tab, lens, n_kv_heads=hkv,
                                          head_dim=d, k_groups=4, compute="fp8",
                                          pack_heads=True)
    names = [n for n, _ in launched]
    assert names == ["_fp8_paged_decode_split_f8dot"] * 2, names
    fell = [w for w in rec if "falling back to the split fp8 kernel" in str(w.message)]
    assert len(fell) == 1, [str(w.message) for w in rec]
    assert str(MEASURED_OVERFLOW) in str(fell[0].message)
    assert (d, hkv, 16) in FA._PACKED_UNFIT
    # the split grid was derived for the split kernel (B * H_kv CTAs per
    # split), not inherited from the packed grid (B CTAs per split)
    (_, grid), = launched[:1]
    assert grid[0] % (B * hkv) == 0 and grid[0] // (B * hkv) >= 1
    FA._PACKED_FALLBACK_WARNED.clear()
    FA._PACKED_UNFIT.clear()


class OutOfResources(Exception):
    """The shape of ``triton.runtime.errors.OutOfResources`` (matched by
    class NAME in the wrapper, so this stand-in exercises the same branch)."""

    def __init__(self, required, limit, name):
        self.required, self.limit, self.name = required, limit, name
        super().__init__(f"out of resource: {name}, Required: {required}, "
                         f"Hardware limit: {limit}")


def _device_free_wrapper(monkeypatch, launches: dict):
    """Run the wrapper with no device: the limit, the capability queries and
    the kernel launches are stubbed; ``launches`` maps a kernel name to a
    callable invoked with the launch's positional args (raise from it to
    simulate Triton)."""
    monkeypatch.setattr(FA, "device_shared_mem_limit", lambda dev=None: SM120)
    monkeypatch.setattr(FA, "paged_attn_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev: type("P", (), {"multi_processor_count": 170,
                                                   "major": 12, "minor": 0})())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev=None: (12, 0))
    monkeypatch.setattr(FA, "_fuse_counters", lambda n, dev: torch.zeros(n, dtype=torch.int32))
    if not FA._TRITON:
        import types
        monkeypatch.setattr(FA, "triton", types.SimpleNamespace(
            next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length(),
            cdiv=lambda a, b: -(-a // b)), raising=False)
    seen = []

    class _Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def run(*a, **k):
                seen.append((self.name, grid))
                fn = launches.get(self.name)
                if fn is not None:
                    fn(*a)
            return run

    for name in ("_fp8_paged_decode_packed_f8", "_fp8_paged_decode_packed",
                 "_fp8_paged_decode_split_f8dot", "_fp8_paged_decode_split",
                 "_fp8_combine"):
        monkeypatch.setattr(FA, name, _Launch(name), raising=False)
    return seen


def _problem(hkv=8, d=256, hq=16):
    from fp8_kv import kv_block_bytes
    q = torch.zeros(1, hq, d, dtype=torch.bfloat16)
    k_row = kv_block_bytes(16, hkv, d) + 16 * hkv * 4 * 3
    v_row = kv_block_bytes(16, hkv, d)
    kp = torch.zeros(4 * k_row, dtype=torch.uint8)
    vp = torch.zeros(4 * v_row, dtype=torch.uint8)
    tab = torch.zeros(1, 4, dtype=torch.int32)
    lens = torch.tensor([64], dtype=torch.int32)
    return (q, kp, vp, tab, lens), dict(n_kv_heads=hkv, head_dim=d, k_groups=4)


def test_f32_packed_overflow_at_the_launch_falls_back_to_split_f32(monkeypatch):
    """No calibrated model for the f32 packed kernel: Triton's overflow at
    the launch (previously a raw OutOfResources to the caller) is the
    signal; the split f32 kernel serves the call, one warning names the f32
    mode, the geometry is remembered, and the call tallies once."""
    FA._PACKED_FALLBACK_WARNED.clear()
    FA._PACKED_UNFIT.clear()
    FA.reset_compute_counts()

    def overflow(*a):
        raise OutOfResources(268288, SM120, "shared memory")

    seen = _device_free_wrapper(monkeypatch, {"_fp8_paged_decode_packed": overflow})
    args, kw = _problem()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        FA.fp8_paged_decode_attention(*args, **kw, compute="f32", pack_heads=True)
        FA.fp8_paged_decode_attention(*args, **kw, compute="f32", pack_heads=True)
    names = [n for n, _ in seen]
    # first call: packed attempt, then the split f32 kernel (+ its combine
    # when the f32 fused combine is off); second call: split only
    assert names[0] == "_fp8_paged_decode_packed"
    assert names.count("_fp8_paged_decode_packed") == 1
    assert names.count("_fp8_paged_decode_split") == 2
    assert "_fp8_paged_decode_split_f8dot" not in names
    fell = [w for w in rec if "falling back to the split f32 kernel" in str(w.message)]
    assert len(fell) == 1, [str(w.message) for w in rec]
    assert (256, 8, 16) in FA._PACKED_UNFIT
    assert FA.compute_counts() == {"f32": 2, "fp8": 0}
    FA._PACKED_FALLBACK_WARNED.clear()
    FA._PACKED_UNFIT.clear()


def test_split_overflow_at_the_launch_is_the_typed_refusal(monkeypatch):
    def overflow(*a):
        raise OutOfResources(262144, SM120, "shared memory")

    _device_free_wrapper(monkeypatch, {"_fp8_paged_decode_split_f8dot": overflow,
                                       "_fp8_paged_decode_split": overflow})
    args, kw = _problem(hkv=2, d=1024, hq=4)
    for compute in ("fp8", "f32"):
        with pytest.raises(UnsupportedShapeError) as ei:
            FA.fp8_paged_decode_attention(*args, **kw, compute=compute)
        e = ei.value
        assert (e.need_bytes, e.limit_bytes) == (262144, SM120)
        assert e.shape["head_dim"] == 1024 and "ktile" in e.shape
        assert "ktile" in str(e) and e.kernel.startswith("fp8_paged_attn._fp8_paged_decode_split")


def test_split_out_of_resources_becomes_the_typed_refusal():
    with pytest.raises(UnsupportedShapeError) as ei:
        FA._refuse_out_of_resources(
            OutOfResources(MEASURED_OVERFLOW, SM120, "shared memory"),
            "fp8_paged_attn._fp8_paged_decode_split_f8dot",
            {"head_dim": 1024, "ktile": 64}, "pass a smaller ktile")
    e = ei.value
    assert (e.need_bytes, e.limit_bytes) == (MEASURED_OVERFLOW, SM120)
    assert "head_dim=1024" in str(e) and "ktile" in str(e)
    assert isinstance(e.__cause__, OutOfResources)
    # anything else passes through untouched
    with pytest.raises(RuntimeError, match="unrelated"):
        FA._refuse_out_of_resources(RuntimeError("unrelated"), "k", {}, "")


# ------------------------------------------------------- NF4 M-tile fit-down --
V1, V0 = 1, 0     # register-LUT mainloop (the default), the v5 loop


def test_nvidia_default_configuration_is_untouched():
    for lim in (SM120, H100):
        assert NG.prefill_fit(128, 128, 64, 3, V1, lim) == (128, 3)


def test_v5_loop_trims_stages_on_sm120_exactly_as_before():
    """3 * (128*64*2 + 128*64*2) = 98304 > 101376 - 8192: the inline fit-down
    this function replaced already ran the v5 loop at 2 stages there, and
    so does this (no NVIDIA configuration moves with the refactor)."""
    assert NG.prefill_smem_bytes(128, 128, 64, 3, V0) == 98304
    assert NG.prefill_fit(128, 128, 64, 3, V0, SM120) == (128, 2)
    assert NG.prefill_fit(128, 128, 64, 3, V0, H100) == (128, 3)


def test_cdna3_descends_stages_then_block_m():
    assert NG.prefill_fit(128, 128, 64, 3, V0, CDNA3) == (64, 2)
    assert NG.prefill_fit(128, 128, 64, 3, V1, CDNA3) == (128, 2)


def test_descent_order_is_stages_block_m_stages():
    # budget 31808: (128,3)->98304 (128,2)->65536 (64,2)->49152 (64,1)->24576
    assert NG.prefill_fit(128, 128, 64, 3, V0, 40000) == (64, 1)
    # the estimates the descent evaluates, in order: stages first, then
    # block_m, then stages again, then the final fit check
    seen = []
    orig = NG.prefill_smem_bytes

    def spy(bm, bn, bk, st, v):
        seen.append((bm, st))
        return orig(bm, bn, bk, st, v)

    NG.prefill_smem_bytes, saved = spy, NG.prefill_smem_bytes
    try:
        NG.prefill_fit(128, 128, 64, 3, V0, 40000)
    finally:
        NG.prefill_smem_bytes = saved
    assert seen == [(128, 3), (128, 2), (64, 2), (64, 1)], seen


def test_unqueryable_limit_is_a_no_op():
    assert NG.prefill_fit(128, 128, 64, 3, V0, 0) == (128, 3)


def test_refuses_with_the_numbers_when_nothing_fits():
    with pytest.raises(UnsupportedShapeError) as ei:
        NG.prefill_fit(128, 128, 64, 3, V0, 16384)
    e = ei.value
    assert e.kernel == "nf4_grouped._gemm_nf4_grouped"
    assert e.shape["BLOCK_M"] == 64 and e.shape["num_stages"] == 1
    assert e.need_bytes == NG.prefill_smem_bytes(64, 128, 64, 1, V0) + NG.PREFILL_SMEM_HEADROOM
    assert e.limit_bytes == 16384
    assert "prefill_config=" in str(e)


# ------------------------------------------- interpreter mode: end to end --
def _ref(B, A, sizes, ids, acts, N, K):
    out = torch.empty(sum(sizes), N, dtype=torch.float32)
    row = 0
    for m, e in zip(sizes, ids):
        w = NG.dequant_ref(B[e], A[e], N, K).float()
        for _ in range(m):
            out[row] = w @ acts[row].float()
            row += 1
    return out


def _interp_problem():
    pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")
    from nf4_pack_ref import make_stack
    E, N, K = 3, 128, 128
    sizes, ids = [70, 3], [2, 0]
    B, A = make_stack(E, N, K, seed=5)
    acts = torch.randn(sum(sizes), K, dtype=torch.bfloat16,
                       generator=torch.Generator().manual_seed(6))
    return B, A, sizes, ids, acts, N, K


def test_interp_fit_down_trims_and_still_matches_dequant_ref(monkeypatch):
    B, A, sizes, ids, acts, N, K = _interp_problem()
    monkeypatch.setattr(NG, "_device_shared_limit", lambda dev: 40000)
    built = []
    orig = NG.build_group_tiles

    def spy(sizes_, block_m, dev):
        built.append(block_m)
        return orig(sizes_, block_m, dev)

    monkeypatch.setattr(NG, "build_group_tiles", spy)
    out = NG.gemm_4bit_grouped(acts, B, A, sizes, torch.tensor(ids, dtype=torch.int32),
                               prefill_variant=V0)
    assert built == [64], built          # 128-row group, trimmed from bm 128
    ref = _ref(B, A, sizes, ids, acts, N, K)
    rel = (out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-4)
    assert rel.item() < 1e-2, rel.item()


def test_interp_impossible_limit_refuses_before_any_launch(monkeypatch):
    B, A, sizes, ids, acts, N, K = _interp_problem()
    monkeypatch.setattr(NG, "_device_shared_limit", lambda dev: 16384)

    class _NeverLaunch:
        def __getitem__(self, grid):
            raise AssertionError("the M-tile kernel was launched after a refusal")

    monkeypatch.setattr(NG, "_gemm_nf4_grouped", _NeverLaunch())
    with pytest.raises(UnsupportedShapeError) as ei:
        NG.gemm_4bit_grouped(acts, B, A, sizes, torch.tensor(ids, dtype=torch.int32),
                             prefill_variant=V0)
    assert ei.value.limit_bytes == 16384


def test_interp_explicit_prefill_config_is_launched_as_given(monkeypatch):
    B, A, sizes, ids, acts, N, K = _interp_problem()
    monkeypatch.setattr(NG, "_device_shared_limit", lambda dev: 16384)
    out = NG.gemm_4bit_grouped(acts, B, A, sizes, torch.tensor(ids, dtype=torch.int32),
                               prefill_variant=V0, prefill_config=(128, 4, 3))
    ref = _ref(B, A, sizes, ids, acts, N, K)
    rel = (out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-4)
    assert rel.item() < 1e-2, rel.item()
