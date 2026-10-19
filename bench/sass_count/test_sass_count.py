# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The static instruction counter (``sass_count.py``): its parser and loop finder on a fixture, the JIT specialization
it must carry, and, where triton's NVIDIA backend is installed (it ships in the triton wheel, no GPU needed), K25
compiled at e4b's served plan.

The specialization is the point of the compile test. With the JIT's ``tt.divisibility`` hints, K25's weight bytes
load vectorized. Without them, they load as single bytes with per-element address arithmetic, the inflated count.
So the test fails if the hints go missing, not just if a number moves.

The STRUCTURE is checked on any triton (CI's latest-triton step). The COUNTS are pinned only at the served stack's
toolchain, triton 3.4.0 with its bundled ptxas 12.8.93 (CI's hip-compile step), because a newer one moves them."""
import functools
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
KERNEL = HERE.parents[1] / "kernel"
sys.path.insert(0, str(HERE))
import sass_count as sc  # noqa: E402

FIXTURE = """
        /*0000*/                   S2R R0, SR_CTAID.X ;
        /*0010*/                   MOV R1, 0x10 ;
        /*0020*/                   LDG.E R2, desc[UR4][R4.64] ;
        /*0030*/                   PRMT R3, R2, 0x5140, R1 ;
        /*0040*/                   FSEL R5, R6, R7, P0 ;
        /*0050*/                   HMMA.1688.F32.TF32 R8, R10, R12, R8 ;
        /*0060*/              @P1  BRA 0x20 ;
        /*0070*/                   IADD3 R9, R9, 0x1, RZ ;
        /*0080*/              @P2  BRA 0x10 ;
        /*0090*/                   EXIT ;
"""


def test_the_parser_and_the_inner_loop_on_a_fixture():
    rows = sc.parse_sass(FIXTURE)
    assert [r[1] for r in rows][:3] == ["S2R", "MOV", "LDG.E"] and len(rows) == 10
    assert sc.backward_branches(rows) == [(0x20, 0x60), (0x10, 0x80)]
    loop = sc.inner_loop(rows)                         # both loops hold the HMMA; the inner (smaller) one is taken
    assert (loop["target"], loop["branch"], loop["instructions"]) == ("0x20", "0x60", 5)
    assert loop["classes"] == {"load_global": 1, "logic_shift": 1, "select": 1, "mma": 1, "sync_ctrl": 1}
    pw = sc.per_weight(loop, lanes=128, weights_per_iteration=2048)
    assert pw["lane_instr_per_weight"] == 5 * 128 / 2048 and abs(sum(pw["by_class"].values()) - 5 / 16) < 1e-12


def _loop_with(mmas: int, op: str = "HMMA.1688.F32.TF32") -> dict:
    lines = [f"        /*{16 * i:04x}*/   {op} R8, R10, R12, R8 ;" for i in range(mmas)]
    lines.append(f"        /*{16 * mmas:04x}*/   @P0 BRA 0x0 ;")
    return sc.inner_loop(sc.parse_sass("\n".join(lines)))


def test_the_unroll_guard_counts_whole_iterations_and_scales():
    """K25's served tile (16 x 32 x 64, 4 warps, m16n8k8) issues 8 MMAs per warp per K iteration. A body holding 16
    is a 2-fold unroll and counts against twice the weights. A body holding 12 is no whole number of iterations, and
    is refused."""
    tile = dict(block_m=16, block_n=32, kc=64, num_warps=4)
    assert sc.unroll_factor(_loop_with(8), **tile) == 1
    assert sc.unroll_factor(_loop_with(16), **tile) == 2
    with pytest.raises(ValueError, match="whole number of iterations"):
        sc.unroll_factor(_loop_with(12), **tile)
    assert sc.unroll_factor(_loop_with(4, "HMMA.16816.F32.BF16"), **tile) == 1    # m16n8k16: half the MMAs
    with pytest.raises(ValueError, match="unknown MMA"):
        sc.unroll_factor(_loop_with(8, "HMMA.884.F16.F16"), **tile)


def test_an_unrolled_loop_counts_per_weight_against_all_its_iterations():
    loop = _loop_with(16)                               # 17 instructions, 2 iterations of 2048 weights on 128 lanes
    k = sc.unroll_factor(loop, block_m=16, block_n=32, kc=64, num_warps=4)
    assert sc.per_weight(loop, 128, 2048 * k)["lane_instr_per_weight"] == 17 * 128 / 4096


def test_a_record_without_a_tile_never_claims_the_unroll_check():
    """No ``tile``: the guard did not run, so ``unroll`` is None (the CLI prints 'unchecked'), not a silent 1. With
    the tile, the same 2-fold body reads unroll 2 and counts against both iterations."""
    rows = sc.parse_sass("\n".join([f"        /*{16 * i:04x}*/   HMMA.1688.F32.TF32 R8, R10, R12, R8 ;"
                                    for i in range(16)] + ["        /*0100*/   @P0 BRA 0x0 ;"]))
    bare = sc.loop_record(rows, num_warps=4, weights_per_iteration=2048)
    assert bare["unroll"] is None and bare["lane_instr_per_weight"] == 17 * 128 / 2048      # unchecked: 1 assumed
    checked = sc.loop_record(rows, num_warps=4, weights_per_iteration=2048, tile=(16, 32, 64))
    assert checked["unroll"] == 2 and checked["lane_instr_per_weight"] == 17 * 128 / 4096


def test_a_branch_with_its_predicate_as_an_operand_closes_a_loop():
    """Newer ptxas (12.9.86, triton 3.8.0) closes K25's sm_120 K loop with ``BRA.U UP0, 0xb70``: the uniform predicate
    is an operand, not an ``@UP0`` guard. Missed, the loop is not found at all."""
    rows = sc.parse_sass("""
        /*0000*/                   S2R R0, SR_CTAID.X ;
        /*0010*/                   HMMA.1688.F32.TF32 R4, R56, R36, R4 ;
        /*0020*/                   BRA.U UP0, 0x10 ;
        /*0030*/                   BRA.U !UP1, 0x50 ;
        /*0040*/                   HMMA.1688.F32.TF32 R4, R56, R36, R4 ;
        /*0050*/                   EXIT ;
""")
    assert sc.backward_branches(rows) == [(0x10, 0x20)]      # the forward branch to 0x50 is no loop
    assert sc.inner_loop(rows)["instructions"] == 2


def test_counts_compare_only_within_one_toolchain():
    a = {"triton": "3.4.0", "ptxas": "12.8.93", "lane_instr_per_weight": 17.0}
    b = {"triton": "3.4.0", "ptxas": "12.8.93", "lane_instr_per_weight": 24.125}
    assert sc.compare(a, b) == 17.0 / 24.125
    for other in ({**b, "triton": "3.8.0"}, {**b, "ptxas": "12.9.86"}, {**b, "ptxas": None}):
        with pytest.raises(ValueError, match="across toolchains"):
            sc.compare(a, other)
    with pytest.raises(ValueError, match="across toolchains"):           # unknown on both sides is still unknown
        sc.compare({**a, "ptxas": None}, {**b, "ptxas": None})


def test_no_mma_means_no_inner_loop():
    assert sc.inner_loop(sc.parse_sass("        /*0000*/  @P0 BRA 0x0 ;\n")) is None


class _Fn:
    arg_names = list(sc.K25_SIGNATURE)


class _Backend:
    @staticmethod
    def parse_attr(desc):
        return [["tt.divisibility", 16]] if "D" in desc else []


@pytest.mark.parametrize("shape,not_divisible", [("gate_up", set()), ("down", {"stride_an"})])
def test_the_jit_specialization_marks_every_pointer_and_every_16_divisible_int(shape, not_divisible):
    """down's stride_an = K / 64 = 12 is the one served int NOT divisible by 16; everything else is."""
    sig, _consts, values, _wpi = sc.k25_served(shape)
    attrs = sc.jit_attrs(_Fn, sig, values, _Backend)
    named = {_Fn.arg_names[i] for (i,) in attrs}
    pointers = {n for n, t in sig.items() if t.startswith("*")}
    assert pointers <= named and len(pointers) == 10
    assert named - pointers == set(values) - not_divisible
    assert all(v == [["tt.divisibility", 16]] for v in attrs.values())


def test_an_int_equal_to_one_must_be_a_constexpr():
    with pytest.raises(ValueError, match="constexpr"):
        sc.jit_attrs(_Fn, {"N": "i32"}, {"N": 1}, _Backend)


def _nvidia_backend():
    try:
        import triton  # noqa: F401
        from triton.tools.disasm import path_to_cuobjdump
        return os.path.exists(path_to_cuobjdump())
    except Exception:
        return False


def _pinned_toolchain():
    try:
        import triton
        return triton.__version__ == "3.4.0"
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def _k25_compiles(arch: int) -> tuple:
    """K25's gate_up at the served plan for ``sm_<arch>``: (served, bare), the compile with and without the hints."""
    sys.path.insert(0, str(KERNEL))
    import nf4_smallm
    fn = nf4_smallm._gemm_nf4_grouped_smallm
    sig, consts, values, wpi = sc.k25_served("gate_up")
    kw = dict(arch=arch, num_warps=4, num_stages=3, weights_per_iteration=wpi, tile=(16, 32, 64))
    return (sc.compile_and_count(fn, sig, consts, values, **kw),
            sc.compile_and_count(fn, sig, consts, values, specialize=False, **kw))


@pytest.mark.skipif(not _nvidia_backend(), reason="triton's NVIDIA backend (cuobjdump) is not installed")
@pytest.mark.parametrize("arch", [86, 120])
def test_k25_at_the_served_plan_loads_vectorized_only_when_specialized(arch):
    """STRUCTURE, on any triton: the served instruction shape and count, one K iteration per loop body, and
    vectorized weight loads under the JIT's hints (scalar without them). It holds under triton 3.4.0 and 3.8.0
    alike. The counts do not, so they are pinned only at the served stack's toolchain (the next test)."""
    import triton
    served, bare = _k25_compiles(arch)
    s_ops, b_ops = served["inner_loop"]["opcodes"], bare["inner_loop"]["opcodes"]
    assert s_ops.get("HMMA.1688.F32.TF32") == 8 == b_ops.get("HMMA.1688.F32.TF32")     # the served instruction shape
    assert "LDG.E.U8" not in s_ops, "the served compile scalarized its weight loads: the JIT's hints are missing"
    assert "LDG.E.U8" in b_ops                          # ...and without them they do scalarize
    assert served["unroll"] == 1 == bare["unroll"]         # one K iteration per loop body: no silent unroll
    assert served["triton"] == bare["triton"] == triton.__version__ and served["ptxas"] == bare["ptxas"]


@pytest.mark.skipif(not _nvidia_backend(), reason="triton's NVIDIA backend (cuobjdump) is not installed")
@pytest.mark.skipif(not _pinned_toolchain(), reason="the counts are pinned at triton 3.4.0, the served stack's; "
                                                    "another toolchain moves them, so only the structure is checked")
def test_k25_counts_at_the_pinned_toolchain():
    """The recorded counts, at triton 3.4.0 and its bundled ptxas 12.8.93 only. There the hints LOWER the count (under
    3.8.0 they raise it: 36.2 served against 32.6 bare on sm_120)."""
    for arch, body in ((86, 402), (120, 386)):
        served, bare = _k25_compiles(arch)
        assert served["ptxas"] == "12.8.93"
        assert sc.compare(served, bare) < 1
        assert served["inner_loop"]["instructions"] == body and served["lane_instr_per_weight"] == body * 128 / 2048
