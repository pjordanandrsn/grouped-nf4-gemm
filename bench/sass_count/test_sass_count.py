# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The static instruction counter (``sass_count.py``): its parser and loop finder on a fixture, the JIT specialization
it must carry, and, where triton's NVIDIA backend is installed (it ships in the triton wheel, no GPU needed), K25
compiled at e4b's served plan.

The specialization is the point of the compile test. With the JIT's ``tt.divisibility`` hints, K25's weight bytes
load vectorized. Without them, they load as single bytes with per-element address arithmetic, the inflated count.
So the test fails if the hints go missing, not just if a number moves."""
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


@pytest.mark.skipif(not _nvidia_backend(), reason="triton's NVIDIA backend (cuobjdump) is not installed")
def test_k25_at_the_served_plan_loads_vectorized_only_when_specialized():
    import triton
    sys.path.insert(0, str(KERNEL))
    import nf4_smallm
    fn = nf4_smallm._gemm_nf4_grouped_smallm
    sig, consts, values, wpi = sc.k25_served("gate_up")
    kw = dict(arch=86, num_warps=4, num_stages=3, weights_per_iteration=wpi, tile=(16, 32, 64))
    served = sc.compile_and_count(fn, sig, consts, values, **kw)
    bare = sc.compile_and_count(fn, sig, consts, values, specialize=False, **kw)
    s_ops, b_ops = served["inner_loop"]["opcodes"], bare["inner_loop"]["opcodes"]
    assert s_ops.get("HMMA.1688.F32.TF32") == 8 == b_ops.get("HMMA.1688.F32.TF32")     # the served instruction shape
    assert "LDG.E.U8" not in s_ops, "the served compile scalarized its weight loads: the JIT's hints are missing"
    assert "LDG.E.U8" in b_ops                          # ...and without them they do scalarize
    assert served["lane_instr_per_weight"] < bare["lane_instr_per_weight"]
    assert served["unroll"] == 1 == bare["unroll"]         # one K iteration per loop body: no silent unroll
    if triton.__version__ == "3.4.0":                   # the served stack's version: the counts are pinned there
        assert served["inner_loop"]["instructions"] == 402 and served["lane_instr_per_weight"] == 402 * 128 / 2048
