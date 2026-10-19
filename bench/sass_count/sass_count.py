# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Static issued-instruction count per weight in a kernel's inner K loop, from its SASS (no GPU, no timing).

A standing instrument for kernel PRs: compile a Triton kernel exactly as the JIT would for a served launch, for a named
GPU target, disassemble it, find the inner K loop, and report its lane instructions per weight by class. Every kernel
PR that changes a decode or a load path can quote one number per target.

**The JIT's specialization is part of the instrument.** A served launch specializes every tensor argument whose
data pointer is 16-byte aligned, and every int divisible by 16, to ``tt.divisibility = 16``
(``triton/runtime/jit.py`` ``specialize_impl`` -> the backend's ``parse_attr("D")``). Ints equal to 1 become constexprs.
Those hints are what let loads vectorize. Compiled without them, the same kernel scalarizes its loads and adds
per-element address arithmetic, an inflated count. An early K25 count read 34.5 lane instructions per weight that way,
against 24.1 as served. :func:`jit_attrs` builds the hints from the served call's own values, and the test fails if
they go missing.

**The loop.** The inner K loop is the backward branch whose body holds the MMAs. Per loop iteration, a program covers
``weights_per_iteration`` weights (BLOCK_N x KC for a weight-streaming GEMM) with ``lanes`` threads, so::

    lane instructions per weight = (instructions in the loop body) x lanes / weights_per_iteration

The count assumes one issue per scheduler per cycle. It is a static floor, not a timing.

usage::

    python sass_count.py --kernel-dir ../../kernel --config k25_served --out out/      # K25 at e4b's served plan
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys

#: SASS opcode classes, first match wins.
CLASSES = (
    ("mma", r"^(HMMA|IMMA|QMMA|OMMA|UTCHMMA|HGMMA)"),
    ("load_global", r"^(LDG|LDGSTS|LDGDEPBAR)"),
    ("shared", r"^(LDS|STS|LDSM)"),
    ("select", r"^(SEL|FSEL)"),
    ("logic_shift", r"^(LOP3|SHF|PRMT|LEA|IADD3|IADD|IMAD|ISETP|PLOP3|P2R|R2P|SGXT|BMSK|FLO|POPC)"),
    ("move", r"^(MOV|UMOV|S2R|CS2R|S2UR)"),
    ("fp", r"^(FMUL|FFMA|FADD|F2F|F2FP|I2F|F2I|HADD2|HMUL2|HFMA2|FMNMX)"),
    ("sync_ctrl", r"^(BAR|DEPBAR|BRA|WARPSYNC|BSYNC|BSSY|NOP|EXIT|ERRBAR|MEMBAR|SYNCS|YIELD)"),
)
_LINE = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b([^;]*);")
_BRA = re.compile(r"BRA[A-Z.]*\s+(?:`\()?(0x[0-9a-f]+)")


def classify(opcode: str) -> str:
    for name, pat in CLASSES:
        if re.match(pat, opcode):
            return name
    return "other"


def parse_sass(raw: str) -> list:
    """``[(address, opcode, text)]`` from raw ``cuobjdump -sass`` output (``/*0a10*/  [@P0] OPCODE ... ;``)."""
    out = []
    for ln in raw.splitlines():
        m = _LINE.match(ln)
        if m:
            out.append((int(m.group(1), 16), m.group(3), ln.strip()))
    return out


def backward_branches(rows: list) -> list:
    """``(target, branch)`` addresses of every branch whose target lies above it: the loops."""
    found = []
    for addr, op, text in rows:
        if op.startswith("BRA"):
            m = _BRA.search(text)
            if m and int(m.group(1), 16) < addr:
                found.append((int(m.group(1), 16), addr))
    return found


def inner_loop(rows: list) -> dict | None:
    """The inner K loop: of the loops whose body holds MMAs, the smallest (so an outer persistent loop over work items,
    if any, is not taken for it). ``None`` when no loop holds an MMA."""
    loops = []
    for a, b in backward_branches(rows):
        body = [r for r in rows if a <= r[0] <= b]
        ops = collections.Counter(classify(r[1]) for r in body)
        if ops.get("mma"):
            loops.append({"target": hex(a), "branch": hex(b), "instructions": len(body), "classes": dict(ops),
                          "opcodes": dict(collections.Counter(r[1] for r in body))})
    return min(loops, key=lambda x: x["instructions"]) if loops else None


def per_weight(loop: dict, lanes: int, weights_per_iteration: int) -> dict:
    scale = lanes / weights_per_iteration
    return {"lane_instr_per_weight": loop["instructions"] * scale,
            "by_class": {c: n * scale for c, n in sorted(loop["classes"].items())}}


#: MMA instruction shapes (m, n, k) by the opcode's shape suffix: HMMA.1688 (TF32, m16n8k8), HMMA.16816 (bf16 / fp16,
#: m16n8k16). An MMA the guard does not know is refused rather than guessed.
MMA_SHAPES = {"1688": (16, 8, 8), "16816": (16, 8, 16)}


def unroll_factor(loop: dict, *, block_m: int, block_n: int, kc: int, num_warps: int) -> int:
    """How many K iterations the picked loop body holds. One iteration of a ``block_m x block_n x kc`` tile on
    ``num_warps`` warps issues ``(block_m/m)(block_n/n)(kc/k) / num_warps`` MMAs per warp for the shape (m, n, k).
    ptxas can unroll the K loop, and then the body holds k times that, so dividing by one iteration's weights would
    overstate the count k-fold without an error. Returns k >= 1, or raises on an unknown MMA shape or a count that is
    not a whole multiple of one iteration's."""
    mmas = {op: n for op, n in loop["opcodes"].items() if classify(op) == "mma"}
    shapes = set()
    for op in mmas:
        m = re.match(r"^HMMA\.(\d+)", op)
        if not m or m.group(1) not in MMA_SHAPES:
            raise ValueError(f"unroll guard: unknown MMA {op!r}; known shapes {sorted(MMA_SHAPES)}")
        shapes.add(MMA_SHAPES[m.group(1)])
    if len(shapes) != 1:
        raise ValueError(f"unroll guard: expected one MMA shape in the loop, got {sorted(shapes)}")
    mm, nn, kk = shapes.pop()
    per_iter, rem = divmod((block_m // mm) * (block_n // nn) * (kc // kk), num_warps)
    total = sum(mmas.values())
    if rem or per_iter < 1 or total % per_iter:
        raise ValueError(f"unroll guard: {total} MMAs per warp in the loop is not a whole number of iterations of "
                         f"{(block_m // mm) * (block_n // nn) * (kc // kk)} / {num_warps} warps")
    return total // per_iter


def jit_attrs(fn, signature: dict, values: dict, backend) -> dict:
    """The served JIT launch's specialization, as ``ASTSource`` attrs: ``tt.divisibility = 16`` on every pointer
    argument (torch allocations are 16-byte aligned) and on every int in ``values`` divisible by 16, through the
    backend's own ``parse_attr("D")``. An int equal to 1 must be passed as a constexpr by the caller, as the JIT
    does."""
    for name, v in values.items():
        if isinstance(v, int) and v == 1:
            raise ValueError(f"{name} == 1: the JIT makes it a constexpr; pass it in constexprs")
    div = [n for n, t in signature.items() if t.startswith("*")]
    div += [n for n, v in values.items() if isinstance(v, int) and v % 16 == 0]
    return {(fn.arg_names.index(n),): backend.parse_attr("D") for n in div}


def compile_and_count(fn, signature: dict, constexprs: dict, values: dict, *, arch: int, num_warps: int,
                      num_stages: int, weights_per_iteration: int, out_dir: str | None = None, tag: str = "",
                      specialize: bool = True, tile: tuple | None = None) -> dict:
    """Compile ``fn`` for ``sm_<arch>`` as the JIT would (``specialize=False`` only to show what is lost), count its
    inner K loop, and return the record. Writes the SASS, PTX and cubin to ``out_dir`` when given.

    ``tile`` = (BLOCK_M, BLOCK_N, KC) turns on the unroll guard (:func:`unroll_factor`). A loop body that holds k K
    iterations counts against k x ``weights_per_iteration``, and k is recorded as ``unroll``. A count that is not a
    whole number of iterations is refused."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend
    from triton.tools.disasm import path_to_cuobjdump

    target = GPUTarget("cuda", arch, 32)
    attrs = jit_attrs(fn, signature, values, make_backend(target)) if specialize else {}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    k = triton.compile(src, target=target, options={"num_warps": num_warps, "num_stages": num_stages})
    cubin = k.asm["cubin"]
    tmp = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tmp = os.path.join(out_dir, f"{tag or fn.__name__}.sm{arch}.cubin")
    else:
        import tempfile
        tmp = tempfile.mkstemp(suffix=".cubin")[1]
    with open(tmp, "wb") as f:
        f.write(cubin)
    raw = subprocess.run([path_to_cuobjdump(), "-sass", tmp], capture_output=True, text=True, check=True).stdout
    if out_dir:
        with open(tmp.replace(".cubin", ".sass"), "w", encoding="utf-8", newline="\n") as f:
            f.write(raw)
        with open(tmp.replace(".cubin", ".ptx"), "w", encoding="utf-8", newline="\n") as f:
            f.write(k.asm["ptx"])
    else:
        os.remove(tmp)
    rows = parse_sass(raw)
    rec = {"arch": f"sm_{arch}", "triton": triton.__version__, "specialized": specialize,
           "divisible_args": sorted(signature_name(fn, i) for (i,) in attrs), "total_instructions": len(rows)}
    rec.update(loop_record(rows, num_warps=num_warps, weights_per_iteration=weights_per_iteration, tile=tile))
    return rec


def loop_record(rows: list, *, num_warps: int, weights_per_iteration: int, tile: tuple | None = None) -> dict:
    """The inner loop's record from parsed SASS. With ``tile`` = (BLOCK_M, BLOCK_N, KC) the unroll guard runs, and
    ``unroll`` is the k it found. Without it, ``unroll`` is ``None``: the record never claims a check it did not run,
    and the per-weight figure assumes one iteration per loop body, unchecked."""
    loop = inner_loop(rows)
    lanes = num_warps * 32
    rec = {"lanes": lanes, "weights_per_iteration": weights_per_iteration, "inner_loop": loop}
    if loop:
        unroll = (unroll_factor(loop, block_m=tile[0], block_n=tile[1], kc=tile[2], num_warps=num_warps)
                  if tile else None)
        rec["unroll"] = unroll
        rec.update(per_weight(loop, lanes, weights_per_iteration * (unroll or 1)))
    return rec


def signature_name(fn, index: int) -> str:
    return fn.arg_names[index]


# --------------------------------------------------------------------------------------------- the built-in configs --

#: K25 (``nf4_smallm._gemm_nf4_grouped_smallm``) at experts4bit-qlora's served plan (``hot_residency._K25_PLAN``:
#: BLOCK_N 32, KC 64, 4 warps, 3 stages, the select-tree decode, fp32 weights through TF32 MMA) on Qwen3-30B-A3B's
#: W16 shapes, through the lean route (gather, ``gather_div`` = top-k on gate_up, scatter).
K25_SIGNATURE = {
    "x_ptr": "*bf16", "ord_ptr": "*i64", "w_ptr": "*u8", "am_ptr": "*fp32", "lut_ptr": "*fp32",
    "row0_ptr": "*i32", "rows_ptr": "*i32", "grp_ptr": "*i32", "out_ptr": "*bf16",
    "N": "i32", "stride_we": "i32", "stride_wn": "i32", "stride_ae": "i32", "stride_an": "i32",
    "K": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "KC": "constexpr", "LUT": "constexpr",
    "GATHER": "constexpr", "DOT_BF16": "constexpr", "EVEN_K": "constexpr", "sct_ptr": "*i64",
    "SCATTER": "constexpr", "GDIV": "constexpr"}
K25_PLAN = {"BLOCK_N": 32, "KC": 64, "num_warps": 4, "num_stages": 3, "LUT": 3, "DOT_BF16": False}
K25_SHAPES = {"gate_up": {"N": 1536, "K": 2048, "GDIV": 8}, "down": {"N": 2048, "K": 768, "GDIV": 1}}


def k25_served(shape: str) -> tuple:
    """``(signature, constexprs, values, weights_per_iteration)`` for K25's served call at ``shape``."""
    sh = K25_SHAPES[shape]
    n, k = sh["N"], sh["K"]
    consts = {"K": k, "BLOCK_M": 16, "BLOCK_N": K25_PLAN["BLOCK_N"], "KC": K25_PLAN["KC"], "LUT": K25_PLAN["LUT"],
              "GATHER": True, "DOT_BF16": K25_PLAN["DOT_BF16"], "EVEN_K": k % K25_PLAN["KC"] == 0, "SCATTER": True,
              "GDIV": sh["GDIV"]}
    values = {"N": n, "stride_we": n * k // 2, "stride_wn": k // 2, "stride_ae": n * (k // 64), "stride_an": k // 64}
    return K25_SIGNATURE, consts, values, K25_PLAN["BLOCK_N"] * K25_PLAN["KC"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernel-dir", required=True, help="grouped-nf4-gemm's kernel/ directory")
    ap.add_argument("--config", default="k25_served", choices=("k25_served",))
    ap.add_argument("--module", default="nf4_smallm", help="the module holding the kernel (a K25 variant may differ)")
    ap.add_argument("--arch", type=int, nargs="+", default=[120, 86])
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-specialize", action="store_true", help="drop the JIT's hints (to show the inflation)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.abspath(a.kernel_dir))
    import importlib
    mod = importlib.import_module(a.module)
    fn = mod._gemm_nf4_grouped_smallm
    report = {"config": a.config, "module": a.module, "plan": K25_PLAN, "targets": {}}
    for arch in a.arch:
        for shape in K25_SHAPES:
            sig, consts, values, wpi = k25_served(shape)
            rec = compile_and_count(fn, sig, consts, values, arch=arch, num_warps=K25_PLAN["num_warps"],
                                    num_stages=K25_PLAN["num_stages"], weights_per_iteration=wpi, out_dir=a.out,
                                    tile=(16, K25_PLAN["BLOCK_N"], K25_PLAN["KC"]),
                                    tag=f"{a.module}.{shape}", specialize=not a.no_specialize)
            report["targets"][f"sm{arch}.{shape}"] = rec
            lpw = rec.get("lane_instr_per_weight")
            body = rec["inner_loop"] and rec["inner_loop"]["instructions"]
            unroll = "unchecked" if rec.get("unroll") is None else rec["unroll"]
            print(f"sm_{arch} {shape}: inner loop {body} instr (unroll {unroll}), "
                  f"{lpw and round(lpw, 2)} lane instr/weight", flush=True)
    if a.out:
        with open(os.path.join(a.out, "sass_count.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
