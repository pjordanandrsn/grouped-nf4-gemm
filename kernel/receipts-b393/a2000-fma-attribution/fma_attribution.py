# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""B393 follow-up diagnostic, NOT the lane's reading: is combine_rows' difference from the
separately-rounded slot-order sum exactly a fused multiply-add?

Lane B393 (kernel/PREREG-b393-combine-reduce-bitwise.md) found on the RTX 5090 that combine_rows is
bit-equal neither to the torch chain nor to the fp32 sum taken in slot order with each multiply and each
add rounded separately. Its registered attribution stops at "the kernel's own arithmetic differs too, for
example a fused multiply-add". This script tests that one candidate directly: it emulates

    acc = fma(fp32(dn[j]), w[j], acc)   for j in slot order,   then bf16(acc)

on the CPU and compares it bit-for-bit with the kernel on the census's own combine cases (inputs are drawn
on the CPU, so they are the census's inputs). It also dumps the kernel's PTX and counts fma.rn.f32.

The fma is emulated in fp64: a bf16 value times an fp32 value is exact in fp64 (8 + 24 significant bits),
and the add rounds once in fp64 and once to fp32. That double rounding can differ from a true fma only when
the fp64 sum lands exactly on an fp32 halfway point, so a handful of mismatches would need that checked
before being read as "not fma"; zero mismatches needs no such caveat.

Run on any CUDA box from a clone:  python fma_attribution.py out.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import b393_bitwise_census as C  # noqa: E402


def fma_sequential(dn: torch.Tensor, w: torch.Tensor, T: int, k: int, H: int) -> torch.Tensor:
    x = dn.to(torch.float64).view(T, k, H)             # bf16 -> exact in fp64
    wk = w.to(torch.float64).view(T, k, 1)             # fp32 -> exact in fp64
    acc = torch.zeros(T, H, dtype=torch.float32)
    for j in range(k):
        acc = (x[:, j] * wk[:, j] + acc.to(torch.float64)).to(torch.float32)
    return acc.to(torch.bfloat16)


def sha(t: torch.Tensor) -> str:
    """sha256 of the tensor's bytes, so outputs from two boxes compare exactly without shipping them."""
    return hashlib.sha256(t.contiguous().cpu().view(torch.int16).numpy().tobytes()).hexdigest()


def main(out_path: str) -> int:
    if not torch.cuda.is_available():
        print("needs CUDA", file=sys.stderr)
        return 3
    from int4_b32 import combine_rows
    rows = []
    for fam, k, H in C.FAMILIES:
        for T in C.ROWS:
            for seed in C.SEEDS:
                for heavy in (False, True):
                    dn, w = C.combine_inputs(T, k, H, seed, heavy, "cpu")
                    dg, wg = dn.cuda(), w.cuda()
                    got = combine_rows(dg, wg, k).cpu()
                    chain_gpu = C.combine_chain(dg, wg, T, k, H).cpu()
                    seq_gpu = C.combine_sequential(dg, wg, T, k, H).cpu()
                    ref_fma = fma_sequential(dn, w, T, k, H)
                    ref_sep = C.combine_sequential(dn, w, T, k, H)
                    rows.append({"family": fam, "k": k, "H": H, "T": T, "seed": seed, "heavy": heavy,
                                 "vs_fma_sequential": C.ulp_stats(got, ref_fma),
                                 "vs_sequential": C.ulp_stats(got, ref_sep),
                                 "gpu_sequential_equals_cpu": bool(torch.equal(seq_gpu, ref_sep)),
                                 "sha256": {"fused": sha(got), "chain_gpu": sha(chain_gpu),
                                            "sequential": sha(ref_sep), "fma_sequential": sha(ref_fma)}})
    ptx = []
    cache = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    for p in glob.glob(os.path.join(cache, "**", "_combine_rows*.ptx"), recursive=True):
        s = open(p).read()
        keep = os.path.join(os.path.dirname(os.path.abspath(out_path)), "ptx")
        os.makedirs(keep, exist_ok=True)
        open(os.path.join(keep, os.path.relpath(p, cache).split(os.sep)[0][:12] + "_combine_rows.ptx"), "w").write(s)
        ptx.append({"file": os.path.relpath(p, cache), "fma.rn.f32": s.count("fma.rn.f32"),
                    "mul.rn.f32": s.count("mul.rn.f32"), "mul.f32": s.count("mul.f32"),
                    "add.f32": s.count("add.f32")})
    fma_eq = sum(r["vs_fma_sequential"]["equal"] for r in rows)
    sep_eq = sum(r["vs_sequential"]["equal"] for r in rows)
    diff = sum(r["vs_fma_sequential"]["n_diff"] for r in rows)
    import triton
    summary = {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
               "torch": torch.__version__, "triton": triton.__version__,
               "gpu_sequential_equals_cpu": sum(r["gpu_sequential_equals_cpu"] for r in rows),
               "cases": len(rows), "bit_equal_vs_fma_sequential": fma_eq, "elements_differing_vs_fma": diff,
               "bit_equal_vs_separate_sequential": sep_eq, "ptx": ptx}
    json.dump({"summary": summary, "cases": rows}, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "fma_attribution.json"))
