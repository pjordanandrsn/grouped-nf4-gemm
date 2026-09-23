# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane B393 (#393): are ``combine_rows`` and ``reduce_partials`` BITWISE equal to the torch chains they replace?

Both kernels were introduced as one-launch replacements of a torch chain, and both are tested to a tolerance
(``max|d| <= max|ref| * 2**-7``, kernel/test_int4_b32.py), not with ``torch.equal``. experts4bit-qlora runs
``combine_rows`` on every MoE layer by default (``E4B_FUSE_COMBINE=1``), and its call site says the fused path
takes "the same order and roundings as the chain below"; this module's docstring says the sum is "fp32 in slot
order, as the torch chain's is". Neither statement has been measured bit for bit. This census measures it.

For every case it compares the fused kernel with THE chain experts4bit-qlora runs when the kernel is off:

  combine_rows(dn, w, k)        vs  (dn.to(fp32) * w[:, None]).view(T, k, H).sum(dim=1).to(bf16)
  reduce_partials(part, sk, R, N) vs  part.reshape(sk, R, N).sum(0).to(bf16)

and, to say WHY when they differ, with a third reading: the same fp32 terms summed strictly in slot order
by separate torch ops (a rounded multiply, then one rounded add per slot). A fused result equal to that
sequential reading but not to the chain means the difference is torch's reduction ORDER; one that differs
from both means the kernel's own arithmetic (e.g. a fused multiply-add) differs as well.

Distances are reported in bf16 ULPs (the output dtype), per element, so "differs" is exact: equal bits, or
the number of representable bf16 values between them. Shapes are the served families' own (top-k, hidden)
and decode / verify / prefill row counts; inputs are seeded, with a heavy-tailed variant.

GPU only for a reading. ``python kernel/b393_bitwise_census.py --out census.json``. Exits 0 when every case
ran (the verdict is the JSON, read against kernel/PREREG-b393-combine-reduce-bitwise.md), 3 when CUDA is
absent. ``--device cpu`` under ``TRITON_INTERPRET=1`` is a DRY RUN of the code path only: the JSON records
the device, and an interpreter's arithmetic is not the GPU's, so such a file is never a reading.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: (model_type, top_k, hidden) of every family experts4bit-qlora serves, from each checkpoint's config.json.
FAMILIES = (
    ("qwen3_moe", 8, 2048),
    ("olmoe", 8, 2048),
    ("granitemoe", 8, 1536),
    ("mixtral", 2, 4096),
    ("gpt_oss", 4, 2880),
    ("gemma4_text", 8, 2816),
)
#: Rows per forward: B=1 decode, B=16 decode, a 17-row verify, a 64-row prefill chunk.
ROWS = (1, 16, 17, 64)
#: reduce_partials: the split-K factors the int4 / MXFP4 GEMV planners choose, the decode row counts, and
#: the expert projections' output widths across the families (down = hidden; gate_up = 2 * intermediate).
SPLITS = (2, 3, 4, 8, 16)
PART_ROWS = (1, 4, 16)
WIDTHS = (768, 1536, 2048, 2816, 2880, 4096)
SEEDS = (0, 1, 2)


def bf16_keys(x: torch.Tensor) -> torch.Tensor:
    """bf16 values as integers ordered like the values (+0 and -0 both 0), so ``|a - b|`` is the ULP distance."""
    u = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    mag = u & 0x7FFF
    return torch.where((u & 0x8000) != 0, -mag, mag)


def ulp_stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Bit equality and the bf16 ULP distance between two bf16 tensors of one shape."""
    if got.shape != ref.shape or got.dtype != torch.bfloat16 or ref.dtype != torch.bfloat16:
        raise ValueError(f"need two bf16 tensors of one shape, got {got.dtype}{tuple(got.shape)} / "
                         f"{ref.dtype}{tuple(ref.shape)}")
    d = (bf16_keys(got) - bf16_keys(ref)).abs()
    n_diff = int((d > 0).sum())
    return {"equal": bool(torch.equal(got.view(torch.int16), ref.view(torch.int16))), "n": got.numel(),
            "n_diff": n_diff, "max_ulp": int(d.max()) if d.numel() else 0,
            "max_abs": float((got.float() - ref.float()).abs().max()) if got.numel() else 0.0}


def combine_inputs(T: int, k: int, H: int, seed: int, heavy: bool, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 expert outputs ``[T*k, H]`` and fp32 routing weights ``[T*k]`` normalised per token, the dtypes
    experts4bit-qlora passes (``w = top_k_weights.reshape(-1).to(torch.float32)``)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dn = torch.randn(T * k, H, generator=g) * 0.5
    if heavy:                                    # a few large activations, the case rounding is most exposed in
        idx = torch.randperm(dn.numel(), generator=g)[: max(1, dn.numel() // 1000)]
        dn.view(-1)[idx] *= 64
    w = torch.softmax(torch.randn(T, k, generator=g), dim=-1).reshape(-1)
    return dn.to(dev, torch.bfloat16), w.to(dev, torch.float32)


def combine_chain(dn: torch.Tensor, w: torch.Tensor, T: int, k: int, H: int) -> torch.Tensor:
    """experts4bit-qlora's unfused path (hot_residency.py, E4B_FUSE_COMBINE=0), verbatim."""
    return (dn.to(torch.float32) * w[:, None]).view(T, k, H).sum(dim=1).to(torch.bfloat16)


def combine_sequential(dn: torch.Tensor, w: torch.Tensor, T: int, k: int, H: int) -> torch.Tensor:
    """The same fp32 terms, summed strictly in slot order, each multiply and add rounded separately."""
    terms = (dn.to(torch.float32) * w[:, None]).view(T, k, H)
    acc = torch.zeros(T, H, dtype=torch.float32, device=dn.device)
    for j in range(k):
        acc = acc + terms[:, j, :]
    return acc.to(torch.bfloat16)


def reduce_chain(part: torch.Tensor, sk: int, R: int, N: int) -> torch.Tensor:
    """The chain reduce_partials replaced (its own docstring), verbatim."""
    return part.reshape(sk, R, N).sum(0).to(torch.bfloat16)


def reduce_sequential(part: torch.Tensor, sk: int, R: int, N: int) -> torch.Tensor:
    p = part.reshape(sk, R, N)
    acc = torch.zeros(R, N, dtype=torch.float32, device=part.device)
    for s in range(sk):
        acc = acc + p[s]
    return acc.to(torch.bfloat16)


def run(out_path: str, device: str = "cuda") -> int:
    if device == "cuda" and not torch.cuda.is_available():
        print("B393: CUDA is not available; the census measures GPU arithmetic and cannot run here")
        return 3
    from int4_b32 import combine_rows, reduce_partials
    import triton
    dev = torch.device(device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    cases = []
    for fam, k, H in FAMILIES:
        for T in ROWS:
            for seed in SEEDS:
                for heavy in (False, True):
                    dn, w = combine_inputs(T, k, H, seed, heavy, dev)
                    got = combine_rows(dn, w, k)
                    chain, seq = combine_chain(dn, w, T, k, H), combine_sequential(dn, w, T, k, H)
                    sync()
                    cases.append({"kernel": "combine_rows", "family": fam, "k": k, "H": H, "T": T, "seed": seed,
                                  "heavy": heavy, "vs_chain": ulp_stats(got, chain), "vs_sequential": ulp_stats(got, seq),
                                  "chain_vs_sequential": ulp_stats(chain, seq)})
    for sk in SPLITS:
        for R in PART_ROWS:
            for N in WIDTHS:
                for seed in SEEDS:
                    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
                    part = (torch.randn(sk * R, N, generator=g) * 3).to(dev, torch.float32)
                    got = reduce_partials(part, sk, R, N)
                    chain, seq = reduce_chain(part, sk, R, N), reduce_sequential(part, sk, R, N)
                    sync()
                    cases.append({"kernel": "reduce_partials", "sk": sk, "R": R, "N": N, "seed": seed,
                                  "vs_chain": ulp_stats(got, chain), "vs_sequential": ulp_stats(got, seq),
                                  "chain_vs_sequential": ulp_stats(chain, seq)})
    summary = {}
    for kern in ("combine_rows", "reduce_partials"):
        cs = [c for c in cases if c["kernel"] == kern]
        for ref in ("vs_chain", "vs_sequential", "chain_vs_sequential"):
            summary[f"{kern}.{ref}"] = {
                "cases": len(cs), "cases_equal": sum(c[ref]["equal"] for c in cs),
                "elements": sum(c[ref]["n"] for c in cs), "elements_differing": sum(c[ref]["n_diff"] for c in cs),
                "max_ulp": max(c[ref]["max_ulp"] for c in cs)}
    doc = {"lane": "B393", "prereg": "kernel/PREREG-b393-combine-reduce-bitwise.md",
           "versions": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                        "python": platform.python_version()},
           "device": ({"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
                      if dev.type == "cuda" else {"name": "cpu (TRITON_INTERPRET dry run -- NOT a reading)"}),
           "summary": summary, "cases": cases}
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
    for key, s in summary.items():
        print(f"{key:40s} {s['cases_equal']:4d}/{s['cases']} cases bit-equal; {s['elements_differing']} of "
              f"{s['elements']} elements differ; max {s['max_ulp']} bf16 ulp")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="b393_census.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu: a dry run of the code path under TRITON_INTERPRET=1, never a reading")
    a = ap.parse_args(argv)
    return run(a.out, a.device)


if __name__ == "__main__":
    sys.exit(main())
