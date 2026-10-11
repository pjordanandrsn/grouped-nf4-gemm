### CI: an AMD compile-only check that fails on assembler errors (no GPU, no AMD support claim)

A new `hip-compile` job compiles the shipped kernels for gfx942, gfx950 and gfx1100 on a CPU runner, with triton
pinned at 3.4.0 and a copy-only stand-in for ROCm's `ld.lld` (`scripts/hip_lld_standin.sh`). The specializations come
from the interpreter suites themselves: `scripts/hip_harvest.py` records every kernel launch they make (argument types,
constexpr values, launch options), and `scripts/hip_compile_check.py` compiles up to eight per kernel, chosen so every
distinct value is compiled once. A compile fails on an exception OR on any `error:` line the AMD assembler writes,
captured at the file-descriptor level, because `triton.compile` returns normally when the assembler rejects an
instruction (PTX inline asm, for one).

Strict: an unlisted failure, a stale entry in `scripts/hip_compile_allow.json` (each entry names the refusal that keeps
that path off AMD; today the fused FP8 KV append on gfx1100, which has no fp8), and a kernel in
`scripts/hip_required.json` that compiled nowhere. Reporting only: shipped `@triton.jit` functions no specialization
reached. `scripts/hip_declared.json` holds specializations of kernels the interpreter suites never launch; it starts
empty. The check proves lowering and assembly, not numerics or that a code object loads.

Not covered here: the NF4/int4 kernels still run on ROCm when called directly; the SM-count `>= 160` launch
heuristics and the missing LDS fit-down are tuned for NVIDIA; the f32 attention's `tf32` input precision is unverified
on HIP.
