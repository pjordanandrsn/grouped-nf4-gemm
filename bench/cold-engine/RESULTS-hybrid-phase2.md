# RESULTS — Hybrid tier Phase 2: AVX-512 grouped GEMV, gate G2

Kernels: `gnf4_native/cpu_kernels.c` (compile-at-first-use) +
`kernel/cpu_grouped.py` wrappers, at the commit carrying this file. Bench:
`phase2_gemv_bench.py`. Gate fixed in advance: grouped expert GEMV at
decode shapes ≥70% of the box's measured `B_dram` (STREAM triad), bit-exact
vs reference.

## Verdict

| G2 criterion | result | verdict |
|---|---|---|
| bit-exact vs reference at fp32, locked order | exact on every path tested: AVX-512 hand path (Zen 4 metal, Zen 5), portable path (AVX2 box, CI class), NF4 + MXFP4 incl. 0xFF/e=0 scale edge cases, thread-count invariant | **PASS** |
| sustained ≥70% of B_dram at decode shapes | best **134.0 GB/s = 55.5%** of triad (Zen 4 metal); **126.2 = 55.2%** (Zen 5 VM) | **MISS** |

A miss reported exactly like a pass. The exactness half — the invariant the
whole one-artifact story rests on — holds everywhere; the bandwidth half is
instruction-bound with the causes measured and the fix path named.

## Measured (NF4, qwen3-ish gate_up [1536×2048], 24L×96E arena ≫ L3, k=8, T=1)

| box | triad | best GEMV | fraction | per-core |
|---|---|---|---|---|
| Zen 4 metal (64c Genoa, container) | 241.3 | **134.0** (64t) | 55.5% | 6.3 GB/s |
| Zen 5 VM (Turin, 48 vcpu slice) | 228.7 | **126.2** (24t) | 55.2% | **12.3 GB/s** |

MXFP4 same shape, Zen 5 VM: best 106.4 GB/s. Grouped-scatter (pure-read
ceiling) on the same boxes: 335.8 / 309.9 GB/s — the read roofline sits
well above triad; the gate denominator stays triad per the PREREG.

## What the diagnostics established (in order run)

1. **rows-scaling test**: T=4 lowered GB/s ~3.5× — wall time scales with
   row count, so the kernel is ALU-bound, nowhere near the weight stream.
2. **Disassembly**: the generic cell's runtime-indexed `accv[8][4]` forced
   every accumulator to the stack (264 stack moves + 126 scalar adds in the
   hot function). Fixed with a T=1 specialization using named register
   accumulators; exactness retained.
3. **Fold-scale**: `(LUT×scale)[code]` per block replaces per-element scale
   multiplies — identical fp32 values by construction, +12% (119.7→134.0).
4. **Per-ISA arithmetic closes**: Zen 4 double-pumps 512-bit ops (256-bit
   datapaths) → measured 6.3 GB/s/core matches the µop count; Zen 5's true
   512-bit ports double it to 12.3 — the same binary, confirming the
   instruction-bound model. TR PRO 7000 (the directive's reference class)
   is Zen 4 and double-pumps too.
5. **Scaling wall on the Zen 5 instrument is unattributable from inside**:
   the rented "whole machine" is a VM — every vCPU reports one flat L3
   spanning 0–47 (CCD topology hidden, per-CCD pinning meaningless) and
   `AnonHugePages: 0` (THP never granted; the arena ran on 4 KiB pages).
   `OMP_WAIT_POLICY=active` flattened but did not lift the ~126 plateau.

## The named fix path (Phase-3-adjacent, then re-measure G2)

- Pair two output columns per pass (shares every activation load, halves
  µops per weight byte — the standard GEMV blocking, ~1.3–1.5×).
- VBMI `vpermb` nibble expansion (fewer port-5 µops per 32 packed bytes).
- A real-metal instrument with visible CCDs and grantable hugepages —
  Latitude bare metal is the right tool here per the provider playbook —
  plus per-CCD tiling once topology is real.
- The OpenMP per-call region becomes the Phase-3 executor's persistent
  pool (decode-latency work, out of Phase-2 scope by design).

With Zen 5 per-core at 12.3 and the pairing gain, ~16 well-placed cores
reach the 160–195 GB/s gate band; the blocker is placement/instrument, not
arithmetic. That claim is a projection and is labeled as one.

## Locked-order note (the FMA caveat, as required)

The summation tree is mul+add with four round-robin vector accumulators
and a fixed combine (spec: `ordered_gemv_ref`). **GCC's GNU-mode default
`-ffp-contract=fast` silently fused mul+add into FMA and broke exactness
on the first run** — the build now pins `-ffp-contract=off`. Agreement
with the repo's torch oracles (`dequant_ref` + matmul) is tolerance-level
only, because torch matmul defines no summation order.

## G1 re-measure (carried item — closed this round)

With the native single-call epilogue + dense router gemv from this module
wired into e4b's `cpu_router` (and the cache-warm retargeted at the buffer
the native path actually reads — caught live when the first re-measure
didn't move): replica round trip **p50 36.5 µs / p99 50.8** (was
45.1/63.6; bar 35/100). p99 PASS, p50 now −4% from the bar (was −29%).
Real-model host segment halved (54.4 → ~29 µs), which also resolves most
of Phase 1's "unexplained model-arm math". Receipts:
`experts4bit-qlora/bench/cpu-router/g1r_{replica,model}.json`.

## Receipts (this directory, `phase2-receipts/`)

`g2_nf4_qwen.json` (Zen4 pre-fix), `g2_nf4_qwen_zen4.json` (Zen4 final),
`g2_nf4_qwen_zen5.json` / `_zen5b.json` (Zen5 sweeps), `g2_mx_qwen_zen5.json`,
`calib_p2.json` / `calib_zen5.json` (per-box triad/scatter denominators).
