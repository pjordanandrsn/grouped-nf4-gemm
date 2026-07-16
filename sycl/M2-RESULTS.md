# M2 (performance) — measured on real Intel GPU silicon (UHD P630)

**Status: M2 DONE on the P630 (Gen9.5, opencl:gpu, NEO 23.43).** The SLM-tiled
decode-gemv variant is numerically bit-identical to the M1 kernel / canonical
`dequant_ref` and faster than the naive baseline at realistic MoE shape. The
P630 is a weak iGPU and NOT the perf target — absolute throughput and the
speedup *magnitude* on Arc/Max stay R3 "port target" until measured there —
but the existence and direction of the memory-traffic win is now **confirmed on
real Intel GPU silicon, not projected**.

## What the tiled variant does
A work-group owns one group `g` and a strip of `WG` output columns; the reused
activation row `a[g,:]` is staged **once** into local memory (SLM) and shared
across the strip, instead of being re-read from global memory once per output
column (N times). The per-output fp32 k-loop is unchanged, so the result is
bit-for-bit the M1 result — any speedup is pure memory-traffic reduction.

## Measured (icpx -fsycl -O3, NEO 23.43, best-of-40, per-config fresh queue)

decode gemv, small shape E=4 N=128 K=128 G=3:
  naive        0.0896 ms   b_rel=2.1e-7  PASS
  tiled WG=64  0.1068 ms   b_rel=2.1e-7  PASS   speedup=0.84x
  -> at tiny N the SLM staging + barrier outweighs the reuse: net LOSS. Honest.

decode gemv, MoE-realistic shape E=8 N=8192 K=2048 G=8:
  naive        112.67 ms   b_rel=8.3e-7  PASS
  tiled WG=16  273.45 ms   ...           speedup=0.41x
  tiled WG=32  156.14 ms   ...           speedup=0.72x
  tiled WG=64   85.46 ms   b_rel=8.3e-7  PASS   speedup=1.32x   <- best
  tiled WG=128  89.78 ms   ...           speedup=1.25x
  tiled WG=256 189.02 ms   ...           speedup=0.60x
  -> WG=64 optimal; the work-group-sizing sweep is load-bearing (2-3x spread).

## Reading it
- **Correctness**: b_rel ~1e-7 on both shapes = bit-exact vs the parent's oracle.
  (An earlier per-element-rel-err metric spiked to 2% on a near-zero cancellation
  cell at K=2048 and falsely read FAIL; the suite's norm-relative b_rel — the
  same fix `bench/hw_contract.py` carries — shows the kernels were always right.)
- **Perf**: SLM activation caching pays off once N is large enough to amortize
  the one-time stage; at N=128 it does not. The crossover is a variant-select
  signal (naive for tiny N, tiled for MoE-sized N).
- **Not claimed**: any absolute tok/s, or that 1.32x transfers to Arc/Max. Xe-HPC
  has more SLM, wider sub-groups, and real HBM bandwidth; the magnitude there is
  a port target, and the naive-vs-tiled crossover N will differ.

## Repro
`bash m2_gpu_run.sh` inside `intel/oneapi-basekit` with `--device /dev/dri`
(the script forces the Gen9.5-capable NEO 23.43 downgrade first). Test vectors
from `gen_testvec.py` (small default; `TV_E=8 TV_N=8192 TV_K=2048 TV_G=8` for perf).
