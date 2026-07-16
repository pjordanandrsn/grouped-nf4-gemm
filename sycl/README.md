# SYCL / OpenVINO port of grouped-nf4-gemm

**Status: WIP. M0 done. M1 (decode-gemv correctness) PASS on CPU-SYCL AND on
real Intel GPU silicon (UHD P630, opencl:gpu) — max rel err 2.02e-5 vs the
canonical `dequant_ref` on both, identical to the last ulp (2026-07-15).
M2 (perf) needs Arc/Max-class hardware. Not yet published.**

**Gen9.x runtime recipe (the M0 blocker, solved):** the oneAPI basekit image
ships NEO 26.x from the intel-graphics PPA, which dropped Gen9.5 — the P630 is
invisible to it. Fix: remove the PPA source and force the Ubuntu-noble archive
driver, `apt-get install --allow-downgrades intel-opencl-icd=23.43.27642.40-1ubuntu3`,
after which the P630 enumerates as `[opencl:gpu]` and SYCL kernels run on it
(`sycl/m1_gpu_run.sh`-style; compile with icpx in the same container).

Portable SYCL implementation of the fused NF4 grouped GEMM, targeting Intel
GPUs (Arc, Data Center Max, and Gen-class iGPUs). Fills the program's biggest
gap — everything upstream is CUDA/Triton (NVIDIA-only) — and connects the
kernel to the OpenVINO practice. Same MIT license, same Cerin Amroth holder,
same fidelity-first discipline as the parent kernel: **correct before fast,**
validated against the parent's own numeric reference before any perf claim.

## The numeric contract (must match the Triton kernel bit-for-bit)

Ported from `kernel/nf4_grouped.py`. The decode gemv path is the first target
(primary product surface; simplest to get right):

- **Weights** `B[E, N, K/2]` uint8, row-major NF4-packed. Element `2j` is the
  **high** nibble of byte `j`, element `2j+1` the low nibble (`(byte>>4)&0xF`
  then `byte&0xF`).
- **Codebook** `NF4_LUT[16]` (fp32) — the exact bitsandbytes NF4 values, copied
  from the parent (`-1.0, -0.696…, …, 1.0`). Pinned in `nf4_common.hpp`.
- **Scale** `absmax[E, N, K/64]` fp32 — one blockwise scale per 64-element
  K-block (`BLOCKSIZE = 64`, locked).
- **Compute**: `out[g, n] = sum_k ( LUT[nibble(B[eid,n,k])] * absmax[eid,n,k/64] ) * a[g,k]`,
  **fp32 accumulate**, bf16 store. `eid = expert_ids[g]`; one token per group
  (M==1) on the decode path.
- **Validation**: a test vector (packed B, absmax, A, expected out) is dumped
  from the parent's `dequant_ref` + reference matmul (`gen_testvec.py`), so the
  SYCL output is checked against the *identical* numerics the Triton kernel is
  gated on. Fidelity gate: rel-err vs the fp32 reference below the parent's
  bound.

## Milestones

- **M0 — toolchain gate (in progress).** `hello_sycl.cpp`: does a containerized
  DPC++ (`icpx -fsycl`) build see the P630 as a SYCL *GPU* device and run a
  kernel on it? The P630 is Gen9.5; recent `intel-compute-runtime` moved Gen9.x
  to a legacy branch (the exact NEO-version regime the OpenVINO fixes live in),
  so this is a real gate, not a formality. Green ⇒ M1. Not-green ⇒ pin a
  Gen9.5-capable runtime, or fall back to CPU-SYCL for correctness + real Arc/Max
  hardware for GPU validation.
- **M1 — correctness.** SYCL decode gemv matching the test vector on the P630
  (naïve tiling; speed irrelevant). This is the milestone that proves the port.
- **M2 — performance.** Sub-group tiling, coalesced packed-byte loads,
  work-group sizing for the target GPU; the M-tile/prefill path.
- **M3 — OpenVINO integration.** Wire the kernel into the GPU plugin
  (kernel-selector / custom op) so it dispatches for NF4 MoE experts — the path
  that makes it usable in the toolkit Cerin Amroth already contributes to.

## Build (on an Intel-GPU host with oneAPI)

```
icpx -fsycl -O3 hello_sycl.cpp -o hello_sycl && ./hello_sycl
```

Test host = QNAP UHD P630 (`/dev/dri/renderD128`) via a oneAPI container;
correctness only. Perf validation needs Arc/Max-class Intel GPU (cloud, TBD).
