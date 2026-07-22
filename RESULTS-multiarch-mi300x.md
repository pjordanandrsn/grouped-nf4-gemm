# Multi-arch confirmatory #1 — AMD MI300X (CDNA3 / gfx942): correctness CONFIRMED, resident compute census

**Tier: correctness CONFIRMED on CDNA3; compute census MEASURED (resident).
This does NOT confirm a streaming projection row — see scope below.** Receipts
in `bench/multiarch/mi300x-20260722/`. Run 2026-07-22 on AMD Developer Cloud
(DigitalOcean-operated), 1× **MI300X VF**, `gfx942:sramecc+:xnack-`, 304 CU,
**warp width 64**, torch **2.10.0+rocm7.0** (HIP 7.0.51831), triton 3.5.1,
bitsandbytes **0.50.0.dev0 built from HIP source** (`-DCOMPUTE_BACKEND=hip`).
~$3 of the AMD AI Developer Program credit; box deleted + 404-verified.

This is the **first architecture beyond NVIDIA** on which the grouped-NF4
kernel is measured rather than projected. Per `PROTOCOL-multiarch.md` the
correctness gate precedes any perf number; it passed.

## Correctness gate — PASS (PROTOCOL §"Correctness gate", bound: max rel-err < 1e-2)

- **HW contract** (`mi300_contract.log`): 5/5 PASS — gemv + m-tile paths,
  b_rel 1.58e-3 … 1.73e-3 vs the fp64 oracle.
- **Property suite** (`mi300_suite.log`): **44/44 PASS** in 78.6 s — the full
  `test_nf4_grouped.py`, warp-64 and all, against a real bnb-NF4 oracle (the
  HIP-compiled `libbitsandbytes_rocm70.so`, not a mock). This includes
  `TestDecodeExactness::test_decode_matches_bnb_exactly` (`torch.equal`, not
  `allclose`) and every SplitK / prefill-config / census-shape test.
- **Census fidelity** (`mi300_census.json`): all 42 executed cells report
  `b_rel_vs_fp64` in **1.6e-3 … 2.3e-3** — the *same* error magnitude as the
  NVIDIA census runs. The dequant path carries the ~2.2e-3 tier and the fused
  path the ~1.7e-3 tier on both vendors; the nibble order and blockwise absmax
  reproduce bit-for-bit on CDNA3.

Correctness on gfx942 is therefore established at the parent's fidelity bound.

## Scope — why this is NOT a streaming-projection confirmation

`PROJECTIONS-multiarch.md` files MI300X under **"resident, not streaming
(noted, not headlined)"**: 192 GB HBM3 holds a 235B-NF4 model resident, so the
streaming (`link_GBps / bytes_per_token`) model does not apply below 192 GB.
Two independent reasons this run cannot grade a streaming row:

1. It is a **VF (virtual function)** instance — the contract records
   `pcie_linksta: "Speed 2.5GT/s, Width x1"`, a virtualized/passthrough link
   report, not a real host↔device PCIe path. No streaming-waterfall number
   can be honestly taken from it.
2. The phase-1 census here is a **resident GEMM compute census** (weights
   resident, per-op kernel throughput), which is what characterizes the kernel
   on the arch — orthogonal to the discrete-GPU streaming thesis.

So: correctness = confirmed on CDNA3; streaming-decode projection for AMD
discrete GPUs (RDNA4 rows) remains open and needs a real gen5-×16 host.

## Resident compute census (per-op kernel throughput, decode_bs1 + prefill_s2048)

Selected cells (`mi300_census.json`, iters=20, median ms → tok/s); the finding
is the **shape-dependent ordering**, not the absolute microbenchmark rates:

| model (proj) | regime | dequant_grouped | fused_nf4 | fused_v5loop |
|---|---|---|---|---|
| Qwen3-30B (down) | decode | 28.1k | **95.6k** | 94.6k |
| Gemma-4 (down) | decode | 27.2k | **74.1k** | 73.8k |
| OLMoE (down) | decode | 28.5k | **82.3k** | 81.9k |
| gpt-oss-120b (gate_up) | decode | **26.7k** | 12.6k | 10.1k |
| gpt-oss-120b (down) | decode | **27.7k** | 20.6k | 20.6k |

- **The fused kernel's decode win is shape-dependent on CDNA3.** For the
  small-expert MoEs (E=64/128, the down/gate_up projections) fused is ~2–3×
  the per-expert dequant path; for **gpt-oss-120b (E=128, k=4)** the ordering
  **inverts** — `dequant_grouped` wins decode. The same inversion shows on
  NVIDIA for this shape, so it is a shape property, not an AMD regression.
- Prefill (`prefill_s2048`) full numbers in the receipts; fused leads on most
  shapes there.

## Honest finding — `fused_v5loop` prefill exceeds the CDNA3 LDS budget (6 skipped cells)

Every `fused_v5loop` cell in the **`prefill_s2048`** regime for the
small-expert shapes (OLMoE, Qwen3-30B, Gemma-4) reported:

```
status: skipped — out of resource: shared memory,
Required: 98304, Hardware limit: 65536
```

The v5 register-LUT **prefill** mainloop's block config requests **96 KB** of
shared memory; **MI300X's LDS limit is 64 KB** per workgroup (vs the 96–228 KB
available on recent NVIDIA archs the config was tuned against). This is a
**block-config portability gap, not a correctness failure** — the decode
v5loop path (smaller tiles) ran fine on every shape, and the gpt-oss E=128
prefill shape did *not* skip. Per PROTOCOL's **no-tune clause**, it is reported
at full volume here rather than re-rolled; the fix (a CDNA3-specific
`BLOCK_*`/`num_stages` for the v5loop prefill config, gated on
`shared_mem ≤ 64 KB`) is left as a tracked follow-up, not attempted in this
run. dequant_grouped and fused_nf4 prefill cover the regime meanwhile.

## Harness portability fix (in this PR)

`bench/phase1/harness.py` hard-coded `nvidia-smi` to capture the driver
version into the run's `env` metadata — a `FileNotFoundError` that is **fatal
before any cell runs** on a non-NVIDIA box (it killed the first census here).
Replaced with a portable `_driver_version()` that tries `nvidia-smi`, falls
back to `rocm-smi`, and treats a missing tool as empty metadata — never fatal.
The census re-ran green with it (recorded driver `6.19.14` via `rocm-smi`).

## Method / reproduction

Six environment layers stood between "rented MI300X" and "coherent stack"
(none in the kernel): a provisioning-time ssh interceptor, Ubuntu-24.04
PEP-668 pip refusal, a cwd source-tree phantom import of bnb, ROCm off the
default PATH, a cmake-4-vs-ROCm config break, and a torch-swap transaction
aborted by a Debian-owned `typing_extensions`. The working recipe is captured
verbatim in the receipts: `driver-mi300_run.sh` (fingerprint → HW contract →
bnb HIP source build → suite → census) and
`driver-coherent-rocm70-rebuild.sh` (the `torch==2.10.0+rocm7.0`
`--force-reinstall --no-deps` swap + HIP-source bnb that made torch, triton,
and bnb all agree on ROCm 7.0). `MI300_STATE` records every stage's rc.

## Receipts

`bench/multiarch/mi300x-20260722/`: `MI300_STATE`, `mi300_contract.log`,
`mi300_suite.log` (44/44), `mi300_census.json` (48 cells: 42 ok + 6 disclosed
skips), `mi300_census.log`, and the two driver scripts. Checksums:
`SHA256SUMS.mi300x`.
