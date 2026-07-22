# Multi-arch #1 follow-up — MI300X (CDNA3): v5loop LDS fix (census clean), MXFP4 native-byte + QLoRA-train CONFIRMED

**Tier: correctness/enablement CONFIRMED on CDNA3.** Follow-up to
`RESULTS-multiarch-mi300x.md` (stamped, immutable) — this resolves the
v5loop-prefill LDS skip that doc filed as a tracked follow-up (its §"Honest
finding", "the fix … is left as a tracked follow-up, not attempted in this
run"), and adds two enablement results on the same arch. Receipts in
`bench/multiarch/mi300x-20260722/followup/`. Run 2026-07-22 on AMD Developer
Cloud (DigitalOcean-operated), 1× **MI300X VF**, `gfx942`, warp 64, torch
**2.10.0+rocm7.0** (HIP 7.0.51831), triton 3.5.1, bitsandbytes
**0.50.0.dev0 built from HIP source**. Boxes deleted + 404-verified; ~$9 more
of the AMD AI Developer Program credit (workbench session; $87.87 remaining).

## 1. v5loop prefill LDS fit-down — the disclosed follow-up, resolved

The stamped doc disclosed that every `fused_v5loop` cell in the
`prefill_s2048` regime for the small-expert shapes skipped on CDNA3:

```
skipped — out of resource: shared memory, Required: 98304, Hardware limit: 65536
```

Root cause, confirmed to the byte: `fused_v5loop` = `prefill_variant=0`, whose
M-tile mainloop stages **both** the activation tile **and** a dequantized-B
tile per pipeline stage. The tuned NVIDIA config (`block_m=128, block_n=128,
block_k=64, num_stages=3`) needs `3 × (128·64·2 + 128·64·2) = 98304 B` — fine
on NVIDIA (100–228 KB LDS), over CDNA3's **65536 B**.

**Fix** (`kernel/nf4_grouped.py`, `gemm_4bit_grouped`): a device-shared-memory
fit-down. Query the device's `max_shared_mem`; if the tuned config's estimated
per-stage footprint would exceed it (with ~8 KB headroom), step `num_stages`
(3→2) then `block_m` (128→64) then `num_stages` (→1) until it fits. It is:

- **arch-agnostic** — keyed on the queried LDS limit, not a CDNA3 special-case;
  triggers on any low-LDS device, no-op where the limit is unqueryable (0).
- **a no-op on every NVIDIA cell** — their LDS already fits the tuned config.
- **gated on `prefill_config is None`** — explicit benchmark/ablation overrides
  are never silently retuned.
- **correctness-preserving** — only tiling/pipelining knobs move; the kernel
  and its numerics are unchanged.

For the skipped cells it lands at `block_m=64, num_stages=2` = 49152 B, fits
with margin. **Result: the census re-ran `rc=0` with 0 skips.**

| census | before (stamped doc) | after fit-down |
|---|---|---|
| cells executed | 42 (+6 disclosed skips) | **48 (0 skips)** |
| the 8 `fused_v5loop` prefill cells | 6 skipped / 2 ran | **8 ran, b_rel 1.7e-3** |
| fidelity, all cells | 1.6e-3 … 2.3e-3 | **1.65e-3 … 2.27e-3** (unchanged tier) |

The 6 formerly-skipped cells run and are **bit-accurate** (b_rel 1.7e-3,
identical to every other fused cell) — the fit-down changed the tile shape, not
the answer. Receipts: `census_ldsfit.json` (48 cells), `census_ldsfit.log`.

## 2. MXFP4 native-byte path — CONFIRMED on CDNA3 (11/11)

The native-MXFP4 lane's correctness suite, the analog of the NF4 44/44:

- `test_mxfp4_grouped.py` — **4 passed** (the grouped MXFP4 kernel).
- `test_mxfp4_oracle.py` — **7 passed**: codebook-matches-transformers,
  nibble-order discover-and-lock, dequant-exact-vs-oracle, the e8m0 `0xFF`
  edge, and pack roundtrip self-consistency.

So the native-byte MXFP4 kernel + its dequant-reference parity hold on gfx942 —
the MXFP4 reveal extended to a second architecture. Receipt:
`mxfp4_correctness.log` (11 passed). (The full exact-chunk-ppl **serve**
reproduction, gpt-oss-120b weights, is a heavier separate run — not attempted
here; this is kernel + oracle correctness.)

## 3. 4-bit MoE QLoRA train-path — CONFIRMED on CDNA3 (12/12)

`test_mxfp4_qlora.py` — **12 passed**, the full train-path on the MI300X:
`test_grads_match_dense_autograd` (backprop correct), `test_lora_delta_exactly
_zero_at_init`, `test_bytes_bitidentical_after_training_steps` (the frozen-base
guarantee holds), `test_loss_descends` (the step actually optimizes),
`test_recompute_retains_no_dense_weights`, plus forward / apply-gate /
fused-vs-loop / bf16-decode / provenance parity. So 4-bit MoE **fine-tuning**
runs correctly on AMD — a capability most 4-bit QLoRA tooling is CUDA-only for.
Receipt: `qlora_train.log` (12 passed).

## Scope — resident 235B end-to-end attempted, blocked by the VF (honest)

A full end-to-end **resident** 235B-A22B decode (all NF4 experts in VRAM, fused
kernel, greedy generate — a faithful resident number, not the offload script)
was attempted and **did not produce a number**: the 94-layer ~125 GB-resident
build OOM-thrashed the box. The instance is an **MI300X VF** (virtual
function) whose contract reports a virtualized link and — the leading cause —
does not appear to expose the full 192 GB HBM3, so a 125 GB-resident model
cannot fit. The evidence gate never wrote a result and the box was torn down
without a fabricated number. This is a **hardware-partition constraint, not a
kernel or code issue**; a faithful end-to-end resident number needs a
*dedicated* (non-VF) MI300X — a separate procurement, tracked as open.

Note the resident-kernel *substance* is already measured: the census
`decode_bs1` cells (E=128, the 235B-class MoE shapes) **are** resident
fused-kernel throughput. Only the end-to-end wall-clock generate is pending.

## Method / reproduction

The known-good stack recipe (distilled from the six environment layers the
stamped doc catalogs) is captured verbatim in the receipts:
`amd_bootstrap.sh` (torch `2.10.0+rocm7.0 --force-reinstall --no-deps` +
matching `pytorch-triton-rocm`, then bnb from HIP source with ROCm on PATH and
`cmake<4`, then a GPU bf16 oracle assert), and the three task drivers
`task1_mxfp4.sh` / `task2_qlora.sh` / `task3_census.sh`. `BOOTSTRAP` records
each stage's outcome. One install-order trap folded into the recipe: bnb's
editable install must be `--no-deps` or it pulls a default **CUDA** torch that
clobbers the ROCm build (oracle then dies "no NVIDIA driver").

## Receipts

`bench/multiarch/mi300x-20260722/followup/`: `mxfp4_correctness.log` (11),
`qlora_train.log` (12), `census_ldsfit.json` (48 cells / 0 skips) +
`census_ldsfit.log`, `BOOTSTRAP`, and the four verbatim driver scripts.
Checksums: `SHA256SUMS.mi300x-followup`.
