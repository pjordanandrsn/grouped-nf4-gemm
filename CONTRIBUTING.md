# Contributing

Thanks for looking. This kernel is small on purpose; the bar is receipts, not
volume.

## The one rule: claims carry receipts

Every performance or comparative claim in a PR must cite a committed receipt
(a results doc + its evidence JSONs), or be marked "measuring now." The README
examples are CI-executed (`test_readme_cpu_block.py`) so they cannot drift from
the API; keep that invariant — a new documented call gets a runnable block.

## Running the checks

```bash
cd kernel
pip install torch triton pytest numpy      # torch CPU wheel is fine for the CPU suite
python -m pytest test_readme_cpu_block.py test_cpu_refusal.py -q   # CPU-only, no GPU
TRITON_INTERPRET=1 python -m pytest test_interp_contract.py -q      # device-free semantics
```

The pure-torch `dequant_ref` is the CPU-checkable oracle; the fused
`gemm_4bit_grouped` requires CUDA (it says so, loudly, if called on CPU).

## Hardware we'd love help measuring

The cross-vendor projections (`PROJECTIONS-multiarch.md`) are stamped but
pre-silicon on non-NVIDIA parts. On-silicon confirmatory runs are the most
valuable contribution:

- **AMD (ROCm)** — MI2xx/MI3xx or Radeon. AMD's Developer Cloud offers free
  credits that fit a confirmatory run; a `hw_contract.py` pass + a census
  sweep is the ask.
- **Intel (XPU/SYCL)** — Arc / Max; the SYCL port (`sycl-m2`) is cross-vendor
  and wants absolute-magnitude numbers on real Arc silicon.
- **NVIDIA sm_120 (RTX 5090)** — the one gap in our own fleet (cloud stock,
  not code).

Open a "hardware-wanted" issue (template provided) with your device + the
receipt, and we'll fold it into the projections table with credit.

## Scope

Kernel-math changes need a fidelity receipt (the property suite must stay green
and the fused error must stay at/below the dequant baseline). Docs/test PRs are
welcome and low-ceremony.
