# B393 follow-up diagnostic — RTX 5090 (sm_120). NOT the lane's reading.

This is the same `fma_attribution.py` as [`../a2000-fma-attribution/`](../a2000-fma-attribution/), run as a guarded,
non-fatal side step of experts4bit-qlora lane P63's reading (`p63-5090-1`, 2026-09-24). Run details:
- one RTX 5090, driver 595.71.05, 475 W power limit;
- grouped-nf4-gemm 0.33.2 at `f88df1e`, torch 2.8.0+cu128, Triton 3.4.0;
- the same 144 `combine_rows` census inputs, drawn on the CPU.

Result (`fma_attribution.json`):
- **144/144 bit-equal to the fused-multiply-add slot-order sum**, with 0 elements differing;
- 144/144 GPU-sequential equal to CPU-sequential;
- the PTX (`ptx/`, three variants, k = 2, 4, 8) accumulates with `fma.rn.f32` only, with no separate `mul.f32` or
  `add.f32`.

**Against the A2000's diagnostic, by per-case sha256 of the output bytes:**
- the fused output, torch's chain (`(dn.float()*w).view(T,k,H).sum(1)`), the sequential sum and the FMA sum are
  **identical in 144/144 cases** on sm_86 and sm_120.

This retracts the cross-architecture statement first published with the B393 read. That statement was inferred from
count differences between the lane's census and the A2000 rehearsal, and hashes are direct. See the correction in
[`../../RESULTS-b393-combine-reduce-bitwise.md`](../../RESULTS-b393-combine-reduce-bitwise.md).
