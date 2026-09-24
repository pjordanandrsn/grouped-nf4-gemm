# B393 follow-up diagnostic — NAS RTX A2000 (sm_86). NOT the lane's reading.

Lane B393's reading is the RTX 5090 in [`../5090/`](../5090/). This diagnostic ran afterwards on the home NAS's
RTX A2000 12 GB (sm_86), at the merged cut (`69bb93f`), torch 2.8.0+cu128, Triton 3.4.0. It tests one candidate
explanation for why `combine_rows` differs from the separately-rounded slot-order sum: a fused multiply-add.

`fma_attribution.py` does three things.
- It re-draws the census's 144 `combine_rows` inputs on the CPU, so they are the census's inputs.
- It compares the kernel bit-for-bit with a CPU emulation of `acc = fma(fp32(dn[j]), w[j], acc)` in slot order. It
  also checks that the GPU's separately-rounded sequential sum equals the CPU's.
- It hashes every output, and keeps the compiled `_combine_rows` PTX in `ptx/`.

Result (`fma_attribution.json`):
- 144/144 bit-equal to the fma emulation, with 0 elements differing;
- 144/144 GPU-sequential equal to CPU-sequential;
- the PTX accumulates with `fma.rn.f32` only, in all three compiled variants (k = 2, 4, 8).

**Corrected 2026-09-24.** The paragraph below was superseded when the same script ran on an RTX 5090. On that box every output hash equals this A2000's, in 144/144 cases (see [`../5090-fma-attribution/`](../5090-fma-attribution/)). The paragraph is kept as the record of what was first inferred.

**What it does not show (superseded).** It does not show what the RTX 5090's kernel computes. Against the same deterministic
sequential reference, the 5090's per-case difference counts differ from sm_86's in at least 35 of 144 cases. So the
fma identity is exact on sm_86 and not bit-identical on sm_120. Running this same script on a 5090 box would record
the hashes and PTX needed to say what differs.
