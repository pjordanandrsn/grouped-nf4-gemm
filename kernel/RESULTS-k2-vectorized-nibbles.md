# RESULTS — K2 vectorized nibbles: REFUTED-FOR-VARIANT (legacy default)

Run 2026-08-25 against `PREREG-k2-vectorized-nibbles.md` (#242), EPYC
9-series + RTX 5090 (the NUMA pre-gate refused a 7702 slice at
112.5 GB/s first; both boxes destroyed + verified zero). Receipts in
`receipts-k2-vecnib/`; verdict by `k2_verdict.py` (self-tested, 9
branches).

## Verdict table

| gate/bar | registered | measured | result |
|---|---|---|---|
| G-B bitwise (CUDA, both kernels) | exact | **PASS** (both cells) | PASS |
| G-B e2e tokens | exact | **127/127** | PASS |
| G0 (vec / legacy arms) | < 7.5% | 0.064% / 0.021% | PASS |
| H-K vec pair | ≤ 45 µs | **80.0 µs** (legacy 72.8) | FAIL |
| H-E graph step | ≤ 12.6 ms | 14.20 (legacy 13.89) | FAIL |
| GS B=16 | in band | 125.7 ms | PASS |

**⇒ REFUTED-FOR-VARIANT.** The construction claim was TRUE — bitwise
everywhere — and the performance hypothesis was FALSE: the vectorized
path is **10% slower** at the kernel (0.91×) and consistently slower
at the wall. This PR executes the registered consequence: legacy is
the default again (`_vec_loads` becomes opt-IN via
`GNF4_GEMV_VEC_LOADS=1`, kept for A/Bs on other parts), and the
interp CI tripwire is hardened to skip on compiled-mode sessions (it
ValueError'd on the CUDA box — the real gate lived in the bench's
CUDA bitwise assert, which is what passed).

## What the refutation teaches (the lane's next registration)

The duplicate-address byte loads were NOT the bandwidth ceiling: L2
absorbs pair-reads to the same address, and `tl.interleave`'s shuffle
lowering costs more than the halved load count saves. The ~18%-of-
roofline mystery at the winner configs is therefore UNEXPLAINED — and
per the program's core law, the next kernel registration must be an
**attribution stage first** (Nsight Compute metrics at the two census
cells: achieved bytes/sectors, warp stall reasons, occupancy) before
any further mainloop or packing-layout edit is registered. Evidence
before edits; the K2 cycle is what buying that lesson for one box-hour
looks like.

Single-stream state after K2: **74.3 tok/s stands** (K1's cert; the
legacy default this PR restores is exactly that configuration).
