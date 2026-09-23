# RESULTS B393 — `combine_rows` and `reduce_partials` are correct fp32 sums, not bitwise copies of the torch chains (#393)

Pre-registration: [`PREREG-b393-combine-reduce-bitwise.md`](PREREG-b393-combine-reduce-bitwise.md), merged in #396
(`69bb93f`) before the rental. Runner: experts4bit-qlora `bench/b393/`, merged in experts4bit-qlora#715 (`20bf934`).
Run **`b393-5090-1`**: one RTX 5090 (sm_120, 600 W, driver 595.91.07) on Vast verified/secure, instance 52285900.
Installed cut: grouped-nf4-gemm 0.33.2 at `69bb93f`. `int4_b32` resolved to site-packages, not the clone. torch
2.8.0+cu128, Triton 3.4.0. Actual cost **$0.0299**, teardown proven (`receipts-b393/5090/teardown-proof.json`).
Everything below is read from [`receipts-b393/5090/census.json`](receipts-b393/5090/census.json); nothing is hand-transcribed.

## The reading

| kernel | reading | cases bit-equal | elements differing | max bf16 ULP | within the fp32 bound | max bound ratio |
|---|---|---|---|---|---|---|
| `combine_rows` | vs the torch chain | 49 / 144 | 392 of 9,069,312 | 53 | fused 144 / 144 | 0.996 |
| | vs the slot-order sequential sum | 58 / 144 | 297 | 44 | chain 144 / 144 | 0.996 |
| | chain vs sequential | 80 / 144 | 246 | 64 | | |
| `reduce_partials` | vs the torch chain | 221 / 270 | 111 of 4,455,360 | 8 | fused 270 / 270 | 0.996 |
| | vs the slot-order sequential sum | **270 / 270** | **0** | 0 | chain 270 / 270 | 0.996 |
| | chain vs sequential | 221 / 270 | 111 | 8 | | |

## Decision: outcome B for both kernels

Neither kernel is bit-equal to its chain in every case, so outcome A is out. `fused_bound_ratio <= 1` holds in all 414
cases, so outcome C (a defect) is out. The rule's outcome B applies to each kernel: **it is a correct fp32 sum in
some order, and it carries an accuracy contract, not a bitwise one.**

The max ratio of 0.996 is the bf16 cast's own half-ULP at a value just above a power of two. It is the same maximum
the A2000 rehearsal showed. The torch chain reaches it too, so the fused kernels are exactly as accurate as the chains
they replaced.

The large ULP maxima (53, 44, 64) are where the terms cancel and the sum lands near zero. There, two correct fp32
orders round to bf16 values many ULPs apart. The prereg registered this before the run: ULPs are not bounded near
cancellation, so they are reported and not decided on.

## Attribution

- **`reduce_partials` adds in slot order, bit-exactly.** It equals the sequential sum in 270/270 cases, with 0
  elements differing. Its whole difference from the chain is torch's reduction order in `.sum(0)`.
- **`combine_rows` is, on sm_86 exactly, the slot-order sum with a fused multiply-add.** On the RTX 5090 it differs
  from both the chain and the separately-rounded sequential sum. That was the registered attribution's stopping
  point: "the kernel's own arithmetic differs too". A follow-up diagnostic ran one candidate down on the NAS RTX A2000 (sm_86), and it is
  **not** this lane's reading; see [`receipts-b393/a2000-fma-attribution/`](receipts-b393/a2000-fma-attribution/).
  On the same 144 cases:
  - the kernel equals `acc = fma(fp32(dn[j]), w[j], acc)` summed in slot order, then cast to bf16, **bit-for-bit
    in 144 / 144 cases**, with 0 elements differing;
  - its PTX accumulates with `fma.rn.f32` only, with no separate `mul.f32` or `add.f32` in any of the three
    compiled variants.

  On sm_86, Triton contracts `acc += x * w` into one rounding, while the chain and the sequential reference round
  the product first.

## What it does NOT say, including against the diagnostic

**Neither the kernel's bits nor torch's own chain are the same across GPU architectures.**
- **The comparison.** Against the same sequential reference, the per-case differences on the 5090 and on the A2000
  change in 35 of 144 `combine_rows` cases on the count of differing elements (37 counting the max ULP;
  `vs_sequential`), and in 41 of 144 for the torch chain itself (`chain_vs_sequential`). Equal counts do not prove
  equal bits, so these are lower bounds. No T=1 case changes, which is what its element count predicts.
- **Why the reference is fixed.** The sequential reference is elementwise IEEE arithmetic, so it does not change: on
  the A2000 its GPU and CPU outputs are bit-equal in 144/144.
- **What follows.** `combine_rows` on sm_120 is not bit-identical to `combine_rows` on sm_86, and torch's `sum(dim=1)`
  on sm_120 is not bit-identical to torch's on sm_86.
- **What stays open.** The FMA attribution is exact on sm_86. What sm_120's kernel computes in the elements where it
  differs is open. The diagnostic is written to answer it on the next 5090 box: it records per-case output hashes
  and keeps the PTX.

**This bears on any bitwise contract, not only this one.** A contract "bitwise equal to the torch chain" would be a
contract with a reference that is itself not bitwise-stable across architectures. The accuracy bound is
architecture-free: every result on both boxes is within it.

**`reduce_partials` is architecture-stable at these shapes.** Its three readings match the A2000 rehearsal case for
case: 270/270 equal to sequential on both, and the same 111 elements from the chain.

**Nothing here sizes the effect end to end.** That was #393's other branch, and the prereg hands it to
experts4bit-qlora#708's probe, with `E4B_FUSE_COMBINE=0` as the control arm.

## What changes because of this read

- **Two claims registered:** `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23` and
  `gnf4.kernel.reduce-partials-slot-order.5090.2026-09-23`. `gnf4.open.issues` drops #393.
- **The docstrings are corrected to what was measured.** `combine_rows` said "fp32 in slot order, as the torch chain's
  is". Neither half held: the kernel uses a fused multiply-add, and torch's chain is not slot order.
- **The tests move from a tolerance to the accuracy bound.** `test_combine_rows_matches_torch` and
  `test_reduce_partials_matches_torch` asserted `max|d| <= max|ref| * 2**-7`, one max-relative number for the whole
  tensor. They now assert the per-element bound this lane decided on, which is far tighter for every element whose
  magnitude is below the max. Under `TRITON_INTERPRET=1` the cast term is one full bf16 ULP instead of half, because
  the interpreter's bf16 cast does not round to nearest (the prereg's dry-run disclosure). Measured, not assumed: under
  the interpreter at the tests' own inputs, the worst element sits at 1.98 of the half-ULP bound (combine) and 1.98
  (reduce), and at 0.99 of the one-ULP bound. That is a truncating cast, just under one ULP. On the A2000 the tests
  pass at the half-ULP bound. `reduce_partials` also
  gains a `torch.equal` test against its slot-order sum, which this lane measured on silicon.
- **experts4bit-qlora's call site** says the fused path takes "the same order and roundings as the chain below". It is
  corrected in a separate PR against that repository.
- **#393 closes as answered:** no bitwise contract, and a measured accuracy contract instead.
