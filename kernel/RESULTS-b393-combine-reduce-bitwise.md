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
- **`combine_rows` is exactly the slot-order sum with a fused multiply-add, on sm_86 and on sm_120.** On the RTX 5090 it differs
  from both the chain and the separately-rounded sequential sum. That was the registered attribution's stopping
  point: "the kernel's own arithmetic differs too". A follow-up diagnostic ran one candidate down on the NAS RTX A2000 (sm_86), and it is
  **not** this lane's reading; see [`receipts-b393/a2000-fma-attribution/`](receipts-b393/a2000-fma-attribution/).
  On the same 144 cases:
  - the kernel equals `acc = fma(fp32(dn[j]), w[j], acc)` summed in slot order, then cast to bf16, **bit-for-bit
    in 144 / 144 cases**, with 0 elements differing;
  - its PTX accumulates with `fma.rn.f32` only, with no separate `mul.f32` or `add.f32` in any of the three
    compiled variants.

  Triton contracts `acc += x * w` into one rounding (sm_86 here; sm_120 in the correction below, by the same diagnostic), while the chain and the sequential reference round
  the product first.

## What it does NOT say, including against the diagnostic

### Correction (2026-09-24): the cross-architecture statement is RETRACTED

**What this page first said.** "Neither the kernel's bits nor torch's own chain are the same across GPU
architectures." It said `combine_rows` on sm_120 is not bit-identical to sm_86's, and torch's `sum(dim=1)` neither.

**Why that was said.** The census on the lane's 5090 box and the A2000 rehearsal gave different per-case difference
counts against the same deterministic sequential reference: in 35 of 144 `combine_rows` cases, and in 41 for the chain.
From that I inferred that the fused output and the chain had different bits on the two architectures. **That was an
inference from counts, not a comparison of outputs.**

**What direct measurement shows.** The attribution diagnostic records a sha256 of every output. It was run again on a
second RTX 5090 (experts4bit-qlora lane P63's box, [`receipts-b393/5090-fma-attribution/`](receipts-b393/5090-fma-attribution/)),
and compared by hash with the A2000 run:
- **the fused output, torch's chain, the sequential sum and the FMA sum are bit-identical on sm_86 and sm_120 in
  144/144 cases**;
- `combine_rows` equals the FMA slot-order sum in 144/144 on sm_120 too, with PTX `fma.rn.f32` only.

**What stays unexplained.** The lane's own census run on its box (`b393-5090-1`, driver 595.91.07) disagrees with both
hash-checked runs in 37 of 144 cases. The second 5090 ran driver 595.71.05, so that is one box's run against two
others, not an architecture property. The census does not record output hashes, so which tensor differed on that box
cannot be recovered from its receipt. **The lane's decision is unaffected:** every result, on every box, is within the
fp32-summation bound, and outcome B does not depend on this.

**What replaces it.** `combine_rows` is the FMA-contracted slot-order sum, bit-exactly, on both architectures
measured, and it is not bitwise torch's chain. torch's chain itself was bit-identical across the two architectures at
these cases. The accuracy bound is still the contract, because it is the property #393 asked about. The kernel and
torch's chain are two different correct fp32 orders, not one order with noise.

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
