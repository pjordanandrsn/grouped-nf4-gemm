# PREREG B374 — the word-addressed NF4 decode routes past THEIR 2^31 boundary, on an RTX 5090 (#374)

Registered 2026-09-23, before any run. Lane B374. Owner authorization: "you have rental ability" (Jordan,
2026-09-23, chat), within the standing caps. The work item is grouped-nf4-gemm#374.

## Question

The wide-load and dot-pad NF4 decode routes are handed `B.view(torch.int32)`, so they multiply the expert id
by a stride counted in 32-bit words, and their offsets wrap at 2^31 words (8 GiB of packed bytes).
`kernel/test_expert_offset_boundary.py` builds its stacks on the 2^31-*byte* geometry, which exercises both
routes without ever reaching their boundary (#374). Dot-pad has been the **default** decode route on parts
with ≥ 160 SMs since M3. No test anywhere has put it past its boundary, and on a GPU the wide route has not
been put past its boundary either.

Does the shipped int64 promotion hold on both routes, observed past 2^31 words on silicon? And does the test
that says so fail when the promotion is gone?

## Instrument

`kernel/test_offset_boundary_words_gpu.py`, at the commit this lane installs (`GNF4_SHA`, the merge of the PR
that adds this file). Each case builds a real 16 GiB device buffer:

- the target expert (the first whose base is ≥ 2^31 words) sits past 8 GiB;
- a decoy tile sits at the address the int32 product wraps to, inside the same buffer;
- the result is checked against `dequant_ref` of the true tile, and the decoy must be distinguishable (the
  instrument half).

There are four cases, each asserting its route through `dispatch_counts()`:

| case | shape | stride | route |
|---|---|---|---|
| wide, split 1 | N=256, K=1024 | 1 MiB | `scalar` with `GNF4_GEMV_WIDE_LOADS=1` |
| wide, split 4 | N=256, K=1024 | 1 MiB | `scalar_splitk` with wide loads |
| dot-pad | N=1536, K=2048 (census gate_up) | 1.5 MiB (contiguous) | `dotpad` (the default) |
| dot-pad, split 4 | N=1536, K=2048 | 1.5 MiB | `dotpad_splitk` |

The lane runs the file twice, in separate processes:

1. **Shipped:** against the installed kernels.
2. **Unpromoted:** against a copy of the installed `nf4_grouped.py` with **all six** eid promotions removed.
   That is the four `eid = eid.to(tl.int64)` lines (M-tile, the two GEMVs, dgrad) and the two dot-pad loads,
   `eid = tl.load(eids_ptr + g).to(tl.int64)`. The runner asserts exactly six substitutions. Stripping only
   the four `eid.to` lines would leave dot-pad promoted and calibrate nothing.

The test file puts its own directory first on `sys.path`, so where it runs decides which `nf4_grouped` it
imports. The runner copies it out of the clone into two work dirs:

- `shipped/` holds only the test, so it resolves to the installed package;
- `unpromoted/` holds the test beside the stripped module, so it resolves to the copy.

A tripwire imports `nf4_grouped` the same way in each dir and records the file it resolved to. Running it from
the clone's `kernel/` would resolve to the clone and shadow the stripped copy.

## Predictions

- **P1 — the promotion holds.** In the shipped run all four cases PASS and none is skipped: each reads the
  true tile, rel < 5e-2, and the decoy rel is > 10× that.
- **P2 — the test has power.** In the unpromoted run all four cases FAIL. The expected form is the misread
  (`the offset wrapped`, reading the decoy). An illegal-address fault also counts as P2 holding, since the
  promotion is then load-bearing too, but it is recorded separately. A case that PASSES unpromoted refutes P2
  for that case: it did not straddle.

## Decision rule

- **P1 ∧ P2:** #374 is closed by option 2 (a GPU case that straddles 2^31 words).
  - `docs/KERNEL_CONTRACT.md` and the GPU suite's byte-geometry comment say where the word boundary is now
    observed on silicon.
  - A register row `gnf4.kernel.word-boundary-wide-dotpad.5090.<date>` (measured) records the read.
  - The receipts go to `kernel/receipts-b374/5090/`.
- **P1 ∧ ¬P2:** the affected cases are vacuous. They are fixed before anything is claimed, and the file does
  not count as coverage for those routes.
- **¬P1:** a shipped-kernel defect at the word boundary on that route. It is filed and fixed, and the default
  route (dot-pad) takes priority.

## Box and cost

- **Box:** one RTX 5090 (170 SMs, ≥ 17.5 GiB free needed per case) on Vast verified/secure, image
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- **Cost:** guard 1 h at ≤ $0.66/h, so ≤ $0.66 (expected ~10–15 min). A guard ≤ 1 h needs no proving run.
- **Runner:** experts4bit-qlora `bench/b374/`, the K18 pattern. It installs gnf4 at `GNF4_SHA`, clones the
  same sha, proves the installed module is the pinned cut, and runs the two passes from their own work dirs
  (above).
- **Receipts:** `logs/{shipped,unpromoted}.log`, `unpromote.diff`, `versions.txt`, `forensics.txt`,
  `summary.txt` and the teardown proof.
- **Exit codes:** 10 dud box, 15 wrong class, 9 install/tripwire, 31 P1 not met (the lane still runs the
  unpromoted pass and fetches both), 32 P2 not met, 30 no time.
