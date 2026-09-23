# PREREG B393 — are `combine_rows` and `reduce_partials` bitwise equal to the torch chains they replace? (#393)

Registered 2026-09-23, before any run. Lane B393; the work item is grouped-nf4-gemm#393. Owner authorization:
"go ahead with the #703 rental and the issues from the other session" (Jordan, 2026-09-23, chat), within
the standing caps.

## Question

Both kernels replaced a torch chain with one launch, and both are tested to a tolerance
(`max|d| <= max|ref| * 2**-7`, `kernel/test_int4_b32.py:436` and `:452`), not with `torch.equal`:

- `combine_rows(dn, w, k)` (`kernel/int4_b32.py:744`) replaced
  `(dn.to(fp32) * w[:, None]).view(T, k, H).sum(dim=1).to(bf16)`. That is the chain experts4bit-qlora runs
  when `E4B_FUSE_COMBINE=0`. experts4bit-qlora runs the kernel on every MoE layer by default, and its call site
  says the fused path takes "the same order and roundings as the chain below". This module's docstring says
  the sum is "fp32 in slot order, as the torch chain's is".
- `reduce_partials(part, sk, R, N)` (`:268`) replaced `part.reshape(sk, R, N).sum(0).to(bf16)`. Its own
  test already expects "the fp32 sum order differs".

Is either kernel bitwise equal to its chain at the served shapes, on silicon? If not, by how much, and why?

## Instrument

`kernel/b393_bitwise_census.py`, at the commit the lane installs: the merge of the PR that adds this file.
`kernel/test_b393_census_helpers.py` checks the instrument on CPU in CI: the bf16 ULP distance, the verbatim
chains, and slot order in the sequential reading.

- **Cases, `combine_rows`.** Every served family's own `(top_k, hidden)` from its checkpoint config: Qwen3
  8/2048, OLMoE 8/2048, Granite 8/1536, Mixtral 2/4096, gpt-oss 4/2880, Gemma-4 8/2816. Each at T = 1, 16,
  17 and 64 rows, with 3 seeds and a heavy-tailed input variant: 144 cases. The inputs are bf16 `dn` and fp32
  softmax-normalised `w`, the dtypes experts4bit-qlora passes.
- **Cases, `reduce_partials`.** sk ∈ {2, 3, 4, 8, 16}, R ∈ {1, 4, 16}, N ∈ {768, 1536, 2048, 2816, 2880, 4096},
  with 3 seeds: 270 cases.
- **Three readings per case.**
  - `vs_chain`: the fused output against the chain above, verbatim.
  - `vs_sequential`: the fused output against the same fp32 terms summed strictly in slot order, with each
    multiply and each add rounded separately.
  - `chain_vs_sequential`: the chain against that sequential sum.

  Per element, each reading records bit equality and the bf16 ULP distance.
- **Attribution.** If fused equals sequential but not the chain, the difference is torch's reduction order. If
  fused differs from both, the kernel's own arithmetic differs too, for example a fused multiply-add.

A dry run under `TRITON_INTERPRET=1` (`--device cpu`) executed all 414 cases. It is not a reading. The
interpreter showed about 50% of elements 1 ULP off, while the chain and the sequential sum agreed exactly,
which is the signature of the interpreter's own bf16 cast rounding. That is why this lane reads a GPU and
nothing else.

## Expectation (stated, not the decision)

- **`reduce_partials`:** differences in some elements, since its test comment already says the order differs.
- **`combine_rows`:** differences in some elements too. torch's non-innermost reduction is not strictly
  sequential, and Triton may contract `acc += x * w` into an FMA.

In both, 1 bf16 ULP at most, because two fp32 sums of the same terms round to bf16 within one step of each
other at these magnitudes. The attribution arm says which cause it is.

## Decision rule

For each kernel separately:

- **A — bitwise at every case.** Register a measured claim ("bitwise equal to the torch chain at the served
  shapes, RTX 5090, torch/Triton as recorded"). Add a GPU test asserting `torch.equal` over the census cases,
  and keep the docstring and the experts4bit-qlora comment, now measured. #393's "register it" branch.
- **B — differs, at most 1 bf16 ULP.** Register a measured reorder-class claim: max ULP, fraction of elements
  differing, and the attributed cause. Correct the gnf4 docstring and the experts4bit-qlora call-site comment
  to what was measured. Tighten `test_combine_rows_matches_torch` / `test_reduce_partials_matches_torch` from
  the `2**-7`-relative bound to "at most 1 bf16 ULP", which is strictly stronger. #393's other branch, sizing
  the effect end to end, goes to experts4bit-qlora#708's probe, with `E4B_FUSE_COMBINE=0` as the control arm.
  #393 then closes as answered: no bitwise contract, a measured bounded one instead.
- **C — more than 1 bf16 ULP anywhere.** A defect. It is filed and fixed before anything is claimed.

## Box and cost

- **Box:** one RTX 5090 on Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. There is
  no model and no download beyond the package.
- **Cost:** a 1 h guard at ≤ $0.66/h, so ≤ $0.66. Expected under 15 minutes.
- **Runner:** experts4bit-qlora `bench/b393/`, the B374 pattern. It installs gnf4 at `GNF4_SHA`, proves the
  installed module is the pinned cut, and runs the census from a work dir that resolves `int4_b32` to the
  installed package.
- **Receipts:** `census.json` (every case), `census.txt` (the summary), `versions.txt`, `forensics.txt` and the
  teardown proof, into `kernel/receipts-b393/5090/`.
- **Exit codes:** 10 dud box, 15 wrong class, 9 install/tripwire, 3 no CUDA, 30 no time. The verdict is read
  from the JSON, not from the exit code.
