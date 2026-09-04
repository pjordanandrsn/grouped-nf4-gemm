# Writing release notes

The first paragraph of a release note is for the person deciding whether to
upgrade. It says, in ordinary language and in this order:

1. **which problem changed** — what a user can now do, or what stopped being
   wrong (not which function was added);
2. **who is affected** — the model families, hardware and environments the
   change reaches, and the ones it does not;
3. **whether to upgrade** — "upgrade if …", "no action if …", and any floor
   on the related package.

Everything the project already does well follows unchanged after that
paragraph: the mechanism, the measurements with their receipts and tiers,
the corrections and retractions, the caveats and refused arms. A number in a
release note is a quote of an entry in `docs/claims.json`; the entry, not the
note, decides whether it is still current.

Historical release notes are not rewritten to this shape.

Example opening:

> **0.30.0.** Decode steps that run the int4-b32 or MXFP4 expert GEMVs spend
> less on their tail: `int4_b32.combine_rows` folds the top-k weighted combine
> into one launch, beside 0.29.0's `reduce_partials`, which reduces the fp32
> split-K partials and casts to bf16 in one launch where a multi-dispatch torch
> chain ran before; values are unchanged to bf16 rounding. Affects serving through
> experts4bit-qlora's collapsed decode forward on sm_80+ NVIDIA GPUs under
> Linux; nothing changes for prefill, training or the NF4 grouped GEMM.
> Upgrade if you serve the decode GEMVs through the consumer; no action
> otherwise. No new number: the register has no entry for these kernels, and
> the glue position they extend is claim `gnf4.serve.decode-glue-kernels`
> (measured-private), which does not yet include them.
