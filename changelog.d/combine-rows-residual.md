### `combine_rows(..., residual=)` adds the decoder layer's residual in the combine's epilogue (experts4bit-qlora#1313)

- **What.** `int4_b32.combine_rows` takes an optional `residual` (bf16, `[T, H]`). It rounds the combine to bf16,
  widens both, adds them in fp32 and rounds to nearest even, which is how torch adds two bf16 tensors. Without
  `residual` nothing changes.
- **Why.** experts4bit-qlora's decoder layer adds the MoE output to its residual in a separate launch each layer. With
  `residual=` the combine does it. This is lane P127's Phase 2 (item c), the layer-local form.
- **Bitwise.** The result is bitwise `combine_rows(dn, w, k) + residual`. A silicon test checks this at 1 and 16 rows,
  three widths (one not a multiple of the block) and two residual scales. The interpreter skips it, because its bf16
  cast does not round to nearest. A second test checks the residual's shape and dtype.
