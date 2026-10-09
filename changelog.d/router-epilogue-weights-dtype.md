### `router_epilogue(..., weights_dtype=)` stores the routing weights in the caller's dtype (experts4bit-qlora#1313)

- **What.** `int4_b32.router_epilogue` takes `weights_dtype` (fp32, bf16 or fp16; default fp32, unchanged). The
  kernel computes the weights in fp32 as before and rounds them on the store, to nearest even.
- **Why.** transformers' Qwen3-MoE router returns its weights in the model's dtype, so experts4bit-qlora casts the
  kernel's fp32 weights to bf16 in a separate launch each layer. With `weights_dtype=torch.bfloat16` the store does
  it. This is lane P127's Phase 2 (item b1).
- **Bitwise.** The stored weights are bitwise torch's `.to(weights_dtype)` of the fp32 weights, and `first` and the
  indices are unchanged. A silicon test checks this at 1 and 16 rows, in both selection modes, for bf16 and fp16.
  The interpreter skips it, because its bf16 cast does not round to nearest.
