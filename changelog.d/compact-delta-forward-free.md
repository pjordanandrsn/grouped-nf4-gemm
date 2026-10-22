### Compact delta: free the padded block in the forward before the delta is built

`_CompactPaddedDelta.forward` (`NF4_QLORA_COMPACT_DELTA=1`) frees the zero-padded `[G * W, K]` block, and the gathered `A`, right after
the first product, and the gathered `B` after the second. The block is not saved for backward (backward rebuilds it), so freeing it
early changes no value. The block and the `[G, W, N]` delta are no longer live together, which takes about the smaller of the two off
the forward's transient. With fp32 adapters on Qwen3-30B-A3B's widest groups, that is about 0.6 GB. This is the forward twin of the
backward's order (#584).

- Tests: byte-equality of every output and gradient against the old forward (distinct and repeated ids, bf16 and fp32 adapters,
  scaling 1 and 2, host reuse on and off); a TorchDispatchMode census showing the block and the delta are never live together, where
  the old forward holds both; and on CUDA, the forward's peak drops by at least 0.9 × the smaller of the two.
