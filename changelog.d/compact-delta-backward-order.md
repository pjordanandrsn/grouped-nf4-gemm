### `NF4_QLORA_COMPACT_DELTA=1`: the backward never holds the rebuilt block and its grad together

`_CompactPaddedDelta.backward` now computes the adapters' product (`gAt = x^T @ gh`) before the block's grad (`gx = gh @ A`), and frees
the rebuilt zero-padded block `x` between them. The two `[G * W, K]` buffers, in the adapter dtype, are never live together, so the
backward's transient is one block smaller. Every op, operand, layout and dtype is unchanged, so every output and gradient is the same
bytes. The test pins this against the old order, for fp32 and bf16 adapters. With fp32 adapters at Qwen3-30B-A3B's field recipe the two
were 0.811 GB each and set experts4bit-qlora's training peak once the padded delta and the LM loss were lean (TC1 amendment 75).
- `kernel/test_compact_delta.py` adds three tests. The first checks bit-identity with the old order, kept as a test-local reference. The
  second is a `TorchDispatchMode` census that the old order holds two `[G * W, K]` buffers and the new order one; it fails when the
  order is reverted. The third, CUDA-only, checks that the backward peak drops by about a block.
