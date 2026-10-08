### The bucketed LoRA delta as one autograd node: same bytes, a fraction of the memory it holds (`_CompactBucketedDelta`)

- Bucketed LoRA-delta calls can run as one autograd node instead of the autograd body. It writes each bucket's two `bmm`s into one
  preallocated output, saves only the input, the adapters, the plan and the first `bmm`s' `[P, r]` output, and rebuilds the block in
  backward.
- Output and every gradient are `torch.equal` to the autograd body's, on CPU and on an RTX A2000 (`kernel/test_compact_buckets.py`).
- On the A2000 (64 experts, fp32 adapters) the memory one call holds from forward to backward fell from 188.2 to 1.8 MiB, and the
  backward peak from 432.6 to 357.1 MiB. It is now the default (`NF4_QLORA_COMPACT_BUCKETS=0` turns it off).
