### `NF4_QLORA_COMPACT_BUCKETS=1`: the bucketed LoRA delta as one autograd node, same bytes, a fraction of the memory it holds (opt-in)

- The bucketed delta's autograd body keeps, per projection, the zero-filled padded block (each bucket's first `bmm` saves its slice)
  and the adapters'-dtype copy of the input (`index_copy_`'s backward keeps its source), and its forward builds every bucket's output
  before one `cat`. experts4bit-qlora's training-phase census (TC1 amendment 65, Qwen3-30B-A3B, packed 4,096-token rows, fp32
  adapters, RTX 5090) found those sites (`nf4_qlora.py` 672, 673 and 679) holding about 1.70 GB at the training peak, the largest
  group of e4b's transient excess over Unsloth.
- With the flag set, `_CompactBucketedDelta` writes each bucket's two `bmm`s into one preallocated output (no list, no `cat`), saves only
  the input (an alias), the adapters, the plan and the first `bmm`s' `[P, r]` output, and in backward rebuilds the block and issues
  per bucket the calls autograd's own backward would. Output and every gradient are `torch.equal` to the autograd body's on every
  bucket case and dtype pair, unique and repeated ids, on CPU and on an RTX A2000 (`kernel/test_compact_buckets.py`).
- On the A2000, at a 64-expert Zipf-like routing with fp32 adapters, the memory one call holds from its forward to its backward fell
  from 188.2 MiB to 1.8 MiB and its backward peak from 432.6 to 357.1 MiB (bf16 adapters: 56.5 to 1.0 MiB held, backward peak
  unchanged).
- Off by default, and independent of `NF4_QLORA_COMPACT_DELTA` (the single block's): the flag never moves the single block or the
  ladder's plans. `COMPACT_BUCKETS_STATS["calls"]` counts the calls it takes. A training-peak and speed A/B is for an
  experts4bit-qlora TC1 box to read before any default.
