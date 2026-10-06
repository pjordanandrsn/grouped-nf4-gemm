### `NF4_QLORA_PAD_BUCKETS=1`: the lean padded LoRA delta pads each bucket of similar-sized groups to its own widest, not every group to the hottest (opt-in; default unchanged)

- **Why.** An allocator census in experts4bit-qlora (TC1 amendment 47, row `e4b.train.memory-census.packed-4k.qwen3.5090.2026-10-06`) on an RTX 5090 (Qwen3-30B-A3B, packed rows of 4,096 real tokens,
  micro-batch 1, top-8 of 128 experts, fp32 adapters) put its training peak 7.47 GB above Unsloth's. 6.1 GB of that was the
  padded LoRA delta: the zero-padded input block `[G * widest, K]` and the `bmm` products.
  - Every non-empty group is padded to the hottest group's rows. At the down projection that is about 380,000 padded rows
    for 32,768 real ones (~11.6x).
  - `NF4_QLORA_COMPACT_DELTA=1` saves less for backward but still allocates the same-width block in its forward.
- **What.** Behind the new flag, the lean padded path (`lora_delta_grouped`'s default body) pads by buckets.
  - **The rule** (`_pad_buckets`, host only): sort the non-empty groups by rows (stable), then cut greedily from the
    narrowest. A bucket takes every next group with at most twice its first group's rows. So in every bucket the widest
    group has at most 2x the narrowest's rows, and the padded rows are at most twice the real rows. The single block's
    `G * max(rows)` is unbounded in the skew.
  - **The delta** (`_lora_delta_bucketed`): one zero-filled buffer of `sum(G_b * W_b)` rows in the adapters' dtype takes
    every real row in one `index_copy_`. It is split into the buckets' `[G_b * W_b, K]` blocks (views). The adapters are
    gathered once in bucket order and split the same way. Each bucket runs the two `bmm`s at its own width. The outputs are
    concatenated, and one `_GatherRows` gather through the flat index puts the real rows back in the caller's order. Only
    standard autograd ops (`index_copy_`, `split`, `bmm`, `cat`, the gathers), so gradients take the single block's route,
    bucket by bucket.
  - **The plan** is built from host facts as the single block's is: one batched `to_device_i32` upload (row counts, ids in
    bucket order, each group's first padded row) and `repeat_interleave(..., output_size=total)`. Nothing reads the device,
    so the path is capture-safe in the same way.
  - **`GNF4_HOST_REUSE`'s plan memo is extended, not bypassed.** A bucketed plan carries its bucket list and is stored under
    the single block's key plus a `"buckets"` tag, so neither path reads the other's plan. The down delta reuses the gate_up
    delta's bucketed plan.
  - **With `NF4_QLORA_COMPACT_DELTA=1` also set, buckets win:** the bucketed path never calls `_CompactPaddedDelta`.
  - **Unchanged:** the `auto` route rule still sizes the single block, so a call it sends to the loop still loops.
    `NF4_QLORA_LEAN_DELTA=0`'s body, the loop and `grouped_mm` are not bucketed.
  - **Counted:** `LORA_PATH_STATS["padded_bucketed"]` counts bucketed calls (instead of `padded`). Each bucketed call
    writes `LORA_PAD_WASTE`'s `last_rows_single` (`G * widest`), `last_rows_bucketed` and `last_buckets`; nothing else
    writes those three keys.
- **Unset or `0` is main, op for op.** A dispatch-mode log of every aten op, forward and backward, with its operands'
  shapes and dtypes, is identical to main's single-block body. Outputs and gradients are `torch.equal`, and the stats are
  the same. Checked against main's module across the padded, auto and loop routes, lean on and off, and compact on and off
  (288 cases), and pinned in the new test against a verbatim copy.
- **Values with the flag on: equal to rounding, not bit for bit.** Each real row gets the same arithmetic, but the `bmm`s
  run at other shapes, and a BLAS can pick its kernel, or split a reduction, by shape. On CPU (torch 2.13, arm64,
  Accelerate) the same rows through `bmm` at M = 60 and at M = 2 already round differently. Over the test's 7 routing cases,
  3 dtype pairs, scaling 1 and 2 and 5 seeds:
  - single-bucket cases are bit-identical;
  - with bf16 adapters, outputs and input gradients are bit-identical. Adapter gradients are too, except where one expert
    id repeats three times (the accumulation runs in bucket order): at most 2^-7.5 of the tensor's largest entry, about one
    bf16 ulp there;
  - with fp32 adapters, every fp32 tensor differs by at most 2^-21.3 of its largest entry (a few fp32 ulps). A bf16 input
    gradient differs by bf16 rounding, at most 2^-15.6 of its largest entry.
- **Padded rows**, from the bucket rule on seeded routings: 4,096 tokens, each sent to 8 distinct experts of 128 with
  Zipf(s) weights. These are row counts, not an allocator read.

  | s | real rows | single block, rows (x real) | bucketed, rows (x real) | buckets | fewer padded rows |
  |---|---|---|---|---|---|
  | 0.8 | 32,768 | 342,144 (10.44x) | 44,067 (1.345x) | 6 | 7.76x |
  | 1.0 | 32,768 | 440,960 (13.46x) | 44,375 (1.354x) | 6 | 9.94x |
  | 1.2 | 32,768 | 499,200 (15.23x) | 43,050 (1.314x) | 7 | 11.60x |

  At the census's shape the rule bounds the bucketed rows at 65,536, at least 5.8x fewer than its ~380,000.
- **Not read here.** No GPU was used. The training step's peak and time with the flag on are open, as is its launch cost:
  about two more `bmm`s per extra bucket, plus a `cat` each way. The CUDA-only tests below did not run.
- **Tests** (`kernel/test_lora_delta_pad_buckets.py`, wired into CI's GPU-labelled step). On CPU: 81 passed, 7 skipped
  (CUDA-only).
  - The plan follows the rule and tiles the padded rows.
  - Output and all three gradients match the single block to rounding: skewed, hot-expert, empty, single-group, and
    distinct and repeated ids; bf16/bf16, bf16/fp32 and fp32/fp32; scaling 1 and 2. They match the per-expert loop to
    the existing tolerance.
  - The Zipf(1) case above runs through both paths, and its padded rows equal the rule's.
  - Unset and `0` are main's single block, op for op.
  - `NF4_QLORA_LEAN_DELTA=0` ignores the flag, and with the compact flag buckets win.
  - CUDA only: device-tensor ids, the plan memo's hit on the gate_up -> down twin, and the bucketed forward and backward
    peaks below the single block's on a hot-expert case.
