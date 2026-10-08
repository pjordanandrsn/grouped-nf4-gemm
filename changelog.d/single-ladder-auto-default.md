### `NF4_QLORA_SINGLE_LADDER` defaults to `auto` (behaviour change for fp32 adapters)

- **What changed.** Unset (or empty) now means `auto`: the single padded block runs on the ladder's rungs exactly when the
  adapters are fp32. `0` turns it off; `1` takes it for any adapter dtype; other values are off, as before.
- **Why.** experts4bit-qlora TC1 amendments 70–72 read it at TC1's field recipe on Qwen3-30B-A3B and one RTX 5090. With
  fp32 adapters it stepped 0.797 of the time on a host-bound box and 1.031 on a GPU-bound one, within the registered 1.05;
  held-out loss was unchanged. bf16 adapters (e4b's default) are untouched (1.002), so `auto` never engages for them.
- **Who is affected.** Training with fp32 adapters on the lean single block: the batched products run at padded shapes,
  so values match the single block to rounding, not bit for bit. The way back is `NF4_QLORA_SINGLE_LADDER=0`.
- **Tests.** `kernel/test_single_ladder.py` pins unset == `auto` op for op on both dtypes, and empty as unset. The compact,
  lean and pad-bucket bit-identity tests set it to `0`, since they hold other paths to the single block.
