### `NF4_QLORA_SINGLE_LADDER=auto`: the single-block ladder exactly when the adapters are fp32 (opt-in value)

experts4bit-qlora TC1 amendment 70 read `NF4_QLORA_SINGLE_LADDER=1` at TC1's field recipe on a host-bound RTX 5090 box. Its
fp32-adapter arm stepped **0.797** of its time: `aten::bmm`'s CPU self time per call fell from 305 µs to 24.5 µs, for 4.1 % more
device time and 0.33 GB more peak. Its bf16-adapter arm stepped **1.015**: that `bmm` already took about 28 µs a call, so the
padding's 3.2 % of device time bought nothing. The per-new-shape host cost is cuBLAS's fp32 batched product's.

`auto` takes the ladder when the adapters, and so the padded block, are fp32, and is the single block op for op otherwise
(`kernel/test_single_ladder.py`). Unset is unchanged: off. experts4bit-qlora registers the A/B that decides whether `auto` becomes
the default.
