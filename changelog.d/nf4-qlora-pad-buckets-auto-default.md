### Bucketed LoRA-delta padding is on by default as `auto` (experts4bit-qlora TC1 amendments 47-50)

- **What changes.** `NF4_QLORA_PAD_BUCKETS` unset now means `auto`: a grouped-LoRA delta call carrying at least 16,384 routed rows pads
  each bucket of similar-sized expert groups to its own widest group, and every smaller call keeps the single padded block, op for op.
  `0` keeps the single block everywhere; `1` buckets every call. `NF4_QLORA_PAD_BUCKETS_MIN_ROWS` still moves the gate.
- **Why, by experts4bit-qlora's registered rule (TC1 amendment 50).**
  - Amendment 47's allocator census put 6.1 GB of e4b's packed-row excess over Unsloth in the single block, which pads every expert to
    the hottest expert's rows (about 11.6x the routed rows at 4,096 tokens).
  - On packed 4,096-token rows, buckets stepped the fp32 arm at 0.893 of the single block's time with 4.29 GB off its peak, and the bf16
    arm at 0.933; held-out unchanged (amendment 48).
  - At the field recipe (calls of at most 9,040 routed rows) the gate never fired: 49,152 single-block calls per arm, with held-out and
    peak unchanged (amendment 50). Amendment 49 could not settle unconditional buckets' speed there, which is why the default is gated.
- **The evidence's scope.** The calls this changes, those with at least 16,384 routed rows, were measured on one model
  (Qwen3-30B-A3B, top-8, packed 4,096-token rows) on one RTX 5090. Other families' calls that reach the gate (gpt-oss at 4,096 tokens
  x top-4 sits exactly at it) change on this package's correctness tests: every bucketed geometry here matches the single block and the
  per-expert loop within rounding, not bit for bit. No speed or memory was measured for them.
- **The way back.** `NF4_QLORA_PAD_BUCKETS=0` restores the previous behaviour exactly: the single block on every call, op for op.
- **Tests.** The bucket test file pins the new default; CI's kernel list reads 548 passed on CPU and 996 passed on an RTX A2000 (CUDA).
