### The compact bucketed delta is the default (experts4bit-qlora TC1 amendment 66); `NF4_QLORA_COMPACT_BUCKETS=0` restores the autograd body

- `_CompactBucketedDelta` (#505) now serves every bucketed LoRA-delta call unless `NF4_QLORA_COMPACT_BUCKETS=0`. It gives the same bytes as
  the autograd body, forward and every gradient (`kernel/test_compact_buckets.py`).
- The read: experts4bit-qlora TC1 amendment 66, Qwen3-30B-A3B on packed 4,096-token rows, one RTX 5090 (GPU-bound, device busy 0.99),
  torch 2.12. The matched arm's training-phase peak fell 0.654 GB (26.56 -> 25.91). Both arms stepped faster, 0.972 (matched) and 0.977
  (shipped), with less device time per step, 0.973 / 0.978. Held-out was identical at step 0 and within 0.00013 at the end.
- Scope: one model, one card class, torch 2.12, packed rows. Buckets engage only at 16,384 routed rows or more (`auto`), so shorter calls,
  TC1's field recipe among them, never reach this node. The single block and the ladder's plans keep their own bodies.
