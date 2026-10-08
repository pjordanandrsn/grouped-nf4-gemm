### The compact bucketed delta is the default (`NF4_QLORA_COMPACT_BUCKETS=0` turns it off)

- Licensed by experts4bit-qlora TC1 amendment 66. On Qwen3-30B-A3B, packed 4,096-token rows, one RTX 5090 and torch 2.12, the training
  peak fell 0.654 GB and steps ran 0.972 (matched) and 0.977 (shipped) of the old path, with held-out unchanged.
- Only calls of 16,384 routed rows or more are bucketed, so shorter calls (TC1's field recipe among them) never reach it. The single
  block and the ladder keep their own bodies.
