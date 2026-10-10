### Nearby-token expert reuse, now with Qwen3-30B-A3B (experts4bit-qlora#1469)

`bench/cold-engine/routing-trace` adds Qwen3-30B-A3B's four decode traces from the experts4bit-qlora#1469 item 1
census (read in experts4bit-qlora#1555), byte for byte, as `qwen3_{prose,code,math,dialogue}.jsonl`. It adds the model's shape to `locality_1469.py`
and its Belady/cache/LRU rows to `oracle-headroom.json`. The rows for the other models are unchanged.
- **Results:** at W = 16 a window cache needs 2,479 rows (40% of the arena) for 127.2 rows/token; Belady at that
  capacity is 15.3.
- **Conclusions:** the results document's conclusions and item 3's decision stand.
