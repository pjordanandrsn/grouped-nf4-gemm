### `NF4_QLORA_PAD_BUCKETS_LADDER=1`: bucket widths and batch counts on a fixed ladder (opt-in)

- The bucketed LoRA delta's per-bucket batched products take a new shape on almost every call, because the router moves every bucket's
  width and group count. A cuBLAS fp32 batched product costs host time per new shape: experts4bit-qlora TC1 amendment 24 measured
  119 µs on a new shape against 38 µs on a repeated one, on an RTX 5090. Amendment 53 profiled the fp32-adapter arm under torch 2.8 at
  about 268 µs of `aten::bmm` host time per call, with the device idle on it.
- With the flag set, each bucket's width and batch count are rounded up to quarter-octave rungs, at most 25 % over each. The padded groups
  take zero adapters and no rows, so values equal the unladdered buckets' to rounding. Over 40 Zipf(1) routings of 4,096 tokens, the
  distinct bucket shapes fall from 227 to 41.
- Off by default until a training A/B reads it. Laddered and unladdered plans never share a plan-memo entry.
