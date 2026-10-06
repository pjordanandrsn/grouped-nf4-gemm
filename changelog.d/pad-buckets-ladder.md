### `NF4_QLORA_PAD_BUCKETS_LADDER=1`: bucket widths and batch counts on a fixed ladder (opt-in)

- The bucketed LoRA delta's per-bucket batched products take a new shape on almost every call, because the router moves every bucket's
  width and group count. A cuBLAS fp32 batched product costs host time per new shape: experts4bit-qlora TC1 amendment 24 measured
  119 µs on a new shape against 38 µs on a repeated one, on an RTX 5090. Amendment 53 profiled the fp32-adapter arm under torch 2.8 at
  about 268 µs of `aten::bmm` CPU self time per call against 86 µs under torch 2.12. That is an upper bound on host work, since self time
  also counts waits on a full launch queue. 59.7 % of torch 2.8's added step was not device time.
- With the flag set, each bucket's width and batch count are rounded up to quarter-octave rungs, at most 25 % over each. The padded groups
  take zero adapters and no rows, so values equal the unladdered buckets' to rounding. Over 40 Zipf(1) routings of 4,096 tokens, the
  distinct bucket shapes fall from 227 to 41.
- Off by default. Laddered and unladdered plans never share a plan-memo entry.
- First training A/B: experts4bit-qlora TC1 amendment 54, on an RTX 5090 in torch 2.8, fp32 adapters, packed rows. The ladder stepped
  1.010 of the default buckets' time on a host where the step was not host-bound (device time 99 % of the step). It cut `aten::bmm`'s
  CPU self time per call about tenfold (~145-150 µs to ~14-17 µs), and the padding added about 2 % of device time and 0.34 GB of
  peak. It stays opt-in: whether it helps where the host is the bottleneck has not been read.
