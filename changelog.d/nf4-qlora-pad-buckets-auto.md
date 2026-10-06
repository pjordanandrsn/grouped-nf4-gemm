### `NF4_QLORA_PAD_BUCKETS=auto`: bucketed padding only where a call carries at least 16,384 routed rows

- **Why.** experts4bit-qlora's TC1 read bucketed padding on packed 4,096-token rows as 0.893 of the single block's step and 4.29 GB
  lighter on the fp32 arm (amendment 48). At the field recipe its two boxes could not settle the speed: the single block's own draws came
  8-21 % apart (amendment 49 and its re-ask). Their census of every delta call put the field recipe's calls at 9,040 routed rows at most
  (median 3,968). Packed rows carry exactly 32,768 a call.
- **What.** `NF4_QLORA_PAD_BUCKETS=auto` buckets a call when its routed rows (the sum of its group sizes) reach `_PAD_BUCKETS_AUTO_MIN_ROWS`
  (16,384; `NF4_QLORA_PAD_BUCKETS_MIN_ROWS` overrides it), and gives every smaller call the single block, op for op. `1` still buckets
  every call; unset or `0` is unchanged. Opt-in.
- **Tests.** `kernel/test_lora_delta_pad_buckets.py`: the gate's boundary, the override and unknown values; end to end, a call under
  the gate is the single block bit for bit, and a call at the gate is bucketed and equal to it to rounding.
