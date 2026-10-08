### `NF4_QLORA_SINGLE_LADDER=1`: the single padded block's group count and width on the ladder's rungs (opt-in)

Every call below the bucket gate (`NF4_QLORA_PAD_BUCKETS=auto`) takes the single padded block. Its two `bmm`s run at
`[G, widest, K]`, and the router makes nearly every `(G, widest)` new, so cuBLAS pays its per-shape host cost on nearly every
call. At TC1's field recipe on one RTX 5090 (experts4bit-qlora TC1 amendment 69, Qwen3-30B-A3B, torch 2.12), the padded
delta's `aten::bmm` ran about 3,076 calls a step at about 172 us of CPU self time each, above its 120 us of device time. The
step's GPU was busy for 0.49 of it.

With `NF4_QLORA_SINGLE_LADDER=1` the block runs at `[_ladder_up(G), _ladder_up(widest), K]`. The `NF4_QLORA_PAD_BUCKETS_LADDER`
rungs are four per octave, so a rung is at most 25 % above its value and the block holds at most 1.5625× the single block's rows.
The padded groups take zero adapters (`F.pad`) and zero rows, and the output gather never reads them. Every real row gets the
same arithmetic at a repeating shape: values and gradients equal the single block's to rounding (`kernel/test_single_ladder.py`,
CPU and CUDA, unique and repeated ids, three dtype pairs). Over 40 Zipf(1) top-8 routings of 512 tokens, the 24 distinct
shapes become 1.

Scope:
- lean single block only: not `NF4_QLORA_LEAN_DELTA=0`'s body, not `NF4_QLORA_COMPACT_DELTA=1`'s node, not bucketed calls;
- its plans sit under their own memo key;
- `SINGLE_LADDER_STATS` counts calls and laddered rows.

Off by default: unset and `0` are the single block op for op. experts4bit-qlora registers its A/B at the field recipe.
