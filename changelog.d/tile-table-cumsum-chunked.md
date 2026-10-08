### `build_group_tiles_fused(..., rank="cumsum")` walks the rows in chunks above 8,192 table elements (`rchunk`)

The cumsum rank (#515) built its whole [E, R] hit matrix, the running count and the rank select in one program. At 128
experts and 512 routed rows that is 65,536 lanes per tensor. On an RTX 5090 the launch took 218 µs, against 40 µs at 40
experts, and experts4bit-qlora's 64-row decode step got 1.44× slower with it (lane P120; the switch there stays opt-in).

`_tile_table_r1` now takes `RCHUNK`. The rows go in chunks: one pass sums each expert's hits, a second ranks each chunk's
rows by its running count plus a per-expert carry of the rows before it. No tile exceeds [E, RCHUNK].

The wrapper's `rchunk=None` (the default) chunks once `next_pow2(E) × next_pow2(R)` exceeds `CUMSUM_TILE_ELEMS` (8,192),
which is 64-row chunks at 128 experts. `rchunk=0` keeps one piece, and a power of two ≥ 16 forces that chunk.

The integers are the same every way:
- the permutation equals `torch.argsort(expert_ids, stable=True)`;
- the five tables equal the chained builder's and the one-piece cumsum's;
- the lean variant's six outputs equal the default's.

`kernel/test_tile_table_cumsum_chunked_interp.py` covers 1 to 1,024 rows at 40, 128 and 256 experts, with chunks of 16,
64 and automatic, under uniform, skewed and single-expert routing. Its speed is experts4bit-qlora's to read.
