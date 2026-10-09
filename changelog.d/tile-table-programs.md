### `build_group_tiles_fused(..., rank="cumsum", programs=P)` splits the cumsum tile table over P programs (opt-in; e4b#846)

The chunked cumsum table (#519) still runs as one program. At 128 experts and 512 routed rows it walks every row twice in
`[128, 64]` chunks: e4b's P122 read it at 2.51 ms of a 64-row decode step, 52 µs a launch over 48 layers, still 1.75×
the chained builder's named kernels.

`_tile_table_cumsum_mp` gives each of `P` programs a slice of the experts. Each program:
- takes every expert's count from one `tl.histogram` of the ids, with the masked rows' sentinel in a bin of its own, so
  it knows every expert's row and tile offsets without talking to the others;
- ranks and writes only its own experts' rows, in `[EB/P, RCHUNK]` chunks with the per-expert carry.

A row's expert has exactly one owner, so no address is written twice. Program 0 zeroes the padding slots when `lean`.

`programs=None` or `1` (the default) is the one-program kernel, launched exactly as before. `programs > 1` needs
`rank="cumsum"`.

The integers are the same at every `P`:
- the five tables, and the lean variant's six, equal `programs=1`'s, the chained builder's, and the pairwise rank's
  where it applies;
- `order` equals `torch.argsort(expert_ids, stable=True)`;
- every row is written by exactly one program, counted by the kernel's test-only `OWNERS` flag. This holds with more
  programs than experts and with `P` not dividing the padded expert axis.

`kernel/test_tile_table_programs_interp.py` covers 40, 128 and 256 experts, 1 to 1,024 rows and `P` = 2, 3, 8 and 64,
under uniform, skewed and single-expert routing, with both the histogram and the counting pass (202 cases). It passed
under the interpreter and compiled on an RTX A2000 with triton 3.4 (correctness only). The speed is experts4bit-qlora's
to read. This targets the grouped-nf4-gemm release after 0.44.0.
