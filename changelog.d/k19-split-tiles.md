### K19 over a split tile table: full 32- or 64-row tiles for a heavy expert, 16-row tiles for the rest (opt-in; e4b#846)

On Qwen3-30B-A3B the prefill rows of a MoE layer are bimodal. A shape census on experts4bit-qlora's SC2 prompts found,
per 512-token chunk (4,096 routed rows), one expert taking a median 398 rows and the other ~90 a median 24. On K19's one
16-row table the heavy expert runs as ~25 tiles, and each re-reads and re-dequantises the same weight slice. Uniform
taller tiles would pad the light experts instead: 30 % at 32 rows and 48 % at 64, against 16 % at 16.

- `nf4_grouped.build_group_tiles_split(expert_ids, E, big_m)` builds two tables from one `order`:
  - each expert's `c // big_m` full `big_m`-row tiles;
  - its remainder in 16-row tiles.

  It runs on device like `build_group_tiles_device` (no host sync, static shapes, so it is capture-legal). The padding
  is exactly the 16-row table's. On the census, weight passes per N-block drop from 307 to 204 at `big_m=64` and 212 at
  32.
- `gemm_int4_b32_grouped_smallm` takes `block_m` (16, the default, then 32 or 64) and `out=`, which stores into an
  existing buffer so two launches fill one result. `gemm_int4_b32_grouped_split` launches the full tiles, then the
  remainders. The gather, `scatter` and `gather_div` work as on the 16-row table.
- **Defaults are unchanged.** `block_m=16` with no `out` is K19's launch, bit for bit. Nothing in this package routes to
  the split on its own. Its speed is experts4bit-qlora's to read, on a rented card.
- **Numerics.** The outputs keep K19's contract: within one bf16 ulp of the dequant reference. Whether a row on a tall
  tile is the same bits as on a 16-row tile depends on the card's codegen, so it is tested per target, not assumed.
  Compiled on an RTX A2000 (sm_86), the split equals the 16-row table with `torch.equal` at K 768 and 2048, at
  `big_m` 32 and 64 (correctness only, no timing).
