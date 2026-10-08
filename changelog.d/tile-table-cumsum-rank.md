### `build_group_tiles_fused(..., rank="cumsum")`: the one-launch tile table above 256 routed rows (opt-in)

The one-launch expert-major tile table (`_tile_table_r1`, lane K23's builder) ranked rows by an O(R^2) pairwise compare,
so it took at most 256 routed rows, and anything wider fell back to the chained builder (`nf4_grouped.build_group_tiles_device`:
argsort, scatter, cumsum, searchsorted and index_select, about 16 launches a layer). On experts4bit-qlora's 64-row decode
step (512 routed rows at top-k 8) that chained table costs 2.87 ms of a 15.64 ms step on an RTX 5090, against 0.93 ms for
the one-launch table at 256 rows (experts4bit-qlora lane P119).

`rank="cumsum"` ranks each row by a running count of its expert's rows along the hit matrix the kernel already builds
([E, R] lanes instead of [R, R]), so one program takes R up to 1024. The integers are the same: the permutation equals
`torch.argsort(expert_ids, stable=True)` bit for bit, all five tables (and the lean variant's six) equal the chained
builder's, and at R <= 256 the cumsum and pairwise kernels agree (`kernel/test_tile_table_cumsum_interp.py`, R from 1 to
1024, uniform, skewed and single-expert routing). The default stays `rank="pairwise"` with its 256 cap.
