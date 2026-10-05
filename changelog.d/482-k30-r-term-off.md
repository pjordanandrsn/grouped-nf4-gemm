### The int4-b32 split-K R term is off on every part (`SPLITK_R_TERM_MAX_SMS = 0`, K30's registered consequence)

- K30 read the term OFF on a rented NVIDIA L4 (#481). Over the 24 cells at R >= 16 the R-aware pick's summed time was
  1.0101x the N-only pick's, worst cell 1.1781x; KEEP needed <= 0.97 and <= 1.02. Every card now plans split-K from
  (N, K) alone, as the 5090 already did. Outputs on <= 64-SM parts at R >= 16 move within the split-K reorder class.
  B=1 decode is unchanged, since it was below the floor.
- The mechanism stays, and its tests run under a monkeypatched gate. `test_the_r_term_is_off_by_default_on_every_sm_count`
  enforces the default; with the gate set back to 64 it fails 8 of 8.
