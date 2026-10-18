### `decode_split_plan` and `split_cap`: the paged decode's split planner as a pure function (experts4bit-qlora#1313)

- `fp8_paged_attn.decode_split_plan(B, n_kv_heads, table_cols, *, block_tokens, ktile, compute, pack_heads, sms,
  split_cap=None)` is the split count `fp8_paged_decode_attention` plans when no `n_split` is passed. It was an inline
  block in the kernel's wrapper; the wrapper now calls the function, so a consumer can read the plan without a GPU.
  `kernel/decode_split_plan_table.json` freezes the pre-refactor plan, executed from commit `8060ed74`. It covers B
  1..64 × SMs {128, 170, 188} × three table widths × pack × compute × KV heads {4, 8} × default and explicit ktile:
  9216 values, all unchanged.
- `split_cap` (default None: the plan as before) plans `min(split_cap, plan)`; a cap at or above the plan changes
  nothing. A smaller split count regroups the fp32 log-sum-exp combine, so results agree within tolerance, not bit for
  bit. `fp8_paged_decode_attention(..., split_cap=None)` passes it through; an explicit `n_split` is used as given.
- Why: experts4bit-qlora's P143 microbench timed 16 splits faster than the planned 32 at B=1 on an RTX 5090. Its
  served read passes `split_cap=16` for one family; every other caller keeps today's plan.
