# The prefill M-tile height rule: `max` vs `cost` (RTX A2000 evidence)

`gemm_4bit_grouped`'s M-tile path picks one tile height for every group in a launch. The shipped rule keys it on the
largest group (`_prefill_block_m(max(sizes))`). The opt-in `GNF4_PREFILL_TILE_RULE=cost` picks the height that minimises
`tiles x (D + BLOCK_M)` over the actual sizes (`_prefill_block_m_cost`, `D` = 96 rows by default).

- **`tiles_sweep.py` → `tiles_sweep_a2000.jsonl`.** 42 synthetic routed batches: 16–4096 tokens, top-8 over 128 experts, and
  Dirichlet router skews 2.0, 0.5 and 0.15, at Qwen3-30B-A3B's two expert shapes. Each batch was timed at every height.
  - The model `time = tiles x (a + b x BLOCK_M)` fits to R² 0.998, with `a / b` = 118 rows (gate_up) and 109 rows (down).
  - Slowdown against the best height per batch, worst and geometric mean: `max` 1.656 and 1.179; `cost` at D 96 1.175 and
    1.024 (flat for D in 64–128).
- **`route_probe.py` → `route_probe_a2000.json`** (needs experts4bit-qlora and the checkpoint; `mk_slice.py` builds the 4-layer
  slice). This run uses Qwen3-30B-A3B's real routers: a 4-layer slice of the real checkpoint, trained through experts4bit-qlora's
  fused step on the TC1 alpaca token rows (micro-batch 2 × accum 4, r16 alpha 16).
  - Over the 384 forward calls, the median group is 35 rows, the largest group in a call has a median of 155 rows and a 90th
    percentile of 377. So `max` always takes 128-row tiles.
  - Replaying 64 recorded calls at every height: `max` 711.5 ms, `cost` 545.5 ms, best 538.5 ms.
  - Training step, ABBA order: `max` 1.619 / 1.589 s, `cost` 1.484 / 1.493 s.
  - The outputs are bit-identical across rules (`kernel/test_prefill_tile_rule.py`).

**On an RTX 5090** (experts4bit-qlora TC1 amendment 14, `tc1-5090-43`, Ryzen 9 7950X): `cost` / `max` = **0.924** [0.918, 0.930] on the shipped arm
and **0.968** [0.964, 0.971] on the matched arm, held-out unchanged; `max` launched 128-row tiles on ~92 % of calls, `cost` mostly 64 (64 %) and 32
(24 %). By the amendment's decision rule `cost` is the default; `GNF4_PREFILL_TILE_RULE=max` restores the previous rule. The fit above is sm_86's.
