### `bake_nf4(fused_marker=...)`: GraniteMoe's fused expert stack bakes

- The fused layout (one 3-D `[E, 2I, H]` gate-first stack per layer, plus its `[E, H, I]` down partner) was found
  only under names containing `.experts.`, the Gemma-4 spelling.
- GraniteMoe has no `experts` segment: `model.layers.N.block_sparse_moe.input_linear.weight` and
  `...output_linear.weight`. It now bakes with `fused_marker=".block_sparse_moe."` and
  `fused_proj=("input_linear.weight", "output_linear.weight")`.
- The default stays `.experts.`, so nothing else changes.
- When a checkpoint has no `expert` keys but does have `...input_linear.weight`, the no-experts diagnostic now names
  those two arguments.
- Checked end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` (RTX A2000 host):
  - 1,280 rows (1.7 GB) baked in 47 s;
  - experts4bit-qlora's paged server built on the arena and served 4 × 1,024-token prompts;
  - the serve estimate held (allocator peak 1.3% under it).
