### `rope_norm_qk`: the query's and the key's per-head norm and rotary in one launch (experts4bit-qlora#1313)

- **What.** `int4_b32.rope_norm_qk(q, k, q_weight, k_weight, cos, sin, q_eps, k_eps)` is `rope_norm_heads` on q and on k
  in one launch. The grid is (rows, HQ + HK), and each head takes its own projection's input, weight and eps.
- **Why.** experts4bit-qlora's attention fold calls `rope_norm_heads` twice per layer. This is lane P127's Phase 2
  (item d).
- **Bitwise.** Both forms run one `@triton.jit` helper, `_rope_norm_head`, which holds the arithmetic
  `_rope_norm_heads` ran inline, statement for statement. A test checks the outputs are bitwise the two calls', at 1
  and 16 rows and three head layouts, with distinct weights and eps; it holds under the interpreter too.
  `test_pdl`'s kernel list and launch count include the new kernel.
