### `gemm_4bit_grouped(..., gather_div=)`: the singleton decode reads the token rows where they are (experts4bit-qlora#1313)

- **What.** With `gather_div=k`, `a_cat` holds token rows and row `r` of the call reads token `r // k`. These are a
  top-k MoE's (token, slot) rows, token-major. The option serves the singleton decode only (every group one row) and is
  refused elsewhere.
- **Bandwidth GEMV.** `_gemv_nf4_bw` takes a `GATHER_DIV` constexpr and reads the token rows in place, in one pass and
  in split-K.
- **Other single-row routes.** They make the expansion in the wrapper.
- **Why.** experts4bit-qlora copies each decode token's row k times (an `index_select`) before the gate_up GEMV. Under
  the bandwidth GEMV that copy's launch goes. This is lane P127's Phase 2 (item b2).
- **Bitwise.** Every route's output is bitwise the expanded call's:
  - under the interpreter: the bandwidth GEMV (one pass and split-K) and the scalar route;
  - on the card: both decodes, one pass and split-K, at Qwen3's gate_up and down shapes.
