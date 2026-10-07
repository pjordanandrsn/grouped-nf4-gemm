### `GNF4_GEMV_BW=1`: a bandwidth-targeted NF4 single-row decode GEMV (opt-in; every default unchanged)

- **Why.** At experts4bit-qlora's B=1 decode on Qwen3-30B-A3B the served NF4 expert GEMV (`_gemv_nf4_dotpad`) is
  2.47 ms of a 6.46 ms graphed step at 3.8x its byte floor (e4b SV2's census); the scalar route runs at ~18 % of the
  streaming ceiling. K3 / K4 / K26 put the cost in load-instruction issue and the per-element codebook gather, not in
  bytes; the int4-b32 GEMV reads the same bytes per parameter at ~66 % of the ceiling with an arithmetic decode.
- **What.** `nf4_grouped._gemv_nf4_bw`, reached from `gemm_4bit_grouped`'s single-row branch when `GNF4_GEMV_BW=1`
  (ahead of dot-pad):
  - one `[BLOCK_N, KC/64, 8]` int32 tile per K-step (8 contiguous words per row and absmax block, the int4-b32 tile
    shape), `absmax` applied to each 64-block's sum, fp32 accumulation, optional split-K (fp32 partials, host-reduced in
    split order);
  - the codebook decoded exactly: `prmt32`, PTX byte-permute (`prmt`) lookups of the fp32 codebook's byte planes and
    `lop3` selects, no memory gather, on a compiled NVIDIA target; `tree`, an exact 4-level select tree, under the
    interpreter (`GNF4_GEMV_BW_DECODE` forces one; `prmt32` where PTX cannot run is refused);
  - the PDL preamble, so `GNF4_PDL` / `GNF4_PDL_MAX_ROWS` reach the NF4 route as they reach the int4-b32 kernels;
  - plan `(BLOCK_N, KC, warps, split_k)` = `(16, 256, 4, 1)` by default, `GNF4_GEMV_BW_PLAN` per shape, `bw_config=` on
    `gemm_4bit_grouped` for harness sweeps; the dispatch tally gains `bw_tree`, `bw_prmt32`, `bw_splitk`;
  - `GNF4_GEMV_BW=auto` engages only at `_BW_SHAPES`, which is empty until lane K33 reads.
- **Values.** The two decodes are bitwise equal; one-hot activations read `dequant_ref` exactly; the result never
  depends on the call's row count. Against the scalar and dot-pad routes the reduction tree differs, so the tolerance
  contract applies (no worse than the scalar route's error by more than 5 %).
- **Tests.** `kernel/test_nf4_gemv_bw_interp.py` (interpreter; 28 cases, including the `prmt32` assembly executed by a
  byte-exact model of `prmt` / `lop3` over every byte value in every position, and a mutant selector it must catch);
  `kernel/test_nf4_gemv_bw.py` (compiled: `prmt32` bitwise the tree at every candidate plan and the families' expert
  shapes, one-hot readback, the tolerance contract, the PTX, PDL on sm_90+).
- **Not measured.** No speed is claimed until lane K33's microbench reads on its registered card.
