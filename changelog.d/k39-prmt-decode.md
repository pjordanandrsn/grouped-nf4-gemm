### K39: `lut="prmt"`, a byte-permute codebook decode for K25 (opt-in, bit-identical, no speed claim)

`gemm_nf4_grouped_smallm` gains `lut="prmt"`: NF4 nibbles decode through byte-permute (`prmt`) table lookups in
inline PTX instead of the select tree. It is off by default.

- **Bit-identical to `lut="tree"`:** it decodes the packed BYTES (K25's own load), so the dot's operand layout and
  accumulation order are K25's.
  - On an RTX A2000 under triton 3.4.0 it matched the tree bit for bit in 394 of 394 cases: real W16 routing for
    gate_up (lean route) and down, an 8-tile expert, masked columns, a masked K tail, the bf16 path, no gather,
    determinism, and CUDA-graph replay.
  - The merged kernel compiles to the same machine code as the build that was read, at every configuration the
    suite ran.
  - A K25 built on `nf4_grouped._nf4_prmt32` (int32 words) was NOT bit-identical (~1 bf16 ulp on most calls), which
    is why the byte path is used.
- **Fewer instructions (static, no timing):** `bench/sass_count` at experts4bit-qlora's served plan counts 17.0 lane
  instructions per weight on sm_120, against the tree's 24.1. It also uses 64 registers against 80 (theoretical
  occupancy 67 % against 50 %).
- **The same table words** as K33's GEMV decode: `nf4_grouped._bw_table_words()`.
- **NVIDIA only:** refused on ROCm; under the interpreter it decodes through the tree.
- **Tests:**
  - `test_nf4_smallm_prmt.py` executes the shipped PTX on the CPU, over every 16-bit chunk.
  - `test_nf4_smallm_prmt_gpu.py` checks bit-identity on CUDA.
  - `test_nf4_grouped_smallm_interp.py` adds `prmt` to its decode modes.
