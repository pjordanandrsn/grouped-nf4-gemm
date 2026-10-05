### The decoded training route, opt-in: `GNF4_TRAIN_GEMM=decoded` (dequant_groups + one grouped bf16 GEMM launch per chunk of groups; `auto` untouched)

- **What it is:** `nf4_route.decoded_forward` / `decoded_dgrad`, also reached through `FusedGroupedNf4` (forward and dgrad):
  - per chunk of present groups, one `dequant_groups` launch decodes the chunk's experts to bf16;
  - then one new Triton kernel, `_grouped_bf16_gemm_kernel`, runs every group of the chunk, with bf16 operands, fp32
    accumulation and one bf16 rounding.
- **The cap:** each chunk's decode transient is at most `GNF4_DECODED_MAX_BYTES` (default 256 MiB, and at least one expert).
  `decoded_chunks` is the plan as a pure function.
- **Origin:** experts4bit-qlora's RD1 probe arm `decoded_cap`, moved here with the arithmetic unchanged.
- **Opt-in only.** `route_for` answers `decoded` only when asked, and `auto` never takes it, on any card or group count.
- **Correctness gate in CI:** RD1's gate, the route's error against an fp32 reference at most 2× the dense route's.
  - It runs in the interpreter on CPU (`kernel/test_nf4_decoded_interp.py`).
  - It also runs compiled, against the real `dense` route and the fused kernels (`kernel/test_nf4_route.py`).
  - Two more checks: a capped call is bit-identical to the uncapped one, and the cap bounds the measured transient.
- **No speed claim.** Whether the route shortens a training step is experts4bit-qlora's TC1 full-step A/B, registered before
  its box.
- **Measured memory:** RD1's per-call reading on one RTX 5090 put the route's peak at 180–448 MiB above its inputs at the
  256 MiB cap. That is a bound measured on one card, not a guarantee.
