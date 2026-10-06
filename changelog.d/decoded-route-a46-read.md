### The decoded training route's docs cite its training-step read: no measurable saving, so `auto` does not take it (docs only)

- **Where.** `nf4_route`'s module comment, the `grouped-nf4-gemm` capability's limitation, and the README route paragraph now cite
  experts4bit-qlora's TC1 amendment 46 (one RTX 5090, experts4bit-qlora#1221).
- **The read.** Decoded/fused 1.005 [0.979, 1.031] on OLMoE-1B-7B and 1.066 on Qwen3-30B-A3B.
- **Unchanged.** `GNF4_TRAIN_GEMM=decoded` stays opt-in, and "no speed is claimed" stays alongside.
