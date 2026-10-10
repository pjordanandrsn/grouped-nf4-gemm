### Design note: backward through a frozen GGUF k-quant weight (zero rental)

`bench/gguf-backward/DESIGN.md` sets out how a backward pass through frozen `kquant_ref` weights (Q8_0, Q2_K–Q6_K)
would be checked once one exists. It covers:
- a reference autograd Function that decodes, then runs a bf16 matmul; this reference is also the bf16 control;
- an fp64 oracle decoded independently with gguf-py, with the bars B-rel ≤ 2× the control and B-abs ≤ 1e-2;
- the adjoint identity, LoRA-wiring checks, and five mutations applied only to the path under test, which must fail a
  bar (`raises=AssertionError`), not crash;
- fixtures from bounded random packed blocks and the existing real-byte files;
- arms: CPU first, then an RTX A2000, correctness only.

No package code changes. The i-quants are out of scope.
