### `gemm_4bit_grouped` takes int64 expert ids as they are (experts4bit-qlora#1313)

- **What.** A CUDA `expert_ids` tensor in `nf4_grouped.EXPERT_ID_DTYPES` (int32 or int64) reaches the kernels
  uncast. Any other dtype, and a host list or CPU tensor, is converted as before. `dgrad_4bit_grouped` is
  unchanged.
- **Why.** torch's top-k indices are int64, so a caller handing them over paid a cast launch on every call.
  experts4bit-qlora's NF4 route makes two calls per MoE layer. This is lane P127's Phase 2 (item a1).
- **Why it is value-identical.** Every grouped NF4 kernel already loads its id and widens it to int64 before any
  stride product. That covers the bandwidth GEMV, dot-pad and its split-K, the scalar decode and the M-tile. A
  silicon test checks that int64 ids give bitwise int32's output at each route, and that the bandwidth route is
  handed the caller's tensor itself. `Prebound` keys on dtype, so int64 ids get their own specialization.
