### GGUF backward: model-level gradcheck

- **`KQuantLinearRef` / `kquant_linear` take `compute_dtype`:** bf16 by default (unchanged); `torch.float64` runs the
  same Function code exactly, for `torch.autograd.gradcheck`.
- **`kernel/test_kquant_gradcheck.py`** checks a tiny 2-layer model over frozen k-quant weights for each type (Q8_0,
  Q2_K–Q6_K):
  - fp64 gradcheck in slow mode (fast mode passed a transposed backward at a square shape);
  - a backward scaled by 1.001, a zeroed one and a transposed one must each fail it;
  - the bf16 path's gradients must equal those of the dense-bf16 model bit for bit.
