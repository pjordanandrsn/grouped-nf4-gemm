# Backward through a frozen GGUF k-quant weight: design note (zero rental)

**Status:** a design for review only. There is no package code and no rental until it is reviewed. Implementation
gets an owner after the current stabilization items close.

**What this covers:** training LoRA adapters on top of a frozen GGUF k-quant base needs one quantity from the
quantized weight: the input gradient dX = dY · W, where W is the dequantized [N, K] weight. No weight gradient is
needed. This note fixes the reference path, the oracle, the bars, the fixtures and the arms that a backward through
`kquant_ref` weights would be checked against.

It is a capability gap, not a correctness bug: **today no backward path exists through a k-quant weight in this
package or in experts4bit-qlora, so there is nothing yet to check.**

## 1. What exists today

Read at `bf6184f`.

- **`kernel/kquant_ref.py` decodes; it has no backward.**
  - Pure-torch decoders for Q8_0 and Q2_K–Q6_K, plus F32/F16/BF16 passthrough (the `GGML_DEQUANT` table, 176-185);
    entry point `dequantize_ggml` (188).
  - The i-quants are a stated non-goal (24-27): any type outside the table raises (195-200).
  - A row must be whole blocks (`ne0 % elems`, 222).
  - There is no autograd Function and no dgrad.
- **The decode is adjudicated bit-exact against gguf-py** (`kernel/test_kquant_ref.py`).
  - The synthetic arm (58-67) compares 37 random blocks per type by int32 view; disagreement is a stop, not a
    tolerance.
  - Random blocks (47-55) are random bytes with each type's fp16 fields overwritten by finite values.
  - CI installs `gguf` so this arm runs (`.github/workflows/ci.yml`, "GGUF k-quant oracle").
  - The real-bytes arm (82) is gated on `GNF4_GGUF_FIXTURES` and is off in CI.
- **experts4bit-qlora** uses the decoder only to load weights densely, as frozen bf16 parameters, for serving. It has no
  training path over GGUF bases.
- **The NF4 bars this design borrows from:**
  - The fp64 bar ("B-rel ≤ 2× the dequant path, ≤ 1e-2") is a **forward** check: `err_vs_fp64` and `dequant_path_err`
    (`kernel/test_nf4_grouped.py:50,62`), asserted at 147-148.
  - The NF4 **dgrad** checks in `kernel/test_nf4_qlora_grad.py` compare against a bf16 decode-then-matmul reference
    (`_reference_forward`, 42):
    - the exact loop at rel == 0 (146);
    - the dgrad kernel within the bf16 budget, rel < 1e-2 (341, 370).

## 2. Prior art (cited, for orientation)

| project | backward through a quantized GGUF weight | what dX is compared with |
|---|---|---|
| [woct0rdho/torch-ggml-ops](https://github.com/woct0rdho/torch-ggml-ops) | yes, on ROCm: tiles decoded to bf16, then a bf16 matmul | a bf16 dequantize-then-matmul reference, by normalized RMSE |
| [woct0rdho/transformers5-qwen3.5-recipe](https://github.com/woct0rdho/transformers5-qwen3.5-recipe) | yes, through torch-ggml-ops, for MoE LoRA | a bf16 dequantize-then-matmul reference, by `atol` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | the CPU `OUT_PROD` dequantizes quantized rows | `tests/test-backend-ops.cpp` builds gradient cases only for non-quantized `src0` |
| [Hugging Face transformers](https://github.com/huggingface/transformers) | GGUF weights are dequantized on load; the GGUF quantizer reports `is_trainable` false | — |
| [Unsloth](https://github.com/unslothai/unsloth) | its training worker refuses GGUF models as inference-only | — |

None of these compares dX with an fp64 oracle. That is the one thing this design adds.

## 3. The reference path

`KQuantLinearRef`, a `torch.autograd.Function` with W frozen:

- **forward(X, W_bytes, type, shape):** `W = dequantize_ggml(type, W_bytes, shape).to(bf16)`; return `X @ W.T` as a bf16
  matmul (fp32 accumulate, bf16 out).
- **backward(dY):** return `dX = dY @ W`, with W decoded the same way, as a bf16 matmul. Return `None` for the bytes and
  the type: **there is no dW.**

**This reference *is* the bf16 control.** Checking it against the oracle proves the harness, not a kernel: it should
sit at the control's own error by construction. The object checked later is a fused dgrad kernel, which decodes in the
tile and never materializes W; it is scored against this reference and against the oracle.

## 4. The oracle and the bars

- **Oracle, decoded independently:** `W64` comes from **gguf-py's own** `dequantize` (independent code that handles
  the K types), cast to float64, not from `dequantize_ggml`. If the oracle and the path under test shared a decoder, a
  decode bug would move both sides together and no dX bar could see it. The link between the two decoders is the
  existing bit-exact adjudication of `kquant_ref` against gguf-py (`kernel/test_kquant_ref.py`).
  X and dY are drawn in bf16 and upcast to float64 (exactly representable); `dX_ref = dY64 @ W64`.
- **Error:** e = ‖dX − dX_ref‖_F / ‖dX_ref‖_F (relative error, the same normalization as `err_vs_fp64`).
- **Control:** e_ctrl = the error of the reference path in §3.
- **Bars:**
  - **B-rel:** e_test ≤ 2 · e_ctrl.
  - **B-abs:** e_test ≤ 1e-2.
  - **Exact fallback:** any dequantize-then-matmul fallback route must equal the reference bit-for-bit (e == 0
    against it), as the NF4 exact loop does.
  - **Live control:** e_ctrl must itself be > 0 and < 1e-2; otherwise the bars are vacuous.
- **Adjoint identity (fp64, CPU):** ⟨dY, X · W64ᵀ⟩ = ⟨dY · W64, X⟩ to a relative 1e-10. This cheaply catches transpose,
  layout and row-order bugs in the harness itself.
- **LoRA wiring:** the frozen bytes carry no gradient. Adapter A/B gradients are nonzero and within 1e-2 relative of
  the same adapters over the reference path.
- **Sensitivity:** negating one row of dY must change dX.
- **Armed:** each mutation below must **fail a bar**, not crash.
  - **Only the path under test is mutated, never the oracle.** The oracle decodes independently (above), so a decode
    mutation in the path under test shows up as a bar failure.
  - **They run as strict expected failures with `raises=AssertionError`**, the exception the bars raise. A plain strict
    xfail would also count any exception (a shape error, an index error) as the expected failure, so a mutation that
    merely crashed would pass vacuously.
  - The mutations:
    - m1: the Q4_K/Q5_K 6-bit scale/min packing with the high-bit spill broken;
    - m2: the Q3_K `hmask` inversion dropped;
    - m3: Wᵀ used in place of W in the backward, **at N == K**, where the shapes still agree and the result is silently
      wrong. At N ≠ K the same mutation raises a shape error, which belongs at most in a separate shape-refusal test;
    - m4: the super-block index shifted by one row at N = 37;
    - m5: `dmin` zeroed.
- **Calibration only:** scales rounded to bf16 before decoding. Record whether this trips the bars; it is not required to.

## 5. Fixtures

gguf-py dequantizes the K types but does not quantize them, so fixtures come from three sources:

1. **Random packed blocks.**
   - Random bytes with each type's fp16 `d` / `dmin` fields overwritten by finite, bounded values. This is the
     `_random_blocks` pattern (`kernel/test_kquant_ref.py:47-55`); raw random bits would produce NaN and Inf.
   - Scales are narrowed so W has a realistic magnitude (about 1/√K).
2. **The existing real-byte fixtures:** sha256-pinned tensors from released GGUF files, gated on `GNF4_GGUF_FIXTURES`.
3. **Q8_0 also from gguf-py:** quantize a random W with gguf-py.

**Coverage**
- **Types:** Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
- **Shapes:**
  - K ∈ {256, 512, 2816, 4096} (2816 is eleven super-blocks; a partial block in K is invalid by format);
  - N ∈ {1, 7, 37, 129, 2048} (ragged tails);
  - M ∈ {1, 3, 17, 128, 1000};
  - one grouped case with group sizes [1, 7, 0, 5], including an empty group.
- **Seeds:** three per (type, shape), derived from the type id and the shape index.
- **Out of scope:** IQ4_XS and the other i-quants. `kquant_ref` refuses them by design, so they need a decoder of their
  own before any backward can be checked.

## 6. Arms

1. **CPU, first.** The oracle (gguf-py decode), the adjoint identity, the reference path, the control and the
   mutations all run in fp64 or bf16 on CPU (expected to take seconds per case up to 2048 × 4096; not yet
   measured). This arm proves the harness and can join CI.
2. **The project's RTX A2000 (sm_86).** The candidate fused dgrad kernel is scored against the same oracle and bars once
   it exists. Correctness only, no rental, no performance statement.

## 7. Recorded follow-up (no change here)

The NF4 dgrad checks (`kernel/test_nf4_qlora_grad.py`) bound the backward against a **bf16** decode-then-matmul
reference. The **fp64** oracle with B-rel / B-abs bars is applied to the NF4 forward (`kernel/test_nf4_grouped.py`).

Extending the fp64 oracle and B-rel bar to the NF4 dgrad would put both lanes on the same footing. That is recorded
here as a follow-up for review and is not changed by this note.
