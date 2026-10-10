### Backward through a frozen GGUF k-quant weight: reference path, calibrated bars and released-byte fixtures

- **`kquant_ref.KQuantLinearRef` / `kquant_linear`:** y = x · Wᵀ with W decoded from frozen GGUF bytes. The backward
  returns dX = dY · W only (no weight gradient), on the same bf16 decode.
- **`kernel/test_kquant_backward.py`** scores dX against an fp64 oracle decoded independently by gguf-py:
  - the bar e ≤ min(2 · e_ctrl, 1e-2), a live control, the adjoint identity, and LoRA wiring;
  - a decode probe: dY = I must return the reference decode bit for bit;
  - five deliberate decoder faults and a transpose fault, each of which must fail a bar.
  It runs in CI on synthetic blocks; the released-byte arm is gated on `GNF4_GGUF_FIXTURES`.
- **`bench/gguf-backward/CALIBRATION.md`** measured the floor and each fault before the path was scored.
  - The floor is 2.2-2.5e-3; every fault is ≥ 130x the bound on released attention, expert and FFN bytes for Q2_K-Q6_K
    and Q8_0.
  - Super-block scales must stay fp16/fp32. Rounded to bf16 they cost a median 2.1-2.9x the floor on Q2_K/Q4_K/Q5_K,
    and the decode probe pins that.
- **`scripts/fetch_gguf_fixtures.py --ffn`** fetches the expert / FFN fixtures, sha256-pinned at pinned revisions.
  The renamed meta-models source is recorded beside its original URL; its bytes are unchanged.
