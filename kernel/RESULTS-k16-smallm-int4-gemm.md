# Results — lane K16: small-M int4-b32 GEMM for the attention projections (RTX 5090, 2026-09-19)

Pre-registration: [`PREREG-k16-smallm-int4-gemm.md`](PREREG-k16-smallm-int4-gemm.md) (2026-09-18, before any kernel code). Rows: [`receipts-k16/5090/k16_rows.json`](receipts-k16/5090/k16_rows.json); log, versions and forensics beside it. Run `k16-5090-2` (vast instance 51522950; run 1 `k16-5090` was a HARNESS_ERROR — the image had no pytest — and produced no number). Instrument: K14's bench (`k14_bench.py`'s timing loop, the same bf16 dequant-then-GEMM and K14 grouped-GEMM arms re-run beside K16 on the same card in the same process), M=16, Qwen3-30B-A3B's four attention shapes, 96 K16 configs swept per shape (BLOCK_N ∈ {32, 64} × KC ∈ {128, 256} × SK ∈ {1, 2, 4, 8} × warps ∈ {4, 8} × stages ∈ {2, 3}); every K16 config within the correctness contract first (`test_int4_smallm_interp.py` compiled: 9 passed on this card).

Card: NVIDIA GeForce RTX 5090, torch 2.8.0+cu128 / triton 3.4.0. Launch floor (empty kernel) **4.61 µs**.

## The rows (µs per launch, M=16, best K16 config named)

| shape | N | K | bf16 dequant path | K14 grouped GEMM | **K16 best** | config | bf16 / K16 | K16 rel. err. |
|---|---|---|---|---|---|---|---|---|
| `q_proj` | 4096 | 2048 | 10.37 | 12.56 | **6.35** | `k16_bn32_kc256_sk1_w4_s3` | 1.634× | 0.0029 |
| `k_proj` | 512 | 2048 | 6.23 | 12.51 | **4.37** | `k16_bn32_kc128_sk4_w4_s3` | 1.425× | 0.0031 |
| `v_proj` | 512 | 2048 | 6.29 | 12.50 | **4.49** | `k16_bn32_kc256_sk4_w4_s3` | 1.402× | 0.0031 |
| `o_proj` | 2048 | 4096 | 18.61 | 20.73 | **6.39** | `k16_bn32_kc256_sk8_w4_s2` | 2.912× | 0.0033 |

Relative error is against `x_bf16 @ dequant_int4_ref(packed, scales).T` in bf16 (K14's metric; the contract is ≤ 2⁻⁷ ≈ 0.0078 of max|ref|).

## Predictions, read against the rows

- **P1 (within 1.4× of Marlin g128: `q_proj` ≤ 8.9 µs, `o_proj` ≤ 11.6 µs) — HOLDS.** 6.35 and 6.39 µs. Against K15's Marlin rows on the same card class (6.37 / 8.28 µs, `gnf4.kernel.k15-marlin-comparator.5090.2026-09-11`): 0.997× on `q_proj` (parity) and 0.772× on `o_proj` (faster). Different processes and torch/CUDA builds, as K15's comparison was; the comparator figures are biased in Marlin's favour per the standing correction.
- **P2 (beats this package's bf16 dequant path on both) — HOLDS.** 1.63× on `q_proj`, 2.91× on `o_proj` — the thing K14 found no int4 arm could do (K14's grouped GEMM is slower than bf16 on every shape here too: 12.56 / 20.73 µs).
- **P3 (`k_proj` / `v_proj` do not lose to bf16 by more than 1.2×) — HOLDS.** K16 is faster there as well (4.37 vs 6.23; 4.49 vs 6.29 µs), within 1.0 µs of the 4.61 µs launch floor: launch-bound, as K14 said.
- **P4 (per-32-block in-tile scaling costs < 5 % against a single-scale control) — NOT TESTED.** `k16_bench.py` carries no single-scale control arm, so this run says nothing about P4 either way. It is an open item, not a pass: a control arm (one scale per row, same tile shape) added to the bench and one more ≤ $0.10 draw would read it.
- **P5 (model level: the attention GEMM row falls ≥ 0.4 ms/step at B=16 once routed) — PENDING.** Needs the consumer route (experts4bit-qlora `E4B_ATTN_INT4_SMALLM=1`, opt-in) and a P42-style census on a 5090; not a kernel-level number and not read here.

## Decision rule, applied

P1 ∧ P2 → **open the consumer PR and register the kernel-level claim as `measured`.** Registered as `gnf4.kernel.k16-smallm-int4-gemm.5090.2026-09-19`. The consumer route is experts4bit-qlora#578 (opt-in behind `E4B_ATTN_INT4_SMALLM=1`; the default stays the cached-bf16 path until P5 reads). No default here changes: nothing in this package routes to `int4_smallm` on its own.

## What the sweep says about the shape of the kernel (observations, not registered)

- Every best config is **BLOCK_N = 32, 4 warps**; `q_proj` prefers a single split with KC=256 (the whole 2048-wide K in one program per column block), the small-N `k/v_proj` prefer SK=4, and `o_proj` (K=4096) SK=8 — the split count follows K/N, which is what `plan_smallm`'s legaliser exposes and the consumer's construction-time plan can pick per projection.
- The A2000 pilot (sm_86) read 2.21× / 2.41× over bf16 on q/o; the 5090 reads 1.63× / 2.91×. The o_proj gain grew with the card (bf16's o_proj path runs at 60 % of ceiling there, K14's observation); the q_proj gain shrank (bf16's q_proj already runs at the streaming ceiling on the 5090, so the int4 read saving is the whole lever). An sm_86 ratio does not transfer to sm_120 in either direction — P39's rule, again.

## Cost

Run 1 (`k16-5090`, HARNESS_ERROR, no pytest on the image): ~1 min of a $0.65/h box. Run 2 (`k16-5090-2`): ~4 min box time. Lane ceiling $0.65; receipts in the private audit tree (`receipts/experts4bit-qlora/2026-09-19/k16-5090{,-2}/`).
