# RESULTS — sm_120 census: the grouped NF4 GEMM against every engine we could find

RTX 5090 (sm_120), gnf4 main `4f1db333`, torch 2.8.0+cu128, bnb 0.50.1.
Cell: the Qwen3-30B-A3B serving census routing — E=128, top-8, B=16
(R=128 rows, mean group 1.6) at the real expert shapes
(gate_up N=1536 K=2048, down N=2048 K=768). Arms alternate per draw;
A/A twins ride along; every challenger is correctness-gated — bitwise
within a kernel family, `max|Δ| ≤ max|ref|·2⁻⁷` across families —
**before it may be timed; a failing arm is excluded, and the probe
enforces this in code** (the first cut of the singleton arm published a
bitwise-only check; caught in review, gate enforced, receipts
regenerated on a second box). Receipts in [receipts/](receipts/); probes committed
beside them.

## The headline

| challenger | gate_up | down | verdict |
|---|---|---|---|
| **`gemm_4bit_grouped` (this library)** | **0.42 ms** | **0.21 ms** | — (±1% across boxes) |
| `torch._grouped_mm` on **unquantised bf16** | 0.88–1.28 ms | 0.52–1.30 ms | **loses 2.1–6.0× (two boxes; worst case ≥ 2.1×)** |
| v0 mainloop (SMEM-dequant + `tl.dot`) | 0.65 ms | 0.35 ms | loses 1.55× |
| per-row GEMV path (singleton groups) | 0.46 ms | 0.25 ms | loses 0.82–0.98× (rel err 5.0e-3/7.6e-3, gate PASS) |
| bnb `dequantize_4bit` + `mm` per expert | — | — | loses 5.4–9.3× at M=1 (crossover sweep) |

`torch._grouped_mm` is the engine transformers v5 ships as
`grouped_mm_experts_forward` — PyTorch's own grouped-MoE path, running
on weights at **2× the bytes**. Two boxes measured
(`receipts/gemm_p0b.json`, `receipts/gemm_p0b_box2_gated.json`,
rel err vs the NF4 truth ≤ 5.0e-3): **this kernel reproduces to ~1%
across boxes while the torch engine varies up to 2.5×** — quote the
worst-case ≥ 2.1×, not the best box.

## Configuration space is CLOSED on sm_120

144 M-tile cells (variant × groups × BLOCK_N × warps × stages): the
A5000-tuned v6 rule (`bn128/w4/s3`, register-LUT) is optimal **within
0.4%** on both shapes; `GROUPS=2` loses ~9% where it fits (triton's
per-CTA shared limit on GB202 is ~99 KB — the big-SMEM cells cannot
launch); every clean cell was numerically gated, 0 failures
(`receipts/blackwell_tune.json`). The decode-GEMV path's own grid
(dotpad × BLOCK_N × warps × split-K, 192 cells, run on
bandwidth-gated ≥1.4 TB/s silicon): no lever ≥ 1.034×, split-K always
loses (`receipts/gemv_tune.json`).

## Where the remaining roofline gap lives — and why we are not chasing it

The M-tile runs ~350 GB/s against a ~1.7 TB/s warmed DRAM read roofline.
The decode GEMV under CUDA-graph replay runs **34.2 / 20.4 µs per call =
4.1× / 4.8× over its byte floor**; the other ~31–44 µs/call an eager
caller sees is **host issue** the graph decode lane never pays
(`receipts/gemv_latsplit.json`). Mean group size 1.6 starves tensor
cores by construction (`mma.m16n8k16` fixes the M extent — see
RESULTS-k11), gate→act→down is sequentially dependent so the per-layer
call count is already minimal, and four independent engines contested
this kernel and lost. The 4–5× to roofline has **no existence proof**:
we will reopen it when a demonstrably faster engine at this op shape
exists, not on arithmetic.

## Method notes that changed numbers

- **Warm before timing a fresh allocation**: an unwarmed 2 GB read probe
  under `expandable_segments` reported 554 GB/s on 1,677 GB/s silicon
  (first-touch page mapping in the average). One earlier "slow box"
  reading was this artifact; retracted same-day.
- `torch._grouped_mm` wants one offset per batch of `b` (empty groups as
  repeated offsets) — two earlier referee attempts were disqualified
  (a launch-bound python loop; a per-touched-group offs layout).
- Rented "RTX 5090" hosts vary; bandwidth-sensitive sweeps here gate on
  a measured ≥1.4 TB/s warmed read before timing anything.
