# Pre-registration — what does a K3 layer's expert block cost off a real NVMe arena?

**Written 2026-08-14, before the pod was rented.** Committed ahead of the data.

## What is being tested

The K3 lane's claim is *compute on packed bytes off an NVMe arena* — bytes go disk → arena
→ `gemm_mxfp4_grouped` with **no dequantize round trip**. Everything about K3 so far is
derived from the release's own headers (row = 17,547,264 B, 92 layers × 896 experts,
`row_stride == row_bytes` so zero padding) plus an 8-row slice round-trip that showed 48/48
segments byte-identical. **No throughput number exists.**

The full 1.45 TB arena is being baked on the QNAP to zpool3, which is **HDD** — zpool1 is
777.4 GB total against a 1446.5 GB arena, so it cannot hold it even empty. That arena is
good for correctness and useless for timing.

So: a **1-layer slice** on a rented box, baked onto real NVMe from shards the pod pulls
straight from HF (17.0 GB for layer 1 — no home upload), measuring the one number the lane
is missing.

## The case measured, and why decode

**Decode: T=1 token, top-16 of 896 experts.** That routes 16 unique experts = **280 MB per
layer**, which is the regime the tier is actually for. Prefill at T=512 would route nearly
all 896 experts and read the whole 15.7 GB layer — a different question, and not the one the
"batch tier" framing rests on.

## Arms

| arm | what runs | isolates |
|---|---|---|
| `arena` | `moe_layer_forward(src, layer, ...)` — fetch routed experts off NVMe, then the packed GEMM | the whole claimed path |
| `resident` | `fused_stacks` once up front, then the same GEMM on VRAM-resident stacks | the GEMM alone |
| `resident_self` | `resident` again, same round | **control** — self-pair |

`arena / resident` is the tier's overhead. Interleaved in fixed order, 5 scored rounds, first
dropped.

## Gate

Per the amended timing protocol: an effect is resolved when its per-round range does not
overlap the control's. `resident_self / resident` is the control.

## Predictions

1. **`arena` is dominated by the read.** 280 MB at a plausible 3–5 GB/s is 56–93 ms, against
   a grouped GEMM over 16 experts that should be single-digit ms. Predict
   **`arena/resident` > 5×**.
2. **Per-layer arena time 40–120 ms**, so a 92-layer token lands at **3.7–11 s** — the
   existing ~7.5 s/token estimate sits inside that band, and this is the first measurement
   that could contradict it.
3. **Achieved read rate within 2× of this pod's raw device rate** at the same queue depth,
   measured separately on the same file. If it is far below, the tier's read path is the
   limit rather than the device — the same question that turned out to be an artifact on
   the QNAP.
4. **`resident` is single-digit ms** for 16 experts at K3 geometry.

## What each outcome means

- **Prediction 1 holds and the rate is near device** — the lane is bandwidth-bound as
  designed, the number extrapolates, and the tier is doing its job.
- **Rate far below device** — there is a read-path problem to find, and this is the box to
  find it on (a quiet one).
- **`resident` is NOT single-digit ms** — the MXFP4 grouped GEMM is the cost, not the disk,
  and the whole "no dequant round trip" framing is answering the wrong question.

## What this will NOT say

**Not K3 tokens/s.** A real forward needs the ~79.5 GB always-active BF16 plus ~34.9 GB
shared/router, which this pod will not hold. This is the *expert block* of one layer, and
any extrapolation to 92 layers assumes every layer behaves like layer 1 and ignores
attention, routing and the residual entirely. That extrapolation is a sanity check, not a
serving figure, and must not be quoted as one.

## Cost

SECURE on-demand, `interruptible:false`, one modest GPU — 280 MB of routed experts needs no
large card. Container disk sized for one 17.0 GB shard plus a 15.7 GB slice arena. External
teardown backstop armed on the mini before the first run. Expected well under $5.
