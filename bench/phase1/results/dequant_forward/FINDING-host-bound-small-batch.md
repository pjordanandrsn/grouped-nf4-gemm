# FINDING — at small batch this comparison is measuring Python, not kernels

**Measured directly on rented H100 (sm_90) and RTX 4090 (sm_89), report-only,
no prereg grades it.** `bench/phase1/probe_gpu_busy_fraction.py`; receipts in
`host_bound/`.

## The number

GPU busy fraction = summed CUDA kernel self-time per step ÷ wall time per step,
over 200 back-to-back steps with no syncs — i.e. how a training loop actually
runs.

| | decode band (`bs1`/`m8`/`m32`) | `tokbudget_2048` |
|---|---|---|
| **H100**, fused `G_base` | 9–33% (median 18%) | **80–92%** |
| **H100**, baseline `D_base` | **4.4–11%** (median 7%) | 24–76% |
| **4090**, fused `G_base` | 13–52% (median 30%) | **91–96%** |
| **4090**, baseline `D_base` | 4–55% (median 16%) | 43–99% |

On the H100 at decode band, `D_base` does **0.14–0.2 ms of GPU work inside a
~3.3 ms step**, and its wall time is nearly constant at 3.23–3.38 ms across
every decode regime — the signature of a fixed host cost, not a kernel cost.

## What it means for legs 1–3

**The token-budget results are genuine kernel measurements.** Both arms run
80–99% GPU-busy there, so `tokbudget_2048` and `tokbudget_11800` compare
kernels.

**The decode-band results are not.** F1 — the *primary* criterion of legs 2 and
3 — compares one kernel launch against a per-expert Python loop, with roughly
90% of both steps being host time. The three F1 medians (1.588, 1.522, 1.539
across three pairing schemes and two devices) are **real wall-clock step
ratios** and they replicate well, but calling them a kernel result is wrong.
They are step ratios in a regime where neither kernel is the bottleneck.

That is faithful to the published module — `QuantizedNaiveMoe` really does loop
in Python over hit experts — so charging it is fair. It also suggests part of
the "fixed ~2.5 s per-step dequant tax" in GenON's writeup is host overhead
rather than GPU dequant work, which would be worth their knowing.

## The asymmetry that cuts against gnf4

CUDA graph capture, each attempt in its **own process** (a failed capture
poisons the context, so a shared process cannot tell a real failure from
collateral damage):

| arm | H100 | 4090 |
|---|---|---|
| `D_base` (bnb dequant + `F.linear`) | **captured and replayed** | **captured and replayed** |
| `G_base` (fused), `expert_ids` as list | FAIL | FAIL |
| `G_base` (fused), `expert_ids` as device tensor | FAIL | FAIL |
| `G_full` (fused + LoRA) | FAIL | FAIL |

8 of 8 fused attempts failed; 4 of 4 baseline attempts succeeded. **The
dequant-on-forward baseline can be CUDA-graphed and gnf4's fused training path
cannot, as shipped** — and graphing is exactly how the host floor above would be
removed. CUDA's error ("operation failed due to a previous error during
capture") does not say which hazard fires, so the specific cause is not
established here; the candidates named before the run were the per-element
`int()` over `expert_ids` in `FusedGroupedNf4.forward` and the pageable
host-to-device copy `gemm_4bit_grouped` performs when handed a list.

Caveat on scope: this is "cannot as shipped", not "cannot". A change to
`kernel/nf4_qlora.py` may well make it capturable. And CUDA graphs need static
shapes, which MoE routing does not provide without padding or bucketing — so
graphing is a bound on how much host time is removable in principle, not advice
a user can act on today.

## A claim of mine this retracts

I earlier reported that leg 3's per-call timings ran 3.76× longer than leg 2's
on the H100 and attributed it to no-sync spans absorbing CPU stalls. **Both
instruments were re-run here on the same cell in the same process and they
agree: median 1.00× on both devices.** The original 3.76× compared leg 2's
figures from one pod against leg 3's from a different pod — a cross-run
comparison that this program's own rule forbids, and I made it anyway. There
was nothing to explain; the discrepancy was the comparison.
