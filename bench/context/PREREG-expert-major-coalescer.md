# PREREG — the expert-major coalescer

**Tier: CONFIRMATORY.** This is the branch `PREREG-routed-residual` R5
pre-committed to, not a new idea: *"R4 holds (≤0.70×) → build the expert-major
coalescer. Its ceiling is the measured gap; anything claimed beyond that gap is
unsupported."* #52 established **R4 holds at 0.59×** once `gbps` was corrected
to wall time.

## What is actually wrong today

`_copy_rows_into` issues **one `copy_` per (routed expert × tensor)**. At 235B
shapes that is ~32 copies per layer of ~2.7 MB each — below where an H2D reaches
asymptotic bandwidth. Measured: **8.66 GB/s against a 14.68 GB/s pinned ceiling
= 0.59×.**

**The existing arena does not fix this**, and that is the finding this prereg
depends on. `E4B_OFFLOAD_ARENA` packs the four homes into one pinned buffer *per
dtype*, but the layout is **name-major** — all experts of `gate_up_proj`, then
all of `down_proj` — so `home[n]` stays a strided view and the per-row loop
survives. A routed-path coalescer needs an **expert-major** layout, landing at
one copy per (expert × dtype), or one per expert if the arena is rebuilt as a
byte buffer with cast views.

Touches `_build_homes`, `_copy_rows_into`, and the state-dict hook's view
contract.

## Predictions

| quantity | prediction |
|---|---|
| copies per layer | 32 → **8–16** (one per expert, or per expert×dtype) |
| routed implied GB/s | **10.5–13.0** (0.72–0.89× ceiling, from 8.66) |
| routed s/token | **0.78–0.87** (from 0.9223) |
| **routed step speedup** | **1.06–1.18×** |

**The upper bound is 1.70×** and only if the step were *pure* transfer. It is
not: attention, norms, the router and the expert GEMM are inside that 0.9223 s,
and #40 measured experts at 71.3 % of an e4b step. **Anything above 1.20× should
be disbelieved before it is celebrated** — that is the shape of a measurement
error, not a win.

## Gates — bit-identity is not negotiable

- **Greedy ids and logits identical to the control**, `max|Δlogit| = 0.000e+00`.
  Routed staging's entire claim is that it changes *which bytes move*, never what
  is computed. A coalescer that reorders or mis-slices rows breaks that, and the
  failure is silent: the kernel reads whatever is in the destination buffer.
  **Any divergence is a STOP, not a slower result.**
- **Copy-count must actually fall.** Instrumented, asserted. If the count does
  not drop, the coalescer did not engage and any timing difference is noise —
  this is the R2 lesson: a fast path that never fires passes every correctness
  check (`enable_fast` was dead on every offloaded model until #22).
- **`gbps` is the wall-time metric** (e4b `fix/routed-gbps-wall`, merged). Do not
  compare against `gbps_copy_window`.

## Decision rules, fixed now

- **≥1.06× on BOTH cards, bit-identical, copy-count down** → land it.
- **1.00–1.06×** → record; do **not** land. It touches the pinned-memory layout
  every offload path depends on, and a sub-6 % win does not pay for that risk.
- **<1.00× or any divergence** → revert. The copy count was not costing what the
  byte model implies, and the routed lane closes.

## What would make this VOID

- A single-card conclusion (#43 shipped a 2.2× regression that way; #44 shipped
  a wrong-answer kernel that was bit-identical on one card).
- Timing on the QNAP — that box is correctness-only.
- Any arm where the copy-count instrumentation is absent.

## Not claimed

- **Nothing about speculative staging or the expert cache.** Both change which
  bytes cross the link and both are off in every arm, exactly as in
  `PREREG-routed-residual`.
- **Nothing about the gnf4 kernel.** #50/#51/#53 closed that line; this is the
  *transfer* side and is independent of it.
