# PREREG — decomposing the routed residual, and pricing the copy-count fix before building it

**Tier: CONFIRMATORY on R1–R3 (gates), EXPLORATORY on R4 (the decomposition).
Status: DRAFT — must be committed and OTS-stamped BEFORE the pod is created.**
Code: gnf4 @ `9a802c4`; e4b control @ `e62a7a0` (`private`), e4b treatment @
`94e4004` (`perf/routed-staging-overhead`). Both local, unpushed.

## What this asks

Finding #22 left the routed step at **0.936 s/token against a 0.445 s byte
model** — a **5.2 ms/layer** non-byte residual, attributed in one sentence to
"the per-layer host sync and ~32 small copies replacing 4 large ones", and it
declared throughput unquotable until that residual is characterized.

Two fixes follow from that sentence and **they are not the same size**. One is
written (`94e4004`: drop the device sort ahead of the sync, flatten the per-row
dict churn). The other — coalescing the copies — is not, and should not be
written on the strength of a phrase. This prices them.

**The arena does not reach the routed path, and that is a finding this prereg
depends on.** `E4B_OFFLOAD_ARENA` packs the four homes into one contiguous
pinned buffer *per dtype* and cuts bulk staging from 4 copies to 1–2. But the
layout is **name-major** — all experts of `gate_up_proj`, then all of
`down_proj` — so `home[n]` remains a strided view and `_copy_rows_into` still
issues one copy per (routed expert × tensor). A routed-path coalescer is
therefore **not** "enable the arena"; it needs an **expert-major** layout, and
because the weights and absmax differ in dtype it lands at one copy per
(expert × dtype) unless the arena is rebuilt as a byte buffer with cast views.
That is a real change to `_build_homes`, `_copy_rows_into`, and the state-dict
hook's view contract. Worth measuring before worth building.

## Fixture

Qwen3-235B-A22B, 2×A100-SXM-80GB, NF4 experts pinned, `prefetch=False`, routed
staging on, natural prompt, greedy, ctx 512, 12 new tokens, median of 2, **one
process**, `E4B_OFFLOAD_STATS=1`, arms **interleaved rather than blocked** (the
v2-confirmatory methodology note: a blocked sweep under-predicted harness-context
sensitivity).

**Speculative staging OFF and the expert cache OFF in every arm.** Both landed on
`private` after #22 (spec measured 1.330× on its own) and both change which
bytes cross the link. Leaving either on would confound every comparison against
the 0.936 s baseline this prereg is anchored to. They get their own prereg.

| arm | code | purpose |
|---|---|---|
| **C** | `e62a7a0` | control; also the instrumented decomposition run |
| **T1** | `94e4004` | sync fix |
| **C'** | `e62a7a0` | control repeated — the self-pair noise floor |

## Predictions

- **R1 — GATE, bit-identity.** Greedy ids C == T1, exactly. *Falsified by any
  divergence, and a divergence is a **STOP**, not a slower result:* `_routed_ids`
  decides which rows get copied, so a mismatch means a row the router asked for
  was never staged and the kernel read uninitialized memory. That is the precise
  failure routed staging exists to rule out.

- **R2 — GATE, engagement.** In T1, the device-unique path is entered **zero**
  times during decode (instrumented counter; the CPU suite already asserts this
  locally in `tests/test_routed_ids.py`). Registered because equality testing is
  structurally blind here: both branches return identical ids, so a fast path
  that never fires passes every correctness check and reports as "no measurable
  change". That is the `enable_fast` failure — correct, and dead on every
  offloaded model until #22 caught it. *Falsified by any nonzero count.*

- **R3 — no regression.** T1 s/token ≤ C × (1 + the C/C' spread). *Falsified if
  T1 is slower than C by more than the self-pair spread.*

- **R4 — THE DECOMPOSITION.** This is the reason to run at all. From C's
  `offload_stats_report`, the **routed-policy implied GB/s** against the probed
  pinned ceiling (`report_offload_environment`, 22.21 GB/s at #22).
  **Registered prediction: implied GB/s ≤ 0.70 × ceiling** — i.e. transfer
  inefficiency, not host stall, is the majority of the 5.2 ms.
  *Rationale:* at 235B shapes a layer's routed set is ~85 MB across 32 copies,
  ~2.7 MB each — below where an H2D reaches asymptotic bandwidth — whereas the
  sync is a round trip of tens of microseconds and the loop is ~32 dispatches.
  *Falsified above 0.70 ×,* which would mean the transfers are already near
  efficient, the residual is host-side, and T1 is the right lever.

- **R5 — DECISION RULE, registered ahead of the data** so the conclusion cannot
  be chosen after seeing it:
  - **R4 holds (≤0.70×)** → build the expert-major coalescer. Its ceiling is the
    measured gap; anything claimed beyond that gap is unsupported.
  - **R4 falsified (>0.70×)** → **do not build it.** The copy count is not
    costing what #22's phrasing implied. Record the negative and move the lane
    to host-side stall.
  - T1 stands or falls on R1–R3 independently of R4 either way.

- **R6 — T1's magnitude, registered deliberately small: T1/C ∈ [0.95, 1.00].**
  Removing an 8-element device sort saves kernel-launch time, not milliseconds;
  T1's real value is whatever share of the residual R4 leaves on the host side.
  *Above 1.00 is a regression (see R3).* **Below 0.95 the band was wrong** — that
  is recorded as a miss and explained, not claimed as a win. An out-of-band
  result means the model of where the time goes is wrong, which is information,
  not success.

## Cost and teardown

One 2×A100-SXM pod, fewer arms than #22's ~$15 session. **Hard cap $35/job**
per the standing discipline: watchdog with an extendable deadline file, evidence
rsynced off-box continuously, delete-then-verify-404, teardown proven on a
throwaway before any GPU-hour.

## Not claimed

- **Nothing about K3.** The fixture is Qwen3-235B on a specific host. `c_box`,
  any achieved fraction, and any GB/s here do **not** transfer to another host
  class or model — the additive law is per-box and measured, never carried.
- **Nothing about tok/s as a product number.** The byte model stays 2.1× off
  until R4 is answered; #22's "unquotable" stands and this prereg does not lift
  it. R4 is a step toward re-fitting the model, not a throughput claim.
- **Nothing about speculative staging or the expert cache**, which are switched
  off here precisely so they can be measured honestly later.
