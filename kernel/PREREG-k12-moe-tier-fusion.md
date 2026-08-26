# PREREG — K12: is the MoE tier's dynamo exclusion still necessary?

Registered 2026-08-26, before measurement. Grounded in an accounting
pass over SV2's committed census, not a new hypothesis — and the
accounting is stated so it can be checked, because K9 died twice from
one that could not be ([[attribute-from-the-profile]]).

## What the census says

Removing attention (K8's lane), both GEMV families, the router (K10's
lane) and the KV append from SV2's 6.46 ms step leaves:

```
1258.2 us/step over 983 launches  (1.28 us per launch)
  already inductor-fused (triton_*)   362.7 us over  5 row-types
  raw ATen kernels                    895.5 us over 12 row-types
```

The largest raw rows are `unrolled_elementwise_kernel` (245.9 us
x218), `indexSelectS…` (234.0 us x145), `elementwise_kernel<128,4>`
(170.6 us x96) and `reduce_kernel` (109.5 us x48) — small work at
~1.3 us apiece, which is dispatch-shaped, not arithmetic-shaped.

## Why they are unfused, in the code's own words

`--compile-layers`' help text: *"the paged-attention fn **and the MoE
tier forward** are dynamo-disabled so they graph-break cleanly
(PREREG-t1-launchpath)"*. The exclusion is deliberate and registered.
Inductor never sees that region, so those ATen chains are never
fused.

The attention half of the exclusion has a **known, specific cause**
(F1 Stage B): under inductor the paged-decode kernel is re-emitted
through the user-kernel path and dies on a loop-carried `m_i` typed
fp32 then fp64. **The MoE half's necessity is not separately
established** — it was disabled alongside attention.

## The disable site states its own reason, and it may not apply

The two exclusions are applied together, with distinct reasons given:

> *"clean graph breaks: the paged-attention shim (**host-bound KV
> paging**) and the hybrid MoE forward (**CPU tier dispatch**) must
> never be traced -- compile owns only the dense layer body"*

The MoE reason is **CPU tier dispatch** — the hybrid engine's
DRAM/NVMe tier work, which dynamo cannot trace. But every certified
serving measurement in this campaign runs
`--placement-override all-vram`: SV1, SV2, K7's anchor, K8, K10, M2.
**At the all-vram point there is no CPU tier to dispatch to.**

That does not prove the exclusion is unnecessary — the forward may
carry other untraceable work, and `all-vram` is a placement override
rather than a different code path. It makes the hypothesis specific
and checkable, which is the bar this cycle owes after K9:

> Can the MoE tier be compiled while the paged-attention fn stays
> disabled, **at the all-vram placement the certified ladder
> measures**? If yes, inductor fuses those chains for free.

## Instrument required (small, and named here so it is not smuggled in)

One flag on `step_decomp`: keep
`ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(...)` while
skipping the `for m in mods: m.forward = dynamo.disable(m.forward)`
loop. Nothing else changes. Committed and reviewed BEFORE the box,
like every instrument in this campaign.

## Stage A — necessity probe (one box, cheap)

Four arms, A/A pairs, knob-ON (the SV2/K8 frame):

1. `both-disabled` — the shipped configuration, the baseline.
2. `moe-compiled` — attention still `dynamo.disable`d, MoE tier NOT.
3. `both-compiled` — the control that should reproduce F1's failure.
   If it does NOT fail, F1's exclusion may itself be stale, and that
   is a finding to record rather than bury.
4. `moe-compiled` **profiled** — a replay kernel census, to see
   whether the raw ATen rows actually became `triton_*` rows.

**REFUSE** if arm 2 raises, produces recompiles inside the timed
window, or changes the greedy token stream versus arm 1 — compiling a
region must not change what the model says.

## Bars (Stage B is only the productisation; the cut is measured here)

Against the raw-ATen total the census attributes to the excluded
region (**895.5 us**), measured as the step delta of arm 2 vs arm 1:

- **PASS**: ≥ **0.40 ms** off the step (≈45% of the raw-ATen total).
- **PARTIAL**: ≥ **0.15 ms**.
- **REFUTED**: < 0.15 ms — the exclusion is not what those kernels
  cost, and the RESULTS must say where the residue actually lives.

A step delta, not a kernel-count delta: fewer launches that do not
move the step is not a win.

## Attribution gate (Stage A cannot proceed without it)

The census claims 895.5 us sits in the excluded region. Arm 4 must
**show it**: the raw ATen row-types named above must fall in count.
If arm 2 gets faster while those rows are unchanged, the speed came
from somewhere else and the attribution is wrong — REFUSE rather than
bank a number against a mechanism that did not move.

## AMENDMENT (2026-08-26, before the box) — arm 3 needs a second flag

The **Instrument required** section above names one flag and says
"Nothing else changes". But Stage A lists **four** arms, and arm 3
(`both-compiled`) needs the paged-attention `dynamo.disable` lifted
too — which nothing in the harness could do, because that disable is
applied unconditionally whenever `--compile-layers` is on. Arm 3 was
therefore not runnable with the registered instrument.

`--compile-attn-tier` (experts4bit-qlora#289) fills the gap, and is
named **here** for the same reason the first flag was: a prereg whose
instrument section says "so it is not smuggled in" does not get a
quiet second instrument. It requires `--compile-moe-tier` — alone it
would be an attention-only-compiled arm this prereg never registered
and no bar would interpret.

Arm 3 remains **recorded, not gated**: `k12_verdict` already treats
`both_compiled` as optional, and the bars are the arm-2-vs-arm-1 step
delta. Its purpose is the finding the Stage A text already names — if
the both-compiled control does *not* fail, F1's exclusion may itself
be stale.

Found alongside a second defect in the merged instrument: the
post-compile summary line printed "paged attention + MoE tier
dynamo-disabled" unconditionally, including in the `--compile-moe-tier`
arm where the MoE tier is not disabled, contradicting the K12 line
printed a few lines above it. Arm logs are read back as receipts, so
that line now names what was actually disabled.

## Frame note

K11 closed 250; this cannot reopen it. At its PASS bar the step goes
6.281 → ~5.9 ms ≈ **170 tok/s**, which is a real improvement to the
certified path and not a route to the retired target.

## Receipts

`kernel/receipts-k12/` — four arms, the replay kernel census for arms
1 and 2, token streams, box_meta. `k12_verdict.py` (self-tested) is
committed BEFORE the box.
