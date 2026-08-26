# RESULTS — K12: REFUTED. The exclusion is not stale; it is load-bearing

Measured 2026-08-26 on one RTX 5090 (170 SMs, machine 114071), one
provisioning. Receipts in `receipts-k12/`; the verdict reproduces with
`python kernel/k12_verdict.py kernel/receipts-k12/k12_report.json`.

```
K12 VERDICT: REFUTED  (cut -0.672 ms; PASS >= 0.4, PARTIAL >= 0.15)
  step 6.888 -> 7.561 ms
```

| arm | a / b | A/A | recompiles |
|---|---|---|---|
| `both_disabled` (shipped) | 6.897 / 6.879 | 0.26% | 0 |
| `moe_compiled` | 7.559 / 7.563 | 0.05% | 0 |
| `both_compiled` (control) | **raised** | — | — |

Knob-ON throughout (`dotpad=384 scalar=0`). Baseline 6.888 ms inside
the committed anchor gate scaled by the paired knob ratio.

## The answer: compiling that region makes it SLOWER

Not "no benefit" — a **0.672 ms regression**, 9.8% of the step, with
A/A noise of 0.05%. The registered hypothesis was that inductor never
sees the MoE tier, so its raw-ATen chains are never fused; let it see
them and they fuse for free.

The census says the opposite happened. Every tracked row **rose**:

| row | before | after | |
|---|---|---|---|
| `unrolled_elementwise_kernel` | 218 | 314 | +96 |
| `vectorized_elementwise_kernel` | 85 | **469** | +384 |
| `::elementwise_kernel` | 145 | 241 | +96 |
| `indexSelect` | 145 | 145 | flat |
| `reduce_kernel` | 50 | 146 | +96 |

Compiling the tier added **~670 elementwise launches per step**.
Inductor emitted more small kernels, not fewer.

## Where the residue actually lives (the prereg asked for this)

Dynamo cannot trace the MoE forward as one graph. It breaks on a
host sync:

```
Graph break from `Tensor.item()`
  modeling_qwen3_moe.py:346, in  hidden_states = self.mlp(hidden_states)
```

So the 895.5 us of raw ATen the SV2 census attributed to this region
is **not unfused work waiting for inductor**. It is the tier's actual
dispatch work, wrapped around a device-to-host read that forces a
graph break. Compiling around that break fragments the region
further — inductor compiles the pieces between the breaks separately
and adds guard and dispatch overhead, which is where the +670
launches come from.

The lever this points at is **not** compilation. It is the `.item()`:
until the routing decision is device-resident, that region cannot be
one graph, and anything that tries to compile it pays fragmentation
instead of fusion.

## Arm 3: F1's exclusion is not stale either

The prereg registered arm 3 as a control and said that if it did NOT
fail, "F1's exclusion may itself be stale, and that is a finding to
record rather than bury". It failed, with exactly the mechanism F1
Stage B named:

```
triton.compiler.errors.CompilationError: at 54:4:
AssertionError("Loop-carried variable m_i has initial type <['16'], fp32>
               but is re-assigned to <['16'], fp64> in loop!
```

Both exclusions stand, for their own stated reasons.

## Two corrections this cycle forced

**The attribution gate ran unconditionally.** PREREG-k12 scopes it to
"if arm 2 gets FASTER while those rows are unchanged". The code
dropped that clause, so a slower arm with unmoved rows returned
REFUSE where REFUTED is honest — nothing is unattributed about a
treatment that did not pay. Corrected in gnf4#292 from arm 2's timing
alone, before its census existed, and the change can only turn a
REFUSE into a REFUTED. The on-box run used the pre-correction
calculator and printed REFUSE; this document's verdict is the
corrected one, re-run locally over the committed receipts.

**Arm 3 needed an instrument that was never registered.** Stage A
lists four arms but the "Instrument required" section named one flag,
and the paged-attention disable is applied unconditionally under
`--compile-layers`. `--compile-attn-tier` was added as a named
amendment (gnf4#287, e4b#289); review then caught that it only
skipped the INITIAL disable while `_b1d_stage_a` re-wrapped the raw
shim, so arm 3 would have run as arm 2 with a receipt claiming
otherwise.

## Scope

One box. The absolute numbers are this box's (M2: 8.5% inter-box
dispersion); the cut is a same-box delta.

Incidentally settled: `k12_verdict` carried a caveat that the
dot-pad knob ratio's box-invariance was unmeasured. Three boxes now
agree — 6.476/7.25 = 0.8932 (certifying), 7.03/7.84 = 0.8967 (M3),
6.888/7.70 = 0.8946 (here) — a 0.4% spread against 8.5% dispersion in
the absolute times. The ratio is stable even though the step is not,
which is what makes scaling the anchor gate by it defensible.
