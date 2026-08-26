# PREREG — K13 Stage A: WHICH host read breaks the MoE graph?

Registered 2026-08-26, before measurement. A census, not a treatment:
Stage A measures where the break is and registers nothing about
fixing it. Stage B is a separate registration.

## What K12 established

Compiling the MoE tier is **0.672 ms SLOWER** (6.888 -> 7.561 ms, A/A
0.05%). Every tracked raw-ATen row ROSE — `vectorized_elementwise_kernel`
85 -> 469 calls/step — so inductor emitted ~670 MORE launches per step
rather than fusing the chains. Cause: dynamo cannot trace that forward
as one graph, and compiling around the break fragments it.

SV2's 895.5 us in that region is therefore dispatch work wrapped
around a graph break, not unfused work waiting for a compiler
(RESULTS-k12).

## What is NOT established, and a hypothesis I already falsified

K12's log records one break record whose frame stack descends
`modeling_qwen3_moe:346 -> hybrid:480 -> hot_residency:{796,438,356,352,226}
-> nf4_grouped:1168`, reported as ``Graph break from `Tensor.item()` ``.

A stack is not an attribution. Reading the source for the host read,
I formed and then refuted a hypothesis, and it is recorded here so
this cycle cannot quietly inherit it:

- **Refuted before registration:** the routing `.tolist()` at
  `hot_residency.py:125-126` (`sizes = counts.tolist()`,
  `eids = uniq.tolist()`). Those sit in the **T > 1** branch. At the
  T=1 decode this campaign measures, the singleton branch runs
  instead: `sizes = [1] * x_rows.shape[0]` (pure Python off a shape)
  and `eids = local_ids` (stays a device tensor). **No host read.**
- **Still open, and not assumed:** `_all_hot`'s
  `bool(self.is_hot.all())` at line 226. It is memoised in
  `self._all_hot_cache` and its own docstring says "checked once per
  placement, never per step", so it can only break on a COLD first
  trace — which may or may not be the break that shapes the captured
  artifact.

K9 died twice from reading a capability as a fact about a run
([[attribute-from-the-profile]]). So Stage A does not reason from
source at all.

## The instrument: let dynamo name it

One flag on `step_decomp`, `--graph-break-census`, which wraps the
compiled step in `torch._dynamo.explain()` (or equivalently drains
`TORCH_LOGS=graph_breaks`) and writes, per break: **file, line,
enclosing function, dynamo's stated reason, and a count**. Nothing is
inferred from a traceback; the tool names the site.

Committed and reviewed BEFORE the box, like every instrument here.

## Stage A census cells

Knob-ON, `--placement-override all-vram`, b1d graph, the K12 frame:

1. `both_disabled` — the shipped configuration. Expected: no breaks
   attributable to the MoE tier, because it is never traced.
2. `moe_compiled` — `--compile-moe-tier`. The census cell.

## What Stage A reports (no PASS/FAIL bar — it is a census)

- The ranked break list for cell 2, with counts.
- For the top break: whether it fires **once at trace/capture** or
  **per step**, recorded separately. Under CUDA-graph capture a
  once-at-capture break still shapes every replay, so "once" does not
  mean "free" and must not be reported as such.
- Whether any break sits inside the MoE forward at all.

## REFUSE gates

- **Empty census in cell 2.** K12 observed a break; a census that
  finds none means the instrument did not observe what K12 did, and
  the disagreement must be resolved before anything is banked.
- **No break inside the MoE tier's frames.** Same reason: it would
  contradict K12's measurement rather than refine it.
- Cell 1 showing MoE-tier breaks — that region is not traced there,
  so breaks attributed to it would mean the census mis-attributes.
- The arms must carry K12's mechanism receipt (`dispatch_counts`),
  knob-ON, or the cells are not the configuration K12 measured.

## What Stage A explicitly cannot conclude

That the named break is **removable**, or that removing it would let
compilation pay. K12 measured compilation as a 0.672 ms REGRESSION;
nothing here licenses re-running that treatment. Stage B registers
that question only if Stage A names a break that is ours to change.

## Frame note

K11 closed the 250 target and M3 shipped 6.803 ms as the default.
This cannot reopen either. Its value is directional: it says whether
the MoE tier's dispatch residue has an addressable cause, or whether
the campaign should stop looking there.

## Receipts

`kernel/receipts-k13/` — the two cells' censuses, box_meta with the
anchor probe, mechanism receipts. `k13_verdict.py` (self-tested) is
committed BEFORE the box.
