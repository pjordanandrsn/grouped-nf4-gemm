# PREREG — K9: the decode grouping builder, fused

Registered 2026-08-26, before measurement. Follows from an accounting
pass over RESULTS-sv2's committed census rather than from a new
hypothesis: the row labelled "router" is not all router.

## What the accounting found

SV2's 6.46 ms knob-ON step carries 311.4 us/step under `router`,
which is two kernels at 48 calls each — one per layer:

| kernel | us/step | per call | what it is |
|---|---|---|---|
| `sbtopk::gatherTopK` | 174.0 | 3.6 us | the MODEL's top-8-of-128 router |
| `bitonicSortKVInPlace` | 137.4 | 2.86 us | **ours** — `torch.argsort` inside `build_group_tiles_device` |

The second is gnf4 code sorting **eight int64s**, once per layer, at
2.86 us a call. And the argsort is only its most visible piece:
`build_group_tiles_device` issues ~26 distinct torch ops per call
(argsort, zeros, scatter_add, two cumsums, arange, searchsorted, four
index_selects, three wheres, clamps, dtype casts) to group r=8 rows
into at most 8 groups at `block_m=1`. Those land in the census's
generic `elementwise` (937.5 us / 692 launches) and `other`
(971.1 us / 432 launches) buckets, where per-launch costs are 1.35 us
and 2.25 us — i.e. dominated by launch overhead on trivially small
work.

This is general-case machinery running a tiny special case, 48 times
per step.

## Stage A — attribution (a census; it cannot fail a bar)

Measure what `build_group_tiles_device` actually costs at the decode
shape (`r=8`, `n_experts=128`, `block_m=1`) on the box:

1. Per-call wall and device time, chunked-median, times 48 layers.
2. Its launch count per call, from a profiled window.
3. **A confirmation gate, not an assumption**: the census's
   `bitonicSortKVInPlace` must be attributable to this function — if
   the observed argsort count per step is not 48, the attribution
   above is wrong and Stage A REFUSES rather than proceeding on a
   mis-attributed premise ([[eliminate-versus-account]]).

Stage A publishes the function's measured share `X` ms/step. No
treatment is judged against an estimate of `X`.

## Stage B — the fused builder

One Triton kernel producing the same five tensors
(`row0, rows, grp, order, counts`) for the decode shape, replacing
the op chain. The general path is untouched and remains the
implementation for every shape outside the fused case; selection is
by shape, not by env.

### Correctness (STRICT, and the mechanism permits strict)

Unlike the fp8-attention lane, this output is **exact integer
structure** — a selection and a set of offsets, not arithmetic. A
fused builder can and must be **bitwise-identical**:

- **G1**: all five returned tensors `torch.equal` to the current
  builder's, across (a) exhaustive distinct-id cases at r=8, (b)
  randomised cases including REPEATED expert ids (T>1 shapes reach
  this even though T=1 top-k does not), (c) the r=0 and
  all-same-expert edges. Any mismatch REFUSES; there is no tolerance
  band, because there is no rounding in the mechanism
  ([[correctness-bars-derive-from-the-mechanism]]).
- **G2**: run-to-run bitwise determinism on the fused builder.
- **G3**: `order` must match the CURRENT builder's tie-breaking
  (`argsort(stable=True)`), not merely be *a* valid grouping — the
  captured graph and every downstream identity gate depend on the
  exact permutation.

### Bars (ratios against Stage A's own measurement)

`X` is unknown until Stage A runs, so the bars are fractions of it —
the K7 posture, where an absolute bar would have been a guess:

- **PASS**: fused builder cuts its own measured cost by **>= 60%**.
- **PARTIAL**: **>= 30%**.
- **REFUTED**: < 30% — the builder was not the cost, and the RESULTS
  must say where the residue actually lives instead.

A step-level delta is RECORDED alongside but is not the bar: at
~1.3 us per launch the win is launch-count-bound, and step-level
attribution at this magnitude is within the noise a single A/A pair
can resolve.

## REFUSE gates

- Stage A's attribution gate above.
- A/A spread <= 2% on the paired step measurements, tokens identical.
- Anchor health: knob-ON step within +/-5% of the certified
  6.476 ms point.
- Same box for both arms.

## Frame note (stated before measurement)

Even a total win here does not close 250. The measured pool after K7
(0.24 ms) and K8 (0.217 ms) is ~1.38-1.48 ms against a 2.48 ms bar,
and this lane's entire addressable content is a subset of the
~1.70 ms non-attention residue. K9 is registered because it is
measurable, mechanism-named, and ours to fix — not because it
rescues the composition route. Whether 250 survives is decided by
what remains after this, and the RESULTS will say so plainly either
way.

## Receipts

`kernel/receipts-k9/` — Stage A attribution, the correctness matrix,
paired step receipts, box_meta with the anchor probe. `k9_verdict.py`
(self-tested refusal directions) is committed BEFORE the box cycle.

## AMENDMENT (2026-08-26, before measurement) — the bucket
## attribution does not reconcile, and is withdrawn as a premise

Found while preparing the kernel, before renting anything. The
section above asserts that the builder's ~26-op chain "lands in the
census's generic `elementwise` and `other` buckets". That is an
inference, and it does not survive arithmetic:

- A CPU dispatch profile of one `build_group_tiles_device` call at
  the decode shape counts **66 plausibly-launching aten ops** (95
  total, less pure allocation/view ops).
- At 48 layers that is **3168 launches/step**, against a census total
  of **1220** non-matmul launches/step (elementwise 692 + other 432 +
  router 96). The builder alone would exceed the whole bucket by 2.6x.
- More specifically: the builder issues `index_select` 4x per call,
  which at 48 layers is 192 — already more than the census's
  `indexSelectS...` row at **145** calls/step.

Something in the chain is false: aten dispatch counts are an UPPER
bound on kernel launches (a same-dtype `to` launches nothing, and the
kernel-view census deliberately excludes `aten::` op rows to avoid
double-counting), or the builder is not called once per layer, or
both. Source inspection cannot settle it.

**What survives as measured fact:** `bitonicSortKVInPlace` fires 48
times per step at 137.4 us total, the builder contains exactly one
`torch.argsort`, and no other argsort was found in the decode path.
That 137.4 us is the lane's demonstrated FLOOR.

**What is now explicitly unknown:** everything above that floor. The
lane's ceiling is whatever Stage A measures, not what the narrative
above estimated.

This does not change any bar — Stage B's bars were already fractions
of Stage A's own measurement precisely because X was unknown, and
Stage A already carried a refusal gate on the attribution. It changes
what this prereg is allowed to CLAIM before that measurement, which
is: one sort per layer, 137.4 us, and an open question
([[eliminate-versus-account]] — the accounting that produced this
lane also has to survive being checked).

## VOID (2026-08-26, before any measurement) — the registered
## mechanism is not on the path this cycle measures

The first amendment withdrew the bucket attribution as unreconciled.
Checking the call site settles it harder: **`build_group_tiles_device`
is never called on the decode path.**

`hot_residency.py` dispatches with

```
singleton_groups=(T == 1 or (FORCE_SINGLETON_GROUPS[0]
                             and not DEVICE_GROUPING[0])),
device_grouping=(DEVICE_GROUPING[0] and T > 1),
```

At decode `T == 1`, `singleton_groups` is unconditionally **True** and
`device_grouping` unconditionally **False**, regardless of either
flag. The device builder is an opt-in path for the T>1 speculative and
batched arms (`DEVICE_GROUPING = [False]` by default), and SV2's
census — like every single-stream measurement in this campaign — ran
at T=1.

So this prereg's Stage B has no subject. **K9 is VOID.** No box was
rented and no bar was measured against it; the cost was the writing.

**What the two failures have in common,** recorded because the lane's
whole premise came from an accounting pass: both times the error was
reading a *capability* in the source as a *fact about the run*. The
builder exists, is capture-safe, and is genuinely used — by arms this
cycle did not run. Counting ops in a function proves nothing about
whether the measured step called it. Attribution has to start from
what the profile says executed, then find its owner — not from a
plausible owner, then assume it executed
([[eliminate-versus-account]]).

The 311.4 us router row is real and re-registered as K10, whose
Stage A identifies the owner **before** any treatment is designed.
