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
