# The demote scan is gone: 76–82% of the soft-hard gap removed

**Registered outcome: CONFIRMED** (predicted ≥70%).
Preregistered in [`PREREG-demote-heap.md`](PREREG-demote-heap.md) before
implementing, on the strength of the accounting in
[`RESULTS-bounding-the-residual.md`](RESULTS-bounding-the-residual.md).

## Result

Two measurements, two instruments:

| | gap before | gap after | closed |
|---|---|---|---|
| interleaved A/B vs `main` (noise ref 6%) | 0.638 s | **0.152 s** | **76.1%** |
| accounting instrument | 0.633 s | **0.113 s** | **82%** |

`_demote_locked`'s own measured time: **0.575 s → 0.102 s**. The soft arm now
costs **14% more CPU than hard**, down from ~85% more. Read counts identical.

The accounting *improved* rather than degrading — 92.8% → **97.0%** of the
remaining gap still attributed — so the scan's cost was removed rather than
displaced somewhere unexplained.

## What changed

`_demote_locked` built an O(resident) candidate list and evaluated a rank for
every candidate, on every request, to choose ~4 victims. It now pops a lazy
heap of `((freq, last_use, pubseq), key)` over eligible rows.

Stale entries are dropped; entries whose rank moved on are re-filed at the
current rank. The self-heal is sound for the same reason as `_victim`'s: a
touch only ever *raises* `freq` and `last_use`, so an entry sinks and cannot
hide a better candidate. **Becoming eligible** is the change self-healing
cannot cover — a resurrected row must re-enter the heap — and it has an eager
push.

`_vheap` was deliberately **not** reused. It ranks the same keys, but leads
with the reclaimable flag and breaks ties by slot, where demote filters
reclaimables out entirely and breaks ties by insertion order. Sharing it would
have been a behaviour change wearing an optimisation's clothes.

## The tie-break was the trap, and it was real

`nsmallest` with `key=` is stable, so equal `(freq, last_use)` fall to position
in `cands` — `_slot_of` insertion order. **Ties are reachable**: `_last_use` is
a per-request clock, so every key one `ensure()` touches shares a value. A heap
keyed `(freq, last, key)` would break those ties by key and silently reorder
demotions.

The heap therefore carries a **publish sequence number**, assigned at the single
site that writes `_slot_of`, and dropped on eviction so stale entries fail
validation and the map stays bounded. Replacing that element with a constant is
one of the sabotages below, and it is caught.

## Verification

A wrong demotion set raises nothing. It changes which rows are reclaimable,
hence resurrections, hence reads — and would silently invalidate every
measurement in this campaign. So, as in #176:

- `_demote_scan_locked` survives as the oracle, callable `dry=True` to return
  victims without applying them;
- `COLD_DEMOTE_VERIFY=1` compares the heap's victim **list, order included**, on
  every call;
- all seven trace configurations are identical with **both** verifiers on;
- `test_demote_heap_agrees_with_the_scan` puts the property in CI.

**Every sabotage is caught**, which is what makes the verifier worth having:

| removed | caught by trace verifier | caught by CI test |
|---|---|---|
| publish push | yes | yes |
| resurrect push | yes | yes |
| publish-sequence tie-break | yes | — |
| demand-window exclusion | yes | — |

425 tests pass.

## What this does not show

Toy arena, one trace, one capacity; CPU time on a laptop, where the hard-arm
noise reference is 6% of the gap — so 76.1% is 76 ± 6, not 76.1. The two
instruments disagree by 6 points for that reason, and both clear the registered
threshold, which is what the prediction was written to decide.

This removes the *tier-side* cost. Whether it moves #153's wall-clock residual
is a separate question on real NVMe, unanswered here — the accounting says the
gap it removes is the one that was left after #175/#176, but "says" is not
"measured".

## Receipts

`../routing-trace/bound_soft_dheap.json`. Instruments
`../routing-trace/bound_soft_overhead.py`. Offline, no GPU, no spend.
