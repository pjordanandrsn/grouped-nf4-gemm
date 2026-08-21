# Preregistration: an incremental structure for `_demote_locked`

Registered before implementing.

## What the accounting predicts

`RESULTS-bounding-the-residual.md` puts **90.9%** of the soft−hard gap in
`_demote_locked`, split **28.6% `cands` build / 63.7% selection / 7.4% apply**.
Both dominant phases are O(resident) per request. Replacing them with a lazy
heap — `_victim`'s #176 treatment — should remove the `cands` build and the
per-candidate ranking, leaving the apply loop and the guard.

## Predictions, registered

- **CONFIRMED** if the soft−hard CPU gap falls by **≳70%** (the 28.6+63.7 the
  heap replaces, minus overheads it adds) and `_demote_locked`'s own measured
  time falls correspondingly.
- **PARTIAL** if the gap falls materially but well short of that — the heap's
  own push/pop and compaction cost more than the scan it replaced at these
  sizes, as happened in #176's first attempt, where pushing on every touch made
  the heap *slower* than the sweep at rows=256.
- **REFUTED** if the gap does not fall, or rises. Then the accounting is
  measuring something other than what a structural change can remove.

## Correctness is the hard part, not the speed

A wrong demotion set does not raise: it changes which rows are reclaimable,
which changes resurrections, which changes reads. It would silently invalidate
every measurement in this campaign. So, as in #176:

- the current implementation survives as `_demote_scan_locked`, the oracle;
- `COLD_DEMOTE_VERIFY=1` compares the heap's victim **list** — order included —
  against the oracle on every call;
- equivalence over the seven trace configurations must be exact on every
  non-timing counter;
- each push site must be shown to *fail* the verifier when removed.

## The tie-break is the trap

`nsmallest(over, cands, key=…)` is stable, so equal `(freq, last_use)` ties
fall to position in `cands` — which is `_slot_of` **insertion order**, filtered.
Ties are reachable: `_last_use` is a per-request clock, so every key touched by
one `ensure()` shares a value. A heap keyed `(freq, last_use, key)` would break
those ties by key and silently reorder demotions.

So the heap carries a **publish sequence number** as its third element,
assigned when a key enters `_slot_of`. That reproduces insertion order exactly,
and it is unique so comparison never reaches the key.

## Stated in advance

The victim heap (`_vheap`) is **not** reusable here despite ranking the same
keys: it leads with the reclaimable flag and breaks ties by slot, where demote
filters reclaimables out entirely and breaks ties by insertion order. Merging
them would be a behaviour change wearing an optimisation's clothes.
