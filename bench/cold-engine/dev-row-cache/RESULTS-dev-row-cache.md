# Device row cache — first hardware validation

Receipt: [`receipt-3060.json`](receipt-3060.json). Box: RTX 3060 12 GB,
torch 2.13.0+cu130, triton 3.7.1, driver 580.159.03, destroyed after the pull.

**This is not a performance result and does not claim to be.** A 3060 is not
reference silicon and no timing was taken. What is measured here is an
equivalence and a set of mechanism counts, both of which are properties of
the code rather than of the card.

## What passed

| claim | result |
|---|---|
| cache on vs cache off, 24-step trace | **max abs diff 0.0 — bitwise identical** |
| rows filled vs routed cold expert-slots | **19 / 96** |
| host→device bytes vs the uncached arm's cold PCIe bytes | **496,128 / 2,297,856 = 21.6%** |
| tier misses (reads that reached the pinned tier) | **54 → 17** |
| resurrections (reclaimable rows re-hit before overwrite) | **30**, `reuse_before_overwrite` **0.714** |
| stalls at `rows = 2k` | **0** |

The equivalence is the load-bearing one. The cache relocates *where* a row is
read from and must never change *what* the row is; a single differing bit
would mean it had reinterpreted packed bytes, which the cold path is
categorically forbidden to do. 0.0 over 24 steps, with 45 logical evictions
and 12 overwrites in the middle of them, is that claim held under churn.

`reuse_before_overwrite = 0.714` is R2/R3's quantity measured on a real
device for the first time — 30 of 42 resolved evictions were re-hits, not
overwrites. It is not a preregistered gate result and is not scored as one.

## What this does NOT show

**The working set is tiny and maximally favorable.** The fixture has E = 8
experts, k = 4 routed, and a cache of 8 rows — the cache can hold *every
expert in the layer*. That is the best case that exists, and the 78%
byte reduction should be read as "the mechanism works", not as a number that
transfers. A real MoE layer has 128+ experts against a cache sized far below
E, where the hit rate is a routing-locality question this fixture cannot
answer. The honest next measurement is that one, on a real routing trace.

**Nothing here is timed.** The cache trades one extra device-side write per
miss against a PCIe transfer it does not avoid, so whether the byte
reduction becomes a latency reduction is a separate measurement on
reference silicon, against the positional cache that already exists.

## What the validation changed in the code

The first run **failed**, and the failure was real rather than a test bug:
`VramSlots` demotes *after* it allocates, so the previous step's `k` rows are
still ACTIVE — and unprovably quiescent — while the current step claims its
own `k`. A step that misses on all `k` therefore needs `k` free rows beside
them, and `settle()` cannot help because those rows are ACTIVE, not RETIRING.

The engine's guard had required only `rows > k`. It now requires
`rows >= 2k` and says why. Demoting the displaced rows against the
*previous* step's tag before allocating would relax this to `k + 1`; that is
deliberately not done here, because it reorders an allocator that is still
under review and had two live-row bugs found in it this week.
