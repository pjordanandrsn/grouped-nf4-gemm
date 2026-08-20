# The resurrection-off control #152 could not run

Receipts: [`wall-rows-minus-k.json`](wall-rows-minus-k.json),
[`wall-protected-half.json`](wall-protected-half.json), logs alongside.
Box: RTX 3060, 31 GB RAM, real NVMe, **pinned** tier, destroyed after the pull.

**#152 already refuted R2**, on the right tier (`Mxfp4NvmeResidency`) with a
real gpt-oss-20b arena, and it stands. This is not a second attempt at that.
It supplies the one thing #152 says it could not:

> That a resurrection is worthless. It's an avoided transfer by construction,
> so holding a configuration fixed and disabling them could only make it
> slower. **No such control exists** — `protected = rows` leaves nothing
> demotable and `_claim` raises *"no slot available"*.

`VramSlots` cannot express a resurrection-off arm. **`ColdTier` can** —
`protected_rows == hot_rows` empties the reclaimable set and every path
reverts to the pre-Stage-3 tier. So the counterfactual is runnable on the
DRAM tier, at matched capacity, and this runs it.

Synthetic arena at OLMoE's geometry — 16×64, **3,342,336-byte rows**, 3.4 GB.
Page cache dropped between every arm, A/B/A alternation, 2–3 repeats, medians.
~2 TB read across the sweep.

## Turning resurrection off makes the tier faster

Same physical rows in both arms; the only difference is whether ownership is
capped below capacity.

| rows | protected | hard (off) | soft (on) | **Δ wall** | Δ reads | **residual** |
|---|---|---|---|---|---|---|
| 128 | 120 | 117,712 ms | 121,237 ms | **+3.0%** | +1.5% | +1.5% |
| 256 | 248 | 91,552 ms | 96,818 ms | **+5.8%** | +1.4% | +4.4% |
| 384 | 376 | 66,985 ms | 73,612 ms | **+9.9%** | +1.2% | +8.7% |

and with the budget halved, which raises the resurrection rate to R2's band
and past it:

| rows | protected | hard | soft | **Δ wall** | Δ reads | **residual** | resurrections |
|---|---|---|---|---|---|---|---|
| 128 | 64 | 117,447 ms | 120,096 ms | **+2.3%** | +1.4% | +0.8% | 6,991 (**10.7%** of routed) |
| 384 | 192 | 67,202 ms | 72,119 ms | **+7.3%** | +1.1% | +6.2% | 15,961 (**24.4%** of routed) |

**The arm performing 15,961 resurrections is 7.3% slower than the arm
performing none, at identical capacity on identical work.**

## The residual is the finding

Soft eviction reads ~1.2–1.5% more (the offline prediction in
[`RESULTS-r10.md`](../routing-trace/RESULTS-r10.md) said +1.1–1.4%, so that
part reproduces). But wall is **2–8× worse than the read gap**, leaving a
**+0.8% to +8.7% residual** that bytes do not explain, growing with pool size.

That shape names the cost: `_demote` walks the ACTIVE set on **every request**
to decide what to revoke, so the bookkeeping scales with rows, not with
resurrections. The mechanism is paid for per request and refunded per
resurrection, and on this workload the payments exceed the refunds.

This refines #152 rather than repeating it. #152 found resurrection-heavy
configurations slower and attributed it to capacity pressure rather than to
resurrection — correctly, since it had no way to hold capacity fixed. **Held
fixed, the cost does not go away.** It is not only that resurrection fails to
buy wall time; the machinery that makes it possible costs wall time.

## What this does not establish

- **Wrong tier.** R2 names **VRAM**; this is `ColdTier`, the DRAM tier. It is
  the only tier where the control is expressible, which is the reason for the
  substitution and does not make it equivalent.
- **One box, one trace, one arena**, uncontended by any other process.
- **The disagreement with R5 is unresolved.** R5 reports soft eviction
  *faster* under contention; this is a contended regime by R1's definition —
  a 1024-row working set into 128–384 rows — and soft is slower at every
  point. Both are on record; neither is reconciled here.
- **A resurrection is still an avoided read.** Nothing here says otherwise.
  What it says is that the per-request cost of keeping the option open is
  larger, at these pool sizes, than the reads it saves.
