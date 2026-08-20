# R9, scored — the choice it imagines is dominated, so there is nothing to choose

Registered (`PREREG-tribrid-stage3`, R9):

> choosing between simultaneously-valid DRAM and VRAM copies by slack beats
> always taking the highest tier — **refuted if highest-tier-always ties or
> wins**.

**REFUTED.** Highest-tier-always wins at every point where the two policies
differ at all, and ties where they cannot differ. Declining is worse at
**6** of 8 configurations, ties at **2**, better at
**0**.

gnf4#150 scored the *precondition* — two valid copies exist for 0–57.3% of
invocations, entirely as a function of device-cache capacity. This scores the
choice itself. Receipt `r9_policy.json`, scorer `score_r9_policy.py`. No GPU.

## It needed no deadline estimator after all

R9 was parked because "by slack" implies a time-to-deadline estimate and gate
2 established there is none. Two things removed that dependency.

**The currency is transfers, not seconds.** gnf4#152 measured wall on this
path against row transfers at **r = +0.975**, slope **546 µs per 13.22 MB
row** — 24.2 GB/s against a ~28 GB/s PCIe ceiling. A policy that moves fewer
rows is faster, with a measured conversion factor, so a transfer count *is* a
wall comparison here.

**The state machine makes it a dominance argument.** In
`VramSlots._want_locked`, a resurrection promotes `RECLAIMABLE → ACTIVE` on
the slot that **already holds that expert**: no transfer, nothing else
evicted. Declining it to take the DRAM copy still has to `_claim` a slot —
consuming the same capacity, quite possibly that very slot — and pays a
transfer on top. **Same slot pressure, strictly more traffic.**

So the quantity a slack policy would trade against does not exist here.
Taking the higher tier costs no capacity the alternative would have saved.

## Measured, because an argument is not a result

| rows | prot | keep (transfers) | decline | Δ | declined | est. cost (ms) |
|---|---|---|---|---|---|---|
| 256 | 128 | 43338 | 43338 | **+0** | 0 | +0.0 |
| 256 | 248 | 36868 | 37016 | **+148** | 87 | +80.8 |
| 384 | 192 | 42267 | 42456 | **+189** | 138 | +103.2 |
| 384 | 376 | 26468 | 26742 | **+274** | 174 | +149.6 |
| 512 | 256 | 36293 | 36579 | **+286** | 136 | +156.1 |
| 512 | 504 | 17438 | 17850 | **+412** | 205 | +224.9 |
| 1024 | 512 | 16643 | 17145 | **+502** | 198 | +274.0 |
| 1024 | 1016 | 989 | 989 | **+0** | 0 | +0.0 |

The two ties are structural rather than close calls: `declined = 0` at both,
so no reclaimable copy was ever available to refuse and the policies are
identical by construction. Everywhere a choice existed, taking it cost
between **148** and
**502** extra row transfers —
**81
to 274 ms** over
the trace at the measured per-row cost.

## Why this covers every slack policy, not just this one

The policy measured here declines **every** reclaimable copy — the extreme
end of R9's spectrum. A real slack policy would decline some subset.

That is covered, because declining is dominated **per request**: each
individual decline adds a transfer and saves no capacity. Any policy that
declines a subset therefore sits between "never decline" (highest-tier-always,
the best case) and "always decline" (the worst case measured above), and
cannot beat the former. There is no subset whose members are individually
free.

Refuting R9 does not require finding the best slack signal, because no slack
signal can pay for a strictly dominated action.

## What this does not establish

- **It is a property of this engine's state machine**, not of tiered memory
  in general. An arrangement where consuming the VRAM copy *did* cost
  capacity the DRAM path would not — a pinned-on-read design, say — would
  restore the trade R9 describes. `VramSlots` is not such a design.
- One trace, one geometry, 8 configurations, `protected` at half and
  `rows − k` only.
- The estimated milliseconds convert transfers at gnf4#152's 546 µs/row,
  measured on **gpt-oss-20b** rows on one box. The ordering does not depend
  on that constant; only the magnitudes do.
