# R3 — UNDETERMINED as registered: a parameter it does not pin decides it

Receipt: [`r3.json`](r3.json). Harness: [`score_r3.py`](score_r3.py). Trace:
[`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl). No box.

## As registered

> **R3** — DRAM resurrection rate (≈10–30%) exceeds VRAM (≈3–15%).
> **Refuted if VRAM ≥ DRAM.** (`PREREG-tribrid-stage3.md`)

The two sides had never been measured against each other. R1's DRAM figures
came from a model on a rented box; the VRAM figures came from a fixture —
different traces, different pressure. This matches them: **one** captured
routing sequence through **both** state machines at the **same** capacity and
the **same** protected budget. `ColdTier` is the DRAM side, `VramSlots` (via
`DevRowCache`) the VRAM side, and both publish the same quantity —
resurrections over resolved logical evictions.

## The verdict inverts with a parameter R3 does not specify

| rows | protected | DRAM rate | DRAM refills | VRAM rate | VRAM refills | verdict |
|---|---|---|---|---|---|---|
| 128 | 64 (half) | 13.5% | 44,939 | 0.0% | 65,536 | R3 holds |
| 192 | 96 (half) | 19.3% | 38,607 | 0.0% | 65,536 | R3 holds |
| 256 | 128 (half) | 26.6% | 32,570 | 0.0% | 43,338 | R3 holds |
| 384 | 192 (half) | 42.3% | 22,129 | 1.1% | 42,267 | R3 holds |
| 512 | 256 (half) | 58.3% | 13,706 | 1.6% | 36,293 | R3 holds |
| 128 | 120 (rows−k) | 0.4% | 44,965 | **33.9%** | 43,338 | **REFUTED** |
| 192 | 184 (rows−k) | 0.6% | 38,619 | 0.6% | 42,719 | **REFUTED** |
| 256 | 248 (rows−k) | 0.8% | 32,605 | 0.7% | 36,868 | R3 holds |
| 384 | 376 (rows−k) | 1.4% | 22,151 | **2.1%** | 26,468 | **REFUTED** |
| 512 | 504 (rows−k) | 2.8% | 13,720 | **4.2%** | 17,438 | **REFUTED** |

At `protected = rows/2`: **R3 holds 5 of 5.**
At `protected = rows − k`: **R3 refuted 4 of 5.**

Same trace, same capacities, same two implementations. Only the protected
budget differs, and the answer reverses. **R3 does not name a budget**, so as
written it does not have an answer — the number it asks about is not a
property of the tiers.

The magnitude of the sensitivity is the point. At 128 rows the VRAM
resurrection rate moves from **0.0% to 33.9%** on that one setting, and the
DRAM rate moves from 13.5% to 0.4% on the same change in the opposite
direction. Neither is measuring "how much reuse does this tier see".

## Why the rate is a poor payoff metric, stated separately

Resurrection rate is used across R1–R3 as if higher were better. It does not
reliably track cost. Physical refills are printed beside every rate above for
exactly this reason, and they do not move with it:

- At 128 rows the VRAM rate goes 0.0% → **33.9%** while refills *improve*,
  65,536 → 43,338. Rate up, cost down.
- Between victim rules at 256 rows / protected 128, the slot-index policy
  showed **266** resurrections and **54,819** refills; least-recently-used
  showed **0** resurrections and **43,338** refills. Rate down, cost down.

So the rate rises with quality in one comparison and falls with it in
another. A resurrection is a *capacity-relative bookkeeping event*, not a
saving: it counts rows that were demoted and then needed again, which a cache
that demoted better would never have demoted. **Report physical refills; the
resurrection rate on its own cannot carry a claim.**

This does not disturb R1. R1's operational half was measured in *reads*
(−13.5% / −24.5%), not in resurrection rate.

## What would make R3 answerable

Pin the budget, and say which one the deployed system uses. On the evidence
in `RESULTS-dev-cache-real-routing.md` the right VRAM budget is `rows − k` —
one request of demotable margin — and at that budget **VRAM ≥ DRAM at 4 of 5
capacities**, which is R3's own refutation condition. If the prereg is
amended to pin `rows − k`, R3 should be recorded REFUTED. It is left
UNDETERMINED here because amending a registered prediction to match a result
is the thing preregistration exists to prevent.

## Limits

- One model, one prompt, 512 decode steps.
- The DRAM side runs against a **synthetic** 16×64 arena of 224-byte rows.
  Bytes are irrelevant to a residency question — the geometry and the request
  stream are what matter — but no I/O timing is claimed from it.
- `ColdTier` and `VramSlots` are different implementations with different
  victim rules (LFU+LRU vs LRU within preference classes). At matched budget
  the comparison is as close as two separate state machines get, but it is
  not the same code on both sides.
