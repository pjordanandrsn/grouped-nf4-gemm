# PREREG — M2: re-certify the decode anchor

Registered 2026-08-26, before measurement. The anchor is not a
result; it is the INSTRUMENT that decides which boxes may produce
results, and a published ladder entry rests on it. Two defects
surfaced during K10 and are the reason for this cycle.

## Defect 1: the hunt harness gates on an uncertified constant

The repo's certified class is **7.35 ms**
(`RESULTS-k6b.md`: "Certified default: 7.35 ms class ≈ 136 tok/s";
`PREREG-f2-tail.md` / `RESULTS-f2-tail.md` use the same 7.35 ± 3%
window). The box-hunt harness has been gating on **7.39 ms** — a
number that appears in **no** RESULTS document. It entered the
scratchpad harness and was carried forward across cycles unchecked
([[harness-defaults-are-values]]).

Every anchor-gated cycle in this campaign was therefore screened
against a constant 0.5% away from the certified one. That did not
corrupt any verdict — each cycle's own G2 re-checks the knob-ON step
against 6.476 ms — but it did decide which boxes were rented.

## Defect 2: the constant is off-centre against the population

Six single-shot probes this session (`gen 64`, knob-OFF, b1 graph):
**7.23, 7.28, 7.26, 7.12, 7.12, 7.25** — mean 7.210, median 7.240,
min–max spread 2.2%.

| constant | ±3% window | probes inside | centre vs sample mean |
|---|---|---|---|
| 7.35 (certified) | [7.13, 7.57] | 4/6 | **+1.9%** |
| 7.39 (harness) | [7.17, 7.61] | 4/6 | **+2.5%** |

Both windows sit high enough that they admit boxes ~6% SLOWER than
the population while refusing ones ~1% faster. K10 lost two
provisioning cycles to exactly that: attempts 1 and 2 probed 7.12 ms
and were destroyed **for being fast**.

Note the direction. A health gate that preferentially rejects fast
boxes is not conservative — it biases the surviving population toward
the slow tail, which flatters every ratio measured on it.

## What this cycle measures

The current default decode step on the CURRENT main, properly:

- **N >= 3 anchor-compliant-agnostic boxes** — rented WITHOUT an
  anchor gate, since gating on the constant under test would beg the
  question. The AVOID list still applies (known-bad hosts).
- Each box: the knob-OFF b1 graph step, **A/A pair** (two full runs,
  not the single-shot probe the hunt uses), plus the NUMA/triad
  provision gate already in place.
- Recorded per box: both step medians, their spread, tokens
  identical, `recompiles_in_window == 0`, driver and torch versions.

## Decision rule (fixed before any box runs)

- **New anchor** = median of the per-box A/A medians.
- **New window** = ±3%, UNLESS the observed inter-box spread exceeds
  6%, in which case the window is `±(spread/2)` and the RESULTS must
  say the population is too dispersed for a 3% gate rather than
  quietly widening it.
- **REFUSE** if any box's A/A spread exceeds 2% (that box is not
  measuring itself consistently, so it cannot measure the class), or
  if fewer than 3 boxes complete.

## Obligations that follow (stated before the number is known)

1. The harness constant is corrected to the new value **and** the
   scratchpad hunt is changed to read it from a committed source, so
   a scratchpad number can never again gate a campaign cycle.
2. **If the new anchor differs from 7.35 by more than the measured
   A/A noise, the published ladder entry ("certified default 7.35 ms
   ≈ 136 tok/s") is CORRECTED in RESULTS-k6b and every downstream
   statement that quotes it.** This cycle is registered knowing it
   may invalidate a published figure; that is the point of running
   it.
3. If the new anchor is materially FASTER than 7.35, the tok/s
   figures derived from it move UP, and the RESULTS must state
   plainly that the campaign has been quoting a conservative default
   — not treat a favourable correction as a win.

## What this cycle does NOT do

It does not re-open any adjudicated verdict. K6-B, F2, SV1, SV2, K7,
K8 and K10 each carry their own same-box denominators and their own
G2 checks against 6.476 ms; none of them is a ratio against the
anchor constant. Only the ladder's absolute default entry and the
rental screen depend on it.

## Receipts

`kernel/receipts-m2/` — per-box A/A pairs, driver/torch versions,
the composed decision, and box_meta for each. `m2_verdict.py`
(self-tested) is committed BEFORE the box cycle.
