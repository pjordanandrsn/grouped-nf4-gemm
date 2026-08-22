# Preregistration — do the published results survive re-captured traces?

Registered before any re-capture. No trace exists for the manipulation below.

## The problem

[`RESULTS-topk-frequency.md`](RESULTS-topk-frequency.md) found that the
committed OLMoE trace does not reproduce. Same model id, same prompt, same
greedy decode, captured under transformers 5.15.1:

| comparison | agreement |
|---|---|
| module-emitted vs derived, one environment | **8192 / 8192 = 100.00%** |
| **committed vs a fresh capture** | **18.07%** |

The harness is exact; the environment moved. `olmoe_*.jsonl` is the trace
behind **R4**, **R10**, the **crossover threshold** and the
**policy-headroom** numbers, and nothing in the repo recorded which
transformers produced it.

Those results are not thereby wrong. Every comparison inside them was made
between traces captured together, so they are internally consistent. The
question this registers is narrower and answerable: **do they still hold on
traces captured today?**

## Scope

Re-capture **OLMoE**, **Granite** and **Qwen1.5-MoE**, four prompts each, 512
steps, one environment, with `env` provenance recorded (added alongside this
file). gpt-oss is already current — it was captured on 2026-08-21 and again on
2026-08-22, and is the control: if *it* moves, the re-capture itself is
suspect and nothing else is scored.

## Registered outcomes, with tolerances fixed now

Each is scored on fresh traces and compared to the published number.

**T1 — R10's refutation.** Published: 153 REFUTED of 160 cells, the 7
exceptions all at rows=512 and all under 1%.
* **Reproduced** if ≥ 150 of 160 cells are REFUTED and every exception remains
  under 1% on both reads and churn.
* **Broken** otherwise, and `RESULTS-verdict-audit.md` gets a banner.

**T2 — the one-step crossover.** Published: ratio > 1 below `layers × top-k`
and < 1 at or above, 12 traces of 12.
* **Reproduced** if all 12 fresh traces show the same sign pattern.
* This one is expected to hold whatever the routing does —
  `structural_check.py` shows it moves in 0 of 24 synthetic conditions — so
  reproducing it is weak evidence and is registered as a **control**, not a
  result. If it breaks, the re-capture is wrong.

**T3 — the per-model LFU ÷ LRU ratios.** Published: OLMoE 0.761, Granite
0.746, Qwen 0.953. This is the number the whole frequency/recency thread rests
on and the one most able to move.
* **Reproduced** if each model's median is within **0.05** of its published
  value.
* **Moved** otherwise. A move is a real result: it would mean the
  frequency/recency split, and the five explanations eliminated against it,
  were fitted to traces from an environment that no longer exists.

**T4 — R4's refutation.** Published: frequency wins in every scored cell.
* **Reproduced** if frequency still wins in ≥ 80% of cells.

## What would count as a miss

* gpt-oss (the control) moving on any of T1–T4 ⇒ the re-capture is at fault,
  nothing else is scored, reported as a failed capture.
* T3 moving is the outcome with real consequences and will be reported first
  and plainly, not buried under whichever of T1/T2/T4 happened to hold.
* Partial reproduction is reported per-test. "Mostly reproduced" is not a
  verdict.
* No document is edited on the strength of a single re-capture: a moved result
  gets a banner recording both numbers and the environment each came from,
  not a silent overwrite.
* The box is destroyed when the run ends and receipts are committed before it.

## Deliberately not in scope

Deciding which environment is "correct". A trace is a measurement of a model
under a library, both of which change; the repository's job is to record which
one it used, which is what the `env` field now does. Re-capturing does not
make today's environment canonical, and the results will not claim it is.
