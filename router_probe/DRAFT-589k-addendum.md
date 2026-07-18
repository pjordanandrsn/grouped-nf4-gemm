# DRAFT — Family 2 addendum: A1 ceiling attempt at 589k (4× the A1 baseline)

> **DRAFT / addendum-hold.** Not yet appended to RESULTS.md, not stamped, not
> pushed to the public branch. Jordan reviews first. Receipt is banked at
> `receipts/20260718/EXPLORATORY_phase1_qwen3_moe_589k.json`.

**Motivation.** At 294k the reducer abstained ×3 (`plateau-without-overfit-gap`):
the ladder had flattened at ~0.82 but the train–held-out gap wouldn't certify
*model-limited*. Its printed remedy is "extend data or ladder." This run extends
**data** — 589,824 records (24 prompts × 512 tok; the 294k set-A plus a
12-prompt set-B captured via the incremental per-prompt slice banking), audited
on the same committed 7-rung × Δ{1,2,4} ladder. Capture on a DO L40S (resident
NF4), audit on a DO L40S; same `reduce/reduce_ceiling.py`, untouched.

**A1 at 589k** (`receipts/20260718/EXPLORATORY_phase1_qwen3_moe_589k.json`):

| Δ | linear | MLP-d | MLP-4d | attn2 | attn4 | attn4_w512 | attn6_w512 | ceiling | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.497 | 0.493 | 0.551 | 0.789 | 0.818 | 0.840 | **0.845** | 0.845 | plateau-no-gap |
| 2 | 0.492 | 0.489 | 0.546 | 0.784 | 0.812 | 0.832 | **0.837** | 0.837 | probe-limited |
| 4 | 0.475 | 0.449 | 0.592 | 0.777 | 0.800 | 0.820 | **0.827** | 0.827 | probe-limited |

Verdicts verbatim from the committed reducer:

```
{"family": "qwen3_moe", "band": "all_layers", "delta": 1, "heldout_by_rung": [0.497, 0.4927, 0.5512, 0.789, 0.8177, 0.8404, 0.845], "verdict": ["plateau-without-overfit-gap (no verdict; extend data or ladder)"]}
{"family": "qwen3_moe", "band": "all_layers", "delta": 2, "heldout_by_rung": [0.4918, 0.4894, 0.5457, 0.784, 0.812, 0.8318, 0.8372], "verdict": ["probe-limited (ceiling not established)"]}
{"family": "qwen3_moe", "band": "all_layers", "delta": 4, "heldout_by_rung": [0.4752, 0.449, 0.5916, 0.7766, 0.8004, 0.8201, 0.8274], "verdict": ["probe-limited (ceiling not established)"]}
```

**Reading (CHARTER §7).** The second data-doubling moved the verdicts *again* —
and not toward closure. The plateau rose a third time (~0.80 → ~0.826 → 0.845
at Δ1), and at Δ2/Δ4 the extra data **re-opened the top of the ladder**: the
attn4_w512 → attn6_w512 held-out gains grew from <0.005 at 294k to
+0.005/+0.007, so the reducer now reads a *rising* top segment and returns
`probe-limited` — the biggest-capacity rungs are the ones that benefit most
from more data, re-steepening exactly the segment that had flattened. Train-side
H at the top rungs sits at 0.91–0.95 with held-out at 0.82–0.845, so the
generalization gap persists at 4× the original A1 volume.

Three verdict flips across three volumes (147k probe/model-limited mix → 294k
abstain ×3 → 589k abstain + probe-limited ×2) is the result: **the Qwen3-30B
ceiling is not pinnable by data scaling at this ladder** — each doubling buys
~+0.02 of plateau and a different verdict shape. Contrast OLMoE, where every
volume and every Δ returned the same clean model-limited 0.91. The
family-dependence headline strengthens: for OLMoE, H is a stable, cheaply
measurable model property; for Qwen, the measurement itself chases a moving
plateau.

**Decision corollary (unchanged, now at 3 volumes).** Every observed or
extrapolated Qwen plateau (0.80, 0.826, 0.845, trend ≈ +0.02/doubling) sits far
below the ≈0.95 wire-law break-even for speculative expert streaming — the
prefetch-negative conclusion does not depend on pinning the exact ceiling.

**Cost/ops note.** First 589k audit attempt on the home A2000 OOM'd (7.57 GB
needed, ~4.7 free under co-tenant load) — the audit ran on a DO L40S instead;
first collector window (55 min) was shorter than the Δ4 leg and the audit was
re-run with a 135-min ceiling for the complete 3-Δ receipt. Big-ladder audits
need a 48 GB card or a true quiet window, and collectors must outlive the
longest leg.
