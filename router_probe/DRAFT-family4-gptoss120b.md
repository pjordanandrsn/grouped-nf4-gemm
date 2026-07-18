# DRAFT — Family 4: gpt-oss-120b (E=128, k=4, L=36) at two data volumes

> **DRAFT / addendum-hold.** Receipts banked at
> `receipts/20260718/EXPLORATORY_phase1_gpt_oss_120b.json` (110,592 records,
> 256 tok × 12 prompts) and `…_512tok.json` (221,184 records, 512 tok).
> Not appended to RESULTS.md, not stamped, until reviewed.

Same architecture as Family 3 (gpt-oss-20b, E=32) at E=128 — the E-axis test
at fixed k=4. Resident NF4 on H200s (DO), experts4bit streaming loader,
`gpt_oss` adapter (config-driven E/L).

| volume | Δ | linear | MLP-d | MLP-4d | attn2 | attn4 | attn4_w512 | attn6_w512 | ceiling | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 110k | 1 | 0.300 | 0.230 | 0.225 | 0.728 | 0.744 | 0.739 | 0.745 | 0.745 | probe-limited |
| 110k | 2 | 0.299 | 0.225 | 0.230 | 0.718 | 0.733 | 0.730 | 0.740 | 0.740 | probe-limited |
| 110k | 4 | 0.293 | 0.223 | 0.219 | 0.706 | 0.720 | 0.722 | 0.722 | 0.722 | model-limited |
| 221k | 1 | 0.355 | 0.267 | 0.256 | 0.761 | 0.776 | 0.782 | 0.787 | **0.787** | plateau-no-gap |
| 221k | 2 | 0.354 | 0.247 | 0.238 | 0.754 | 0.770 | 0.776 | 0.779 | 0.779 | plateau-no-gap |
| 221k | 4 | 0.343 | 0.246 | 0.237 | 0.739 | 0.755 | 0.762 | 0.763 | 0.763 | plateau-no-gap |

Verdicts from the committed reducer (`reduce/reduce_ceiling.py`), verbatim
class names in the table; train-side H at the top rungs is 0.97–0.99 at both
volumes, so the generalization gap never closes.

**Reading (CHARTER §7).** Doubling the data moved every verdict and lifted the
plateau 0.745 → 0.787 — gpt-oss-120b is **data-starved, tracking Qwen3-30B's
arc one doubling behind** (mixed probe/model-limited → abstain-with-rising-
plateau). The honest statement of its H is therefore **≥ 0.787 and unpinned**,
not the 110k run's 0.745.

**E-axis reading, revised (5 families, honest form).** The earlier framing
"E=32→E=128 drops H 0.83→0.745 at k=4" overstated what the receipts support —
0.745 was a data-starved lower bound. What the five families actually show:

- **Low-E families pin cleanly at first volume** (model-limited ×3):
  gpt-oss-20b E=32 → 0.83; granite E=40 → 0.90; OLMoE E=64 → 0.91.
- **Both E=128 families are data-unpinnable** at every volume tried:
  Qwen3-30B (k=8) — three doublings to 589k, plateau 0.80→0.845, never
  certifies; gpt-oss-120b (k=4) — two volumes, plateau 0.722→0.787, never
  certifies.

So expert count doesn't merely lower H — **at E=128 it makes H unmeasurable by
data scaling on this ladder**, at both k=4 and k=8. The k=4/E=32 point (0.83,
pinned) still sits below every pinned k=8 family (0.90–0.91), so a k effect
survives; the E=128 magnitude claim should be stated as bounds
(≥0.787 / ≥0.845), not point values.

**Curiosity on record:** at both volumes the MLP rungs (0.22–0.27) sit far
*below* the linear rung (0.29–0.36) — unique to this family; the attention
rungs carry the entire ladder. Not investigated further here.

**Ops note.** The 512-tok lane took four droplets to land — none of the
failures were science: a stale bundle predating `--receipt-suffix` (the
incremental-capture upgrade had never been committed; recovered into git,
`c5c9b42`), a `snapshot_download` wall on the repo's consolidated
`original/*` file (fixed with `ignore_patterns` + a download gate), and two
watcher-destroyed SSH-dark-during-download droplets before the API-check law
landed. Capture itself: ~45 min on an H200, audit on-box.
