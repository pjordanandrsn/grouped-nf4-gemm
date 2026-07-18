# WATCHLIST.md — competitive tripwires (R10)

**STATUS: DRAFT for operator review — UNCOMMITTED, private-fork only until
reviewed.** Plan-mandated deliverable (AGENT-PLAN-mxfp4.md §6, rule R10),
drafted 2026-07-17 at mxfp4-lane activation.

**Discipline:** checked at the start of every agent session on the
mxfp4/provenance/kernel lanes; any trigger = **stop work on the affected phase
and report before proceeding**. Per operator 2026-07-15 this list is
**recorded, NOT armed** — no autonomous session-start checks exist; the check
is manual (commands below, ~2 min).

| # | Trigger | Fires what | Status @ 2026-07-17 |
|---|---------|-----------|---------------------|
| 1 | Unsloth ships the MXFP4 **backward** pass (`W_TRANSPOSE` or equivalent) | Byte-provenance novelty gone; B1 headline dies; escalate for re-plan | **NOT FIRED** on bounded check — but see early-warning below |
| 2 | unsloth #4032 closes | bnb-4bit-MoE wedge collapses → pivot to provenance niche | **FIRED + RESOLVED**: CLOSED completed 2026-06-18 (verified via gh API 07-15/16); what shipped = dequant-path MoE (OLMoE/Qwen3-MoE/Gemma-4-text), NOT a fused kernel; pivot to B1 SELECTED — the flattened plan already reflects it |
| 3 | bitsandbytes **v0.50.0** releases | Audit contents for grouped/MoE 4-bit kernels; fires the PR #1965 "congrats + ready" comment per its Monday-gate rule | **NOT FIRED**: latest stable 0.49.2 (2026-02-16); only continuous main-wheel pre-releases since (checked `gh release list` 2026-07-17) |
| 4 | Marlin/vLLM or bnb adds **bnb-NF4 grouped-MoE GEMM** | Kernel format niche closes; host-streaming becomes sole differentiator; re-plan kernel positioning | **NOT FIRED** on release-scan; no full sweep today — last full check 2026-07-16 via #4032 close-context (dequant-path only) |

## Early-warning on trigger 1 (recorded 2026-07-17)

Unsloth commit 2026-06-22 (#6563): *"loader: gpt-oss MXFP4 default-4bit takes
the MXFP4 dtype path, not BnB."* Their **loader** now keeps native MXFP4 for
gpt-oss instead of converting to bnb — the load half of the provenance story.
Trigger 1 fires only on the **backward/training** half. Next check must
verify: does Unsloth's gpt-oss training path (a) train over native MXFP4
(trigger 1 FIRES), or (b) dequantize at forward / convert for training
(trigger holds)? Also re-derive their current 20b QLoRA VRAM number for
COMPETITIVE.md entry 1 — the 12.8–14 GB figure predates this loader change.

## Manual check pattern (run at session start on this lane)

```sh
# trigger 3 — bnb release state
gh release list -R bitsandbytes-foundation/bitsandbytes -L 3
# trigger 1 — unsloth MXFP4 movement (bounded; widen if signal)
gh api "search/commits?q=repo:unslothai/unsloth+mxfp4&sort=committer-date&order=desc" \
  --jq '.items[:5][] | {d:.commit.committer.date, m:(.commit.message|split("\n")[0])}'
gh api "search/commits?q=repo:unslothai/unsloth-zoo+mxfp4&sort=committer-date&order=desc" \
  --jq '.items[:5][] | {d:.commit.committer.date, m:(.commit.message|split("\n")[0])}'
# trigger 4 — vLLM/bnb grouped-NF4 MoE GEMM (keyword scan)
gh api "search/commits?q=repo:vllm-project/vllm+marlin+moe+nf4" --jq '.items[:3][] | .commit.message' 2>/dev/null
# trigger 2 — resolved; re-check only if unsloth re-opens #4032
gh issue view 4032 -R unslothai/unsloth --json state,stateReason
```

Log each check as a dated line below; a fired trigger stops the affected
phase and gets reported before any further lane work.

## Check log

- **2026-07-17** (agent, bounded): T1 not fired (early-warning #6563 recorded);
  T2 fired+resolved (prior); T3 not fired (0.49.2); T4 not fired on
  release-scan. mxfp4 lane proceeded (router-probe-gptoss + provenance
  reference table drafting).
- **2026-07-18** (agent, re-check at draft review): T1/T3/T4 unchanged
  (unsloth-zoo #913 still OPEN unmerged; #849/#850 no maintainer reply; bnb
  latest stable still 0.49.2). Lane activity since last check: e4b PR #24
  MERGED (`7e8dd12`) + SHIPPED as experts4bit-qlora 0.4.0 (trusted-published,
  attested); gpt-oss router-probe sealed public (RESULTS + stamps); PyPI
  kernel family live (grouped-nf4-gemm 0.1.1 + aliases).
