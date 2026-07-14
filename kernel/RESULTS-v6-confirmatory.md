# v6 confirmatory — VERDICT: **CONFIRMED** (first fully-green kernel confirmatory). The register-LUT mainloop is ~1.5× the v5 M-tile loop everywhere, and gate_up prefill is no longer a loser class

**2026-07-14 · Protocol:** `kernel/prereg_v6_confirmatory.json` (OTS pre-data)
· **Frozen:** `6aeb718` · **Adjudicating host:** FRESH RunPod SECURE A5000
(post-stamp, fourth distinct instance of the arc), n=3 fresh-process reps ·
**Reducer:** `bench/phase1/reduce_confirmatory_v6.py` (bars read from the
stamped spec; smoke-validated on synthetic fixtures both sides of every bar
before the stamp).

## The change adjudicated

`kernel/nf4_grouped.py` M-tile mainloop: the NF4 codebook moves from a
per-element global-memory gather (`tl.load(lut_ptr + nib)`, [BN×BK] L1 LDGs
per K-step) into **registers** (`tl.gather` shuffle on a 16-float table),
with its rule (BLOCK_N=128, warps=4, stages=3; group-size-keyed BLOCK_M
unchanged). Fidelity-identical by construction — same codebook values, same
absmax scale, same TF32 dot, same fp32 accumulate. Decode paths byte-untouched.

## Verdict (all five registered criteria)

| criterion | bar | result | outcome |
|---|---|---|---|
| **W1 rewrite effect** (primary) | ≥1.25 all 8 cells, median ≥1.40 | **1.393–1.536, median 1.500** (worst single rep 1.385) | **PASS** |
| W2 dequant floor | ≥0.85 all 7; ≥1.05 on ≥5 | 1.142–2.782; **7/7 ≥1.05** | **PASS** |
| W3 gate_up | ≥0.85 all 3; ≥1.15 on ≥2 | Qwen 1.453 · gemma 1.160 · gpt-oss 1.257; **3/3 ≥1.15** | **PASS** |
| W4 decode guard | ≥1.0 on ≥7/8 | **8/8** (1.66–5.06) | **PASS** |
| Q1 suite | 44/44 | 44/44 (M-tile PFID gates exercise the v6 default) | **PASS** |

**V6_CONFIRMED = true** (`reduction_v6_a5000.json`).

- **R1 (report-only):** OLMoE gate_up dequant-ratio **0.604** — inside its
  0.55–0.62 exploratory band; the smallest-expert census shape remains the
  one prefill loser, now at 0.60× instead of the v5 loop's 0.38×.
- **R2:** the home-A2000 report-only leg runs tonight in the registered
  02:00–06:00 quiet window; its rows land as a separate stamped addendum
  (`RESULTS-v6-a2000-report.md`). No registered bar depends on it.

## What the numbers say

1. **The rewrite is worth ~1.5× at prefill, uniformly.** 1.39–1.54 across
   every census cell — gate_up and down, 1408≤N≤5760, 704≤K≤2880 — on the
   quantity that is instance-stable (both arms share the pod). The v1.1
   hypothesis (the per-element codebook gather was the M-tile bottleneck) is
   confirmed at confirmatory grade.
2. **gate_up prefill exits the loser class.** v1 shipped it at 0.22–0.85×,
   v4's config pass reached 0.4–1.0×; this instance reads 1.16–1.45× with
   all three big gate_ups above 1.15. Combined with the down-projections
   (1.14–2.78×), every census prefill cell except OLMoE gate_up is now a
   dequant-relative win on this instance.
3. **The dequant baseline's host lottery is real and disclosed** — the
   prereg's provenance section documents ~25% swings in the *baseline*
   across A5000 instances (the fused kernel holds within 0.2 ms). This
   instance drew a middle hand (Qwen gu 1.45 vs 1.62/1.21 on the two
   exploratory hosts). W1 was made the primary criterion precisely so the
   verdict would not ride that lottery.

## Disclosed protocol deviation (attempt 1)

The first post-stamp pod ran a **stale runner missing the `fused_v5loop`
backend** (a staging error: the corrected script was edited locally but not
re-copied to the launch host). W1 was unevaluable from that run; W2/W3/W4/Q1
all passed on it. Receipts kept (`conf6_attempt1_*`), the run was not used
for adjudication, and the corrected relaunch got a backends-echo guard in
the state file. Between the stamp and this verdict one commit (`dc86375`)
entered the local branch: it touches only `bench/phase3/` (a different arc's
prereg + offload harness) — the v6-adjudicated surface (`kernel/`,
`bench/phase1/`) is byte-identical from `6aeb718` through this verdict, and
both confirmatory pods ran the frozen `6aeb718` archive.

## Evidence / teardown

`bench/phase1/confirmatory_v6/`: `conf6_a5000_d{1,2,3}.json`,
`conf6_suite.log`, `conf6_a5000_STATE`, `reduction_v6_a5000.json`, attempt-1
receipts, exploratory basis at `bench/phase2/sweeps/v6e_matrix_a5000.json`.
Pods `zviqu1168szo30` (attempt 1) and `zhulc2yj7wngmg` (adjudicating) both
DELETE → 404-verified.
