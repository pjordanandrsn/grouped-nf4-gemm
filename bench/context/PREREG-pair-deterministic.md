# PREREG — the pair on the 235B, with the deterministic scatter

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
Code: gnf4 @ `0248331`(docs), e4b @ `0248331`. Both local, unpushed.

## What changed since the last 235B run

The pair measured **7.88×** on the 235B (#23), but three things have since been
found and fixed:

1. **The grouped kernel was nondeterministic** — atomic `index_add_` in the
   weighted combine, measured as a 9.01e-04 perplexity spread on OLMoE against a
   bit-stable reference (#26). Replaced with a scatter-by-assignment plus a
   fixed-axis sum.
2. **Greedy ids on random prompts were the wrong gate** (#24) — near-uniform
   logits made argmax amplify any perturbation to total disagreement. Gates are
   now on **logits**, on a **natural** prompt.
3. `enable_fast` is now **priced**: +0.0229% perplexity for its 1.32× on 16
   layers (#26).

So the 7.88× was measured with a nondeterministic kernel and judged by an
instrument that could not distinguish signal from chaos. This re-runs it with
both corrected, and asks the question #23 left open using the right instrument.

## Fixture

Qwen3-235B-A22B, 2×A100-SXM-80GB, NF4 experts pinned, KV NF4 host-resident,
`prefetch=False`, **natural prompt**, greedy, 12 new tokens, median of 2, one
process. Cells: `bulk+ref`, `bulk+grouped`, `routed+ref`, `routed+grouped`,
plus `routed+grouped` repeated for determinism and at ctx 32768.

## Predictions

- **R2a — the fix is free.** `routed+grouped` s/token ≤ **1.05×** its atomic-era
  0.706 s, so the pair stays ≥ 7.0× end to end. The new buffer is
  `[tokens*k, hidden]` fp32 = 128 KB at decode; a fixed-axis sum replaces an
  atomic accumulation. *Falsified above 1.15×.*
- **R2b — GATE, and the answer to #23's open question.**
  `max|Δlogit|` between `bulk+grouped` and `routed+grouped` is **exactly 0**.
  Routed staging is bit-identical and the kernel is now deterministic, so the
  two must agree to the last bit. This is what #23 could not measure and #24
  could not interpret. *Falsified by any nonzero difference.*
- **R2c — determinism at depth.** `routed+grouped` repeated gives **identical**
  logits. The fix was validated at 16 layers; 94 layers is 6× the opportunity for
  an atomic to reorder. *Falsified by any difference.*
- **R2d — compounding, reported not predicted.** The reference↔grouped logit
  `rel` at 94 layers, beside OLMoE's **12.9%** at 16. If per-layer error is
  roughly constant this should be substantially larger, which would put a
  concrete depth-scaling caveat on the +0.023% perplexity figure. Recorded as a
  measurement; no interval, because I have no basis for one.

## Pre-committed decisions

- **R2b confirmed** → #23's divergence is closed as an artifact of the greedy-id
  instrument, and the pair is documented as bit-identical to bulk under the same
  kernel. #24's method change is vindicated rather than merely asserted.
- **R2b falsified** → there IS a staging-dependent difference under the kernel
  that four probes missed, and routed+grouped is withdrawn as a recommended
  configuration until it is explained.
- **R2a falsified** → the determinism fix has a real cost at flagship scale and
  the trade (determinism vs speed) gets stated explicitly rather than assumed
  free.
- **R2d large** (say rel > 0.5) → the +0.023% perplexity number carries an
  explicit "measured at 16 layers, unmeasured at 94" caveat everywhere it appears.

## Confounds

1. One box, one prompt, 12 new tokens, median of 2 — the same shape as #23 so the
   comparison is like-for-like.
2. Perplexity is NOT measured here (it needs a corpus pass on a 128 GB model);
   R2d's logit rel is a proxy for the depth story, not a substitute.

## Outcome — INCOMPLETE. The pod was swept by a concurrent session before scoring.

| cell | s/token | tok/s | peak |
|---|---:|---:|---:|
| `bulk+ref` | 5.601 | 0.179 | 18.58 GiB |
| `bulk+grouped` | 5.415 | 0.185 | 18.59 GiB |
| `routed+ref` | 1.026 | 0.975 | 18.58 GiB |
| **`routed+grouped`** | **0.837** | **1.195** | 18.59 GiB |
| `routed+grouped` (repeat) | 0.722 | 1.385 | 18.59 GiB |

End-to-end **6.69×** (first) / **7.76×** (repeat) — consistent with #23's 7.88×,
so **the determinism fix did not cost throughput**. R2a is ambiguous by its own
interval (1.186× on the first grouped run, 1.023× on the repeat): the first run
of a newly-patched path pays warm-up the repeat does not, and the prereg did not
anticipate that. Reported as a spread, not scored.

**R2b, R2c and R2d were never computed.** They are calculated after all arms, and
the pod was deleted before the 32768 arm. **The gate — the definitive answer to
#23's open question — is still unmeasured.**

**Cause, and it is not the provider.** Two other Claude Code sessions were running
on the same Mac (`k3-day0`, `moe-sec-train`), sharing one RunPod account and the
same `backstop.sh`. Every pod this session lost died with **no backstop-log entry
at its death time** — the signature of an external delete. And this session did the
same thing in the other direction earlier, deleting a pod named
`e4b-composite-repro` that it had not created because it was "still billing".
That was another session's live run.

So three lost 235B runs (~$10) and one interrupted foreign run are all one cause:
**cross-session pod sweeping on a shared account.** Recorded as
`feedback_concurrent_session_pod_sweeping`. Not a 45-minute reaper, not provider
flakiness — both of which the evidence superficially fit.
