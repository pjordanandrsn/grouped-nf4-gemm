# Gate-2 assessment — decision memo (owner call)

**2026-07-13.** Gate 2 (`gemm_predictions.json`): *"thresholds met at bs1 decode
on sm_86 OR a roofline-backed narrowing recorded; passing flips the repo public
with receipts and unblocks the one #1949 coordination comment."*

## Verdict: the registered criterion is MET.

- **Speed:** 8/8 census decode-bs1 cells ≥ 1.3× vs the dequant path (1.32–3.46×).
- **Energy:** 8/8 fused J/token strictly below dequant.
- **Fidelity:** property suite 35/35; P-fid holds (fused 0.61–0.79× the dequant
  path's fp64 error — measurably *more* accurate).

Receipts + full honest accounting: `RESULTS-phase2-gate2.md`.

## What a reviewer will poke at (so you decide with eyes open)

1. **The decode config is census-tuned by exact (N,K).** It's legitimate for the
   census benchmark the thresholds are defined on, but it is not a general
   heuristic — off-census shapes fall to the 128/4 default and are unmeasured.
   A skeptic could call the 8/8 "fit to the test set." My read: defensible for a
   *v1 public* kernel with an honest scope note, but a cost model (or a
   correctly-selecting autotune) is the thing that turns "8/8 on the census" into
   "fast on your shape." This is the single most likely critique.
2. **The op-boundary measurement fix was the larger mover** of the two prior
   misses. It's a real fairness correction (no baseline paid the cat+eids tax the
   fused op was charged), and I reported both numbers — but it means "we went
   6/8 → 8/8" is partly a measurement fix, not all kernel speedup. Stated plainly
   in the results; anyone reading the diff sees it.
3. **Prefill is not at parity** (0.22–0.85×). The registered claim there was
   parity+energy not speedup, and it's a compute-bound v1.2 job — but a public
   repo invites "why is prefill slow," so the README must front the decode-only
   scope.

## The two triggers — both yours

**(A) Flip the repo public.** Options:
- *Now, with an honest scope note* — decode-bs1 is the validated win; README
  states census-tuned config + prefill-pending explicitly. Ships the real result,
  invites the cost-model critique but answers it up front. Reasonable.
- *Hold for the cost model + prefill parity* — presents a more complete kernel,
  fewer easy critiques, at the cost of weeks. Also reasonable.
- My lean: **a v1 public flip is defensible now** IF the README leads with scope
  (decode-bs1, sm_86, census-tuned) — the energy result alone (deleting the
  METHODOLOGY §10 1.2–2.3× NF4 penalty, measured, at the point of use) is a
  clean publishable claim. But no strong objection to holding; it's a
  presentation/timing call, not a correctness one. **Your decision.**

**(B) The #1949 coordination comment.** Draft prepared at
`~/e4b-outbox/gnf4-1949-coordination-comment-DRAFT.md` — a plain
building-on-your-work heads-up to matthewdouglas, framed as the expert-grouped
extension of the #1949 family, offering to coordinate. It reveals nothing
commercial (the FusedStore/placement layer stays private; only the kernel is
upstream-relevant). **Not posted.** Open question for you: whether to add an
AI-assistance line — the new attribution default covers commits/PRs; a
conversational issue comment isn't clearly in scope, and I left it clean.
Answer honestly if he asks, per the white-hat standard.

**Recommended order if you greenlight both:** post the coordination comment
first (it's low-stakes and gauges his appetite), *then* flip public — so the
public repo lands into a conversation you've opened, not cold.

## What I did NOT do

No repo made public, no comment posted, no README rewrite for public scope — all
of that waits on your word. Prefill parity and the decode cost model are the
queued engineering, independent of these decisions.
