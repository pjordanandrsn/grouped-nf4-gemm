# K30 — does the int4-b32 split-K R term win on a ≤64-SM card? The 48-cell sweep on one rented NVIDIA L4 (registered 2026-10-05, before any L4 run)

Owner instruction, in the maintainer session's chat on 2026-10-05: **"rent an L4 for the split-K R-term read."** It
followed the A2000-timing audit (grouped-nf4-gemm#475, `docs/audits/a2000-timing-2026-10-05.md` §2). Written down by
the agent, not by the owner.

## Why this lane exists

`int4_b32._plan` sizes split-K from the row count R and the SM count (`gnf4#357`, 0.31.0). It is gated to parts with
`sm_count <= SPLITK_R_TERM_MAX_SMS` (64) after P39's RTX 5090 step read (`gnf4#358`). The only receipt behind the term,
its constant `SPLITK_TARGET_BLOCKS_PER_SM = 8` and the gate is the 48-cell sweep `bench/int4/RESULTS-sk-r-sweep.md`. That
sweep was timed on the NAS RTX A2000, which is a correctness-only testbed under the testbed policy (standing since
2026-07-27). So the register row `gnf4.serve.int4-b32-splitk-row-term.a2000.2026-09-10` has no admissible speed evidence.
The same is true of the CI test `test_plan_choice_is_within_the_measured_bound_where_it_acts` (`SK_R_BOUND`). Even so,
every part with 64 SMs or fewer takes the term by default.

**The cheapest experiment that could falsify the premise comes first.** The premise is that the R-aware pick is faster
than the N-only pick on a ≤64-SM card at the kernel level.

- If it is not, it cannot be faster at the step level, and the term should be off.
- If it is, a step-level read becomes worth registering; that is not part of this lane.

This lane does not touch a model, a checkpoint or a download.

## Instrument (unchanged code, two passes)

- **The sweep.** `bench/int4/sk_sweep.py` at this registration's commit, unchanged:
  - shapes: qwen3_moe / granitemoe / olmoe, gate_up and down;
  - grid: R ∈ {1, 2, 4, 8, 16, 32, 64, 128} × every sk in {1, 2, 3, 4, 6, 8, 16} ∪ {the N-only pick}, up to the cap;
  - CUDA-graph replay, 60 replays after 12 warm-ups, full cost including `reduce_partials`;
  - one shape per process;
  - `E = 128` synthetic experts; timings are data-independent.
- **Two passes**, `OUT_PREFIX=k30p1` then `k30p2`, each covering all six shapes, run back to back on the same box. The
  second pass is the self-pair.
- **The reducer.** `bench/int4/k30_reduce.py` prices both plans from the sweep's own times:
  - the R-aware pick is `_plan(N, K, R, sm_count)` at this card's SM count;
  - the N-only pick is `_plan(N, K)`.

  On the box, `--check-installed-plan` asserts that the installed `int4_b32._plan` picks exactly those sks on all 48
  cells, so the reducer's restatement cannot drift from the code under test.

## Correctness first (no timing if any of these fails)

1. `kernel/test_int4_b32.py -k "gemv_matches_reference or reduce_partials or plan_"`, compiled on the card
   (`TRITON_INTERPRET=0`): the GEMV against its reference, `reduce_partials` against torch and the slot-order sum, and
   the plan's structural tests. The `SK_R_BOUND` receipt tests read the committed A2000 rows. They run because the
   filter selects them, and their result is not evidence of anything here.
2. `bench/int4/k30_sk_check.py`, at R = 16 and R = 128 on every shape:
   - every sk the sweep will time must match the sk = 1 output within split-K's fp32 reorder
     (max |Δ| ≤ 1e-2 · max |out|);
   - otherwise rc 22, and no sweep runs.
3. `k30_reduce.py --self-test`: 6 rule cases, plus the check that at 58 SMs the R term changes the pick on some cells (23 of
   48).

## Rule (in `k30_reduce.py`, fixed here)

The rule is read over the 24 cells at R ≥ 16, the only cells where the R term acts. Below the floor the two picks
coincide by construction, and the reducer asserts it.

- **Per cell, pooled over both passes:** r_c = (R-aware p1 + p2) / (N-only p1 + p2).
- **S** = summed R-aware time / summed N-only time over the 24 cells.
- **W** = the largest r_c.

**VOID** if any of these holds:
- the card is not `NVIDIA L4`, or has more than 64 SMs;
- a cell is missing, or a pick was not swept;
- the passes disagree on the card;
- the instrument is unstable: the two passes' own S differ by more than 0.02.

| verdict | condition | what follows (separate PRs, after the read) |
|---|---|---|
| **KEEP** | S ≤ 0.97 **and** W ≤ 1.02 | The L4 rows become the receipt behind the constant and the CI bound, replacing the A2000 rows. A kernel-level row on the L4 replaces the A2000 row, which is retired. Whether a step-level read on a ≤64-SM card is worth registering is decided then. |
| **OFF** | anything else | `SPLITK_R_TERM_MAX_SMS` is set so that no part takes the R term. Every card then plans N-only, as the 5090 already does. The A2000 row is retired with K30 as the reading that retired it. The `SK_R_BOUND` tests become structural. |

**Why 0.97 and 1.02.** The term buys its speed with a cross-card-class bargain: two cards plan different sk for the same
call, so outputs differ in the fp32 reorder class (the `_plan` CAVEAT). A default with that cost should earn a measurable
win, which I set at 3 % of the summed expert-GEMV time. It should also never cost more than 2 % on any cell, which is
the old "never slower" property with room for noise. These thresholds are judgement, written before the data. They are
not derived from any timing.

## Predictions (direction only; no band)

No band is registered. Only two timings exist for this term: the A2000 sweep, which is inadmissible as a basis, and P39's
5090 step read, which is outside the ≤64-SM class. Neither is a basis, and none is borrowed.

The direction comes from byte arithmetic on the shapes:

- **At R = 128**, the N-only pick for qwen3_moe gate_up (sk 16) writes and re-reads 16 × 128 × 1536 × 4 B ≈ 12.6 MB of
  fp32 partials twice. Its 128 rows read roughly 140–230 MB of expert weights, depending on how many of their experts
  repeat in L2. The R-aware pick (sk 1) moves none of the partials and skips the reduce launch.
  - 12 tiles × 128 rows = 1,536 programs is already ~26 per SM on 58 SMs, so dropping the split costs no occupancy.
- **At R = 16 and 32**, the R-aware pick keeps a smaller split (2–4). There the expected difference is small and could go
  either way.

**My expectation is KEEP.** S below 1 should be driven by the R ≥ 64 cells, and the per-cell worst case should sit near
1.0 at R = 16. That is a stated expectation, not a registered band. The rule above decides.

## What this lane does not claim

- **No step-level effect.** P39 showed a per-call kernel ratio is not a step share.
- **One card is not the class.** An RTX A4000 (48 SMs) or a 4060 Ti (34) may read differently.
- **The constant is not refitted.** This lane does not fit a better `SPLITK_TARGET_BLOCKS_PER_SM`, since one card cannot
  fit a constant for a class. The per-cell optimum is reported beside the verdict, as information.

## Rehearsal (correctness only: the A2000 is never a timing instrument)

The whole box script runs end to end in a throwaway `pytorch:2.8.0-cuda12.8` container on the NAS RTX A2000 before any
rental, with `K30_REHEARSAL=1`. That flag lifts the card check and shrinks the sweep to two R values per shape. The
rehearsal checks install, the tripwire, the three correctness gates, both passes and the reducer's VOID path; on an A2000
the card check makes the reducer say VOID, as it must. No A2000 time is quoted anywhere, and its rows are not committed as
receipts.

## Budget and launch

- **Provider and card.**
  - One NVIDIA L4 (58 SMs, sm_89, 24 GB) on `vast:verified-secure`.
  - Host-RAM band [32, 128] GB: the policy's 98 GB floor matches no L4 offer.
  - `preflight_bandwidth: none`: there is no Hugging Face checkpoint.
  - Declared ≤ $0.40/h; verified offers were $0.27–0.34/h on 2026-10-05.
- **Prerequisite.** `L4` must be in `adertha-agents` `adertha/compute/policy.json` `allowed_gpus`, with its RunPod
  mapping row. It is not yet, and the change is the owner's to make.
- **Guard and estimate.**
  - `R`, `SK` and `K` are `constexpr` in the GEMV, so the first pass compiles about 330 kernel variants. The
    second pass reads them from Triton's cache.
  - No runtime on this card exists to size a guard from. The A2000 rehearsal's durations are not used: sizing a guard
    from them is the P66 mistake. So the guard is generous instead: **1.5 h for the reading** (`k30-l4-<n>`).
  - Because that exceeds 1 h, the policy's **proving rental** comes first: `k30-prove-<n>`, `K30_PROVE=1`, the same
    provider class and image, a 0.25 h guard. It records the card check and forensics and exits, so it proves attach,
    pre-flight, handoff, receipt and teardown.
  - The reading's budget per pass is `K30_PASS_NEED_S` = 1,500 s; a pass that cannot fit is skipped `host-limited`.
  - Estimate: proving rental ≤ $0.10, reading ≤ $0.60 at the declared ceiling (expected well under half of it), plus
    the launcher's download allowance in its own estimate.
- **Stops and attempts.**
  - Hard stop $3 for the lane.
  - At most two attempts. A pre-flight VOID excludes its machine by receipt for the second attempt, and a second VOID
    closes the lane `host-limited`.
- **Where things live.**
  - Lane runs `k30-l4-<n>`, driven from experts4bit-qlora `bench/k30/` (K20's pattern; gnf4 is cloned on the box at
    `GNF4_SHA`).
  - Launched through `pod-launch.sh` from the session's own checkouts on the mini (`ADERTHA_REPO` / `E4B_REPO`), so the
    shared checkouts that live peer lanes pin are not moved.
  - Receipts: `kernel/receipts-k30/l4/`. Results: `kernel/RESULTS-k30-splitk-r-term-l4.md`.

## STOP rules

1. **No timing without correctness.** Any correctness gate failing ends the run with rc 21 or 22.
2. **No pass starts that cannot finish 10 minutes before the launcher's deadline.** A skipped pass is `host-limited`, and
   the lane reads VOID.
3. **No result is reported before its receipt exists** in the private record.
4. **Any teardown that cannot be proven stops the lane.**

Amendments, dated, go below this line before any data is read.

## Amendment 1 — 2026-10-05, after the A2000 rehearsal, before any L4 data

**What the rehearsal found.** Correctness gate 2 (`k30_sk_check.py`) failed on all 12 cells, with errors of order 1 and
NaNs. The cause is not the kernel; it is the instrument both scripts shared:

- With the fused reduce off (the shipped default), `_gemv_int4_b32` stores fp32 partials at **every** sk, including
  sk = 1.
- The served wrapper `gemv_int4_b32` therefore always calls `reduce_partials`, at sk = 1 as the fp32 → bf16 cast.
- `sk_sweep.py` called the reduce only for `sk > 1`. So its sk = 1 cell left the output unwritten (the NaNs), and it
  priced as free a launch plus an R × N fp32 read and bf16 write that the served path always pays.

That bias favours sk = 1, which is exactly the pick the R term makes at large R (23 of 48 cells change pick at 58 SMs,
most of them to sk = 1). The committed A2000 rows behind the term were taken with it. Two comments repeated the error:
`_plan`'s "sk collapses to no reduce", and `test_plan_stops_splitting_once_the_grid_is_full`'s "no reduce launch at
all". Both are corrected.

**The changes, all before any L4 data:**

- `sk_sweep.py` and `k30_sk_check.py` run `reduce_partials` at every sk, so every cell times the served two-launch path.
- The instrument is otherwise unchanged: same grid, same replay, same shapes, two passes.
- "Unchanged `sk_sweep.py`" in §Instrument now means the sweep at this amendment's commit.
- The rule, the thresholds, the gates and the budget are unchanged.

The rehearsal is re-run on this commit before any rental. Its outcome is recorded here as correctness only.
