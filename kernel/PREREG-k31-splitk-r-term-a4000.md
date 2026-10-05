# K31 — does the int4-b32 split-K R term win on a second ≤64-SM card? K30's 48-cell sweep on one rented RTX A4000 (registered 2026-10-05, before any A4000 run)

Lane number claimed by `prereg/k31` (pushed 2026-10-05T12:27Z). Work item: grouped-nf4-gemm#479.

**Owner instruction**, 2026-10-05, in the chat of the session applying grouped-nf4-gemm#475's decisions: **"register and
run the A4000 card yourself"**. It came after two answers in that chat (relayed on #475 issuecomment-5994344079):
- the read is a kernel sweep;
- the classes are L4 and RTX A4000, two cards, "KEEP needs both".

Relayed as the rentals' authorization on #479 issuecomment-5994455012. Written down by the agent, not by the owner.

## Why this lane exists

**K30 is the first card.** K30 (`prereg/k30`, the maintainer session's registration) reads the split-K R term of
`int4_b32._plan` on one rented NVIDIA L4. It is the only admissible reading the term can have: its old receipt was an
A2000 timing, now retired (grouped-nf4-gemm#476). K30 states its own limit: **"One card is not the class. An RTX A4000
(48 SMs) or a 4060 Ti (34) may read differently."**

**K31 is the second.** The term is gated to the whole ≤64-SM class (`SPLITK_R_TERM_MAX_SMS = 64`). K31 reads the same
instrument and rule on an RTX A4000:

| | L4 (K30) | RTX A4000 (K31) |
|---|---|---|
| SMs | 58 | 48 |
| architecture | Ada, sm_89 | Ampere, sm_86 |
| memory | 24 GB GDDR6, ~300 GB/s | 16 GB GDDR6, ~448 GB/s |

The two cards differ in architecture, SM count and bandwidth. That is the two-card rule for performance portability.

**Where the term acts.** This is a code reading confirmed by a receipt, not a timing:
- Since P88 made K19 the default (e4b, 2026-10-01), e4b's int4 server sends every device-grouped decode row at B > 1 to
  K19's small-M GEMM.
- The split-K GEMV whose plan the R term sizes is reached only by:
  - `E4B_INT4_GROUPED_SMALLM=0`;
  - the singleton GEMV at T > 1 (`FORCE_SINGLETON_GROUPS`);
  - direct callers of `gemv_int4_b32`.
- P112's RTX 5090 build launched `_gemv_int4_b32` 576 times. That is 48 layers × 4 projections × 3 B=1 executions, none
  from buckets 2–16 (experts4bit-qlora `bench/p112/receipts/p112-5090-2/arm_P0a.json`).

So the kernel level is where the term acts, and a step-level read of today's default server would compare identical
kernels.

## Instrument: K30's, unchanged

Everything K30 registers under "Instrument" and in its Amendment 1, at `prereg/k30` **`00da974`**, merged into this
branch without edits:
- **The sweep.** `bench/int4/sk_sweep.py` as amended there, so `reduce_partials` runs at every sk, sk = 1 included.
  - shapes: qwen3_moe / granitemoe / olmoe, gate_up and down;
  - R ∈ {1, 2, 4, 8, 16, 32, 64, 128};
  - every sk in {1, 2, 3, 4, 6, 8, 16} ∪ {the N-only pick}, up to the cap;
  - CUDA-graph replay, 60 replays after 12 warm-ups, one shape per process, E = 128 synthetic experts.
- **Two passes**, `OUT_PREFIX=k31p1` then `k31p2`, back to back on the same box.
- **The reducer.** `bench/int4/k31_reduce.py` imports `k30_reduce.py` from the same commit and calls it as it stands.
  It substitutes only the registered card (`NVIDIA RTX A4000`) and the pass prefixes. On the box,
  `--check-installed-plan` asserts that the installed `int4_b32._plan` picks exactly the sks the reducer prices, on all
  48 cells at the card's SM count.

At 48 SMs the R term changes the pick on **all 24 cells at R ≥ 16** (sk 2 or 4 at R = 16, sk 1 above it), and on none
below the floor. The self-test asserts both.

## Correctness first (no timing if any fails)

These are K30's four gates, with K31's names:
1. `kernel/test_int4_b32.py -k "gemv_matches_reference or reduce_partials or plan_"`, compiled on the card
   (`TRITON_INTERPRET=0`), rc 21 on failure.
2. `bench/int4/k30_sk_check.py`, rc 22 on failure: at R = 16 and 128 on every shape, every sk the sweep will time must
   match sk = 1 within split-K's fp32 reorder (max |Δ| ≤ 1e-2 · max |out|).
3. `k31_reduce.py --self-test`:
   - K30's seven rule cases, against this card;
   - three cases of K31's own: K30's card, an RTX 4000 Ada (VOID) and a clear win here (KEEP);
   - the 24-of-48 assertion above.
4. The installed-plan cross-check.

## Rule: K30's, unchanged

The rule is read over the 24 cells at R ≥ 16, pooled over both passes:
- r_c = (R-aware p1 + p2) / (N-only p1 + p2);
- S = summed R-aware time / summed N-only time;
- W = the largest r_c.

**VOID** if any of these holds:
- the card is not `NVIDIA RTX A4000`, or has more than 64 SMs;
- a cell or a pick is missing;
- the passes disagree on the card;
- the two passes' own S differ by more than 0.02.

**KEEP** if S ≤ 0.97 and W ≤ 1.02. **OFF** otherwise. K30's reasons for 0.97 and 1.02 apply here unchanged: they are
judgement written before any data, not derived from any timing.

## Consequence: how K31 combines with K30 (registered now)

The gate covers the whole class, and the owner chose two cards with "KEEP needs both". So:

| K30 (L4) | K31 (A4000) | consequence |
|---|---|---|
| KEEP | KEEP | **The term stays on.** Two kernel-level rows replace the retired A2000 row as its receipts. Whether a step-level read on a ≤64-SM card is worth registering is decided then, knowing it would need `E4B_INT4_GROUPED_SMALLM=0` (above). |
| OFF | any | **The term goes off**, by K30's own consequence. |
| any | OFF | **The term goes off.** `SPLITK_R_TERM_MAX_SMS` is set so that no part takes it, and every card plans N-only, as the 5090 already does. The change is reorder-class on the three paths above, and nothing changes on the default server. |
| KEEP | VOID, not run | K30's KEEP does not stand alone for the class. The term stays as shipped until K31 reads, inside its ceiling. |
| VOID, not run | KEEP | K31's KEEP does not stand alone either. The term stays as shipped until K30 reads. |

K31 adds one condition to K30's consequence and nothing else. A K30 KEEP keeps the term on only together with a K31
KEEP. That is the owner's "KEEP needs both". Each lane's own rows and results stand as read.

## Predictions (direction only; no band, as K30)

No band is registered. The only timings of this term are the retired A2000 sweep and P39's 5090 step read. Neither is a
basis here, and none is borrowed.

The direction comes from byte arithmetic on the shapes, at 48 SMs:
- **At R ≥ 32 the R-aware pick is sk 1.** The N-only pick (sk 4–16) writes and re-reads sk × R × N fp32 partials, about
  12 % of the bytes the expert weights move on these shapes, since the span cap keeps sk near K / 128. At sk 1 the
  wrapper still runs `reduce_partials` as the bf16 cast (K30 Amendment 1), so the saving is (sk − 1) / sk of that
  partials traffic.
  - At R = 32 the grid is already 8–16 tiles × 32 rows = 256–512 programs on 48 SMs, so dropping the split costs
    little occupancy.
- **At R = 16 the R-aware pick keeps a split of 2 or 4**, against 4–16. The difference there is small and could go
  either way.

**My expectation is KEEP**, driven by the R ≥ 32 cells. That is a stated expectation, not a registered band.

## What this lane cannot say

- **No step-level effect.** The term is not on the default server's path at all (above).
- **Two cards are still not every card.** A 4060 Ti (34 SMs) or an RTX 4000 Ada (48 SMs, sm_89) may read differently.
- **The constant is not refitted.** The per-cell optimum is reported beside the verdict, as information.

## Rehearsal (correctness only: the A2000 is never a timing instrument)

**What runs.** The whole box script runs end to end on the NAS RTX A2000, in a throwaway `pytorch:2.8.0-cuda12.8`
container, before any rental, with `K31_REHEARSAL=1`. That flag lifts the card check and sweeps two R values per shape.

**What it checks:**
- install and the tripwire;
- the four correctness gates;
- both passes;
- the reducer's VOID path, which must read VOID on an A2000.

No A2000 time is quoted anywhere, and its rows are not committed as receipts.

## Budget and launch

- **Card.** One NVIDIA RTX A4000 (48 SMs, sm_86, 16 GB) on `vast:verified-secure`.
  - Host-RAM band [12, 128] GB. Verified offers on 2026-10-05 had 15–62 GB, and the lane loads no model.
  - `preflight_bandwidth: none`: there is no Hugging Face checkpoint.
  - Declared ≤ $0.20/h. Verified offers were $0.088–0.149/h on 2026-10-05.
- **Prerequisite.** `RTX A4000` in adertha-agents `allowed_gpus` (adertha-agents#167), merged before the
  first launch.
- **Guard and proving rental: K30's.** First pass ~330 kernel variants compiled; no runtime on this card to size from;
  A2000 durations not used. So:
  - a **proving rental** first: `k31-prove-<n>`, `K31_PROVE=1`, a 0.25 h guard, card check and forensics only;
  - then the reading, `k31-a4000-<n>`, a **1.5 h guard**;
  - `K31_PASS_NEED_S` = 1,500 s a pass, and a pass that cannot fit is skipped `host-limited`.
- **Estimate.** The proving rental costs ≤ $0.05 and the reading ≤ $0.30 at the declared ceiling, plus the launcher's
  download allowance in its own estimate. Both are inside the standing no-ask tier (experts4bit-qlora#564
  issuecomment-5936673039).
- **Stops and attempts.**
  - The lane's hard stop is **$1.50**.
  - Each stage gets at most two attempts.
  - A pre-flight VOID excludes its machine by receipt for the second attempt. A second VOID closes the lane
    `host-limited`.
- **Where things live.**
  - The runner is experts4bit-qlora `bench/k31/`, K30's runner by named substitution. gnf4 is cloned on the box at
    `GNF4_SHA`, this registration's commit.
  - Launches go through `pod-launch.sh` from this session's own checkouts on the mini.
  - Receipts go to `kernel/receipts-k31/a4000/`, and results to `kernel/RESULTS-k31-splitk-r-term-a4000.md`.

## STOP rules

1. **No timing without correctness.** A failed gate ends the run with rc 21 or 22.
2. **No pass starts that cannot finish 10 minutes before the launcher's deadline.** A skipped pass is `host-limited`,
   and the lane reads VOID.
3. **No result is reported before its receipt exists** in the private record.
4. **Any teardown that cannot be proven stops the lane.**

Amendments, dated, go below this line before any data is read.

## Withdrawn — 2026-10-05, before any run

**The owner chose to fold the A4000 into K30** rather than run a separate lane. The question was put in this
session's chat after the maintainer session decided the same on #478. K31 never rented, rehearsed or read anything.

**Where the A4000 reading lives now.** K30 (grouped-nf4-gemm#478, merged `ccf4de9`) carries the RTX A4000 as its box 2,
under its Amendment 2:
- the A4000 is rented only after an L4 KEEP, since an L4 OFF already decides the default;
- KEEP needs the rule to hold on both cards, and OFF follows if either fails.

That is the combination this registration proposed. Box 2 runs under K30's run ids and manifests. The work item
(#479) is closed, and nothing here is deleted.
