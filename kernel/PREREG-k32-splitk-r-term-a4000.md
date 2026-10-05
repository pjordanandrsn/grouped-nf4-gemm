# K32 — K30's split-K R-term sweep on one rented RTX A4000, the second ≤64-SM architecture (registered 2026-10-05, before any A4000 run)

Owner instruction, in the maintainer session's chat on 2026-10-05, after K30 closed OFF: **"rent the A4000 for the
second card."** Written down by the agent, not by the owner.

## Why this lane exists, and what it can and cannot change

K30 (#478, read in #481) read the int4-b32 split-K R term **OFF** on a rented NVIDIA L4. That L4 has 58 SMs and is
sm_89. Over the 24 cells at R ≥ 16 the R-aware pick's summed time was 1.0101× the N-only pick's, and its worst cell
1.1781×. #482 then set `SPLITK_R_TERM_MAX_SMS = 0`, so every card plans N-only.

K30's Amendment 2 registered the RTX A4000 (48 SMs, sm_86) as a second card, to be rented only after an L4 KEEP. Under
the lane rule ("KEEP needs both"), an L4 OFF decided the lane, so the A4000 was never rented. K31 (`prereg/k31`) was
another session's registration of the same A4000 read; it was withdrawn before any run when the A4000 was folded into
K30. Its number is not reused.

The owner now asks for the A4000 read anyway, under the two-card rule for performance portability. **This lane cannot
change the default.**

- **K30's OFF stands as read.** The default stays N-only on every part whatever the A4000 reads, because "KEEP needs
  both" and the L4 already read OFF.
- **What this lane adds** is whether the OFF also holds on the other ≤64-SM architecture, on a rented card.
- **The term's original constant was fitted on sm_86** (the A2000; inadmissible speed evidence, Amendment 1 of K30).
  So an sm_86 card is the architecture where a win was most plausible.

## Instrument and rule: K30's, frozen

- **Instrument.** grouped-nf4-gemm at **K30's measured cut `aa562718b9eadceeab746f0fb8d07462e23c2b28`** (tree-identical
  to main's `ccf4de91` for `kernel/` and `bench/int4/`). It is installed and cloned on the box: `bench/int4/sk_sweep.py`
  (the served two-launch path, with `reduce_partials` at every sk, per K30's Amendment 1), `k30_sk_check.py` and
  `k30_reduce.py`.
  - The pin is deliberate. Main now ships `SPLITK_R_TERM_MAX_SMS = 0` (#482), and the reducer's
    `--check-installed-plan` asserts the gate it prices (64), so it refuses on main by design.
  - The sweep times every sk whatever the gate. The pin only lets the reducer cross-check the R-aware pick it prices.
- **The box runner** is experts4bit-qlora `bench/k30/k30_run.sh`, unchanged, at main's head, with `K30_CARD=A4000`.
  - The exact card name must be `NVIDIA RTX A4000`, with at most 64 SMs.
  - The runner ran end to end on the L4 at the same bytes (`k30-l4-1`). Only the card-name branch differs, and
    `tests/test_k30_staged_pin.py` pins it.
- **The four correctness gates run before any timing**, exactly as in K30: the compiled GEMV/reduce/plan tests, the
  per-sk agreement check, the rule's self-test (the reducer accepts `NVIDIA RTX A4000` by exact name), and the
  installed-plan cross-check at the card's SM count.
- **The rule is K30's, unchanged.** Over the 24 cells at R ≥ 16, pooled over two passes: **KEEP** if S ≤ 0.97 and
  W ≤ 1.02, **OFF** otherwise.
  - **VOID** on the wrong card or more than 64 SMs, a missing cell or pick, passes that disagree on the card, or
    passes whose own S differ by more than 0.02.

## What each verdict means (fixed now)

| A4000 | meaning | what follows |
|---|---|---|
| **OFF** | the term does not earn its default on either ≤64-SM architecture read | nothing changes; one measured row; K30's OFF is now two-card |
| **KEEP** | the term wins at kernel level on sm_86 but not on sm_89 | nothing changes to the default ("KEEP needs both"). One measured row, and the result is reported as an architecture-specific observation. An sm_86-only gate would be a new registration, and it is low priority: the split-K GEMV is off e4b's default server path since P88. |
| **VOID** | the instrument could not read | reported as VOID; one more attempt at most (a pre-flight refusal excludes its machine), then `host-limited` |

## Predictions (direction only; no band)

- **No band is borrowed from any timing.** The A2000's are inadmissible. The L4's are a different card and are not used
  as a basis.
- **My expectation is OFF.** Byte arithmetic gives no reason for sm_86 to differ from sm_89 on the cells that decided
  K30. Those were the large gate_up cells at R = 64–128, where dropping the split to sk 1 left too little parallelism on
  a ≤64-SM part, and 48 SMs is fewer still.
- **The rule decides,** and my K30 expectation was refuted.

## Budget and launch

- **Provider, card and rate.**
  - One RTX A4000 on `vast:verified-secure`.
  - Host-RAM band [16, 128] GB: verified A4000 hosts at the policy's 320 GB disk floor are small. On 2026-10-05 the
    cheapest qualifying offer was $0.088/h, with a 2-core host whose CPU slows the Triton compiles but not the GPU-side
    graph-replay timings.
  - `preflight_bandwidth: none`; declared ≤ $0.40/h.
- **Guard: 1.0 h.** That is not over 1 h, so no proving rental is required. K30's proving rental already proved this
  provider class and image.
  - The guard is sized from the rented L4 run, which took about 5 minutes end to end. It is not sized from any A2000
    duration.
- **Cost.** Estimate ≤ $0.40 at the declared ceiling; expected under $0.15. Hard stop $1.
- **Where things live.**
  - Run `k32-a4000-<n>`; work id this registration's PR.
  - Launched through `pod-launch.sh` from session-owned checkouts on the mini.
  - Receipts: `kernel/receipts-k32/a4000/`. Results: `kernel/RESULTS-k32-splitk-r-term-a4000.md`.

## STOP rules

As K30:
1. No timing without correctness.
2. No pass that cannot finish 10 minutes before the deadline.
3. No result before its receipt exists.
4. Any unproven teardown stops the lane.

Amendments, dated, go below this line before any data is read.
