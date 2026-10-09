# K34 — which K16 plans run the 32- and 64-row tiles fastest at Qwen3-30B-A3B's attention shapes on an RTX 5090? An exploratory census (registered 2026-10-09, before any 5090 run)

Lane number claimed by `lane/k34-k16-plan-census` (pushed 2026-10-09T06:25:02Z).
Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). The maintainer's
GO for this census (bus, 2026-10-09T06:18Z) set two conditions, and this registration carries both:
1. **The census is exploratory.** Picking the fastest of 48 plans has a winner's curse, so its winner licenses nothing.
   Any plan that would move a default gets its own registered confirmatory read against the shipped plan, using P124
   Amendment 1's interleaved blocks on SC2e's served step.
2. **A different split-K or KC is a different arithmetic.** It reorders the fp32 sums. Within one bf16 ulp is this
   census's sanity check, not a quality licence. The confirmatory read carries P110's teacher-forced bar with a mutant,
   as P124 did.

No kernel code changes: `gemm_int4_b32_smallm` already takes `block_n`, `kc`, `sk`, `warps`, `stages` and `block_m`
(#522).

## Why

- **P124 read the route DEFAULT_ON.** experts4bit-qlora P124 (#1427) read `E4B_ATTN_INT4_WIDE` DEFAULT_ON: Qwen3-30B-A3B's
  64- and 32-row decode steps 3.2–3.8 % faster with the attention projections on K16's 32- or 64-row tile instead of
  cuBLAS on a cached bf16 copy.
- **The shipped plan was never tuned for these tiles.** At the shipped plan (BLOCK_N 64, KC 128, split-K 4, 4 warps,
  2 stages), the 64-row tile still cost **0.74** of the cuBLAS time it replaced (1.38 of 1.88 ms a step) and the 32-row
  tile 0.59. The int4 bytes are a quarter of the bf16 copy's. That plan was chosen for K16's 16-row tile on the A2000.
- **The question:** does another plan do markedly better at the larger tiles on the target card?

## Instrument (`kernel/k34_bench.py`)

- **Stores:** per layer its own synthetic int4-b32 store (random packed bytes, fp16 scales), seeded. The two projections
  are the ones P124 served after the q/k/v fusion: `qkv` (N 5120, K 2048) and `o` (N 2048, K 4096), 48 layers each, so a
  graph streams every layer's bytes and nothing stays in L2 (K24's lesson).
- **Graphs:** one per (projection, tile, arm or plan). Each holds 48 launches, one per layer, of
  `gemm_int4_b32_smallm` on the tile's rows (32 or 64), with a workspace sized for 64 rows shared by the layers, as
  experts4bit-qlora shares it.
- **Arms:**
  - `shipped` and `shipped2`: the shipped plan; the repeat is the instrument;
  - `best`: the plan this run selected;
  - `bf16`: cuBLAS on each layer's dequantised bf16 weight, the path the route replaced (reported).
- **The plan grid:** BLOCK_N ∈ {32, 64, 128} × KC ∈ {128, 256} × split-K ∈ {1, 2, 4, 8} × warps ∈ {4, 8}, 2 stages:
  **48 plans**, the shipped plan among them.
- **Selection** (K20's discipline):
  - a plan that fails to compile or launch is recorded and not selectable;
  - a plan whose layer-0 output is further than one bf16 ulp (2^-7 of the output's scale) from the fp32 dequant
    reference is recorded and not selectable;
  - every other plan is timed for 40 SELECT rounds, and the fastest median is `best`;
  - the arms are then timed afresh for 200 rounds, order reversed every round, with CUDA events.
- **The floor:** the graph's int4 bytes over this box's measured copy bandwidth (a 512 MiB copy).

## Rule (`k34_bench.py`'s `verdict`, 12-case self-test), on the re-timed medians

1. **VOID:** the card is not an RTX 5090; the copy floor or a timing is missing; no plan was selectable for a
   (projection, tile).
2. **NOISY:** `shipped2 / shipped` outside [0.98, 1.02] for any (projection, tile).
3. **CANDIDATE:** for a tile, `best / shipped` ≤ **0.90** on both projections. That tile's selected plan is named as a
   candidate, and **nothing moves**. The next step is a registered confirmatory read in experts4bit-qlora:
   - the candidate plan against the shipped plan for that tile;
   - P124 Amendment 1's interleaved blocks on SC2e's served step;
   - P110's teacher-forced bar and a mutant;
   - only that read can make a plan the tile's default.
4. **NONE:** otherwise. The shipped plan stays.

**Reported beside the verdict:**
- every selectable plan's SELECT median;
- every refused plan, with its reason;
- every plan's layer-0 numerics;
- `best` against `bf16`, and both against the floor.

## Predictions (written before the data)

- **The instrument holds:** `shipped2 / shipped` within 1 % on every (projection, tile).
- **Some plan beats the shipped one at 64 rows by 10–25 %.** My guess is 8 warps and a smaller split-K: a 64-row tile
  holds four times K16's accumulator, and split-K 4 with 64 rows reduces four [64, N] partials.
- **The verdict: CANDIDATE at 64 rows; NONE or a smaller gain at 32 rows.**

## Rehearsal (correctness only: the A2000 is never a timing instrument)

On the NAS RTX A2000, gnf4 `4ed26d9`, a throwaway `pytorch:2.8.0-cuda12.8` container:
- `k34_bench.py --self-test` passed (12 cases);
- `--quick` (two plans, 20 rounds) ran end to end: every graph captured, plans selected and re-timed, both projections at
  both tiles;
- the verdict fired VOID, as it must, off the target card.

No A2000 time is quoted anywhere.

## Budget

- **One RTX 5090** (verified/secure), **0.5 h guard at ≤ $0.75/h (≤ $0.375)**. No model download. The guard is under
  1 h, so no proving rental is needed.
- **Runtime estimate:** install about 3 min, then K16's compiled contract tests on the card first (no timing if they
  fail). Then 2 projections × 2 tiles × 48 plans × 40 SELECT rounds and 4 arms × 200 rounds: about 10 min of GPU.
- **Hard stop $1.00.** Lane `k34-5090-<n>`, driven from experts4bit-qlora `bench/k34/` (K33's runner with the bench,
  the tripwire and the premise replaced).
- **Premise on the card:** `kernel/test_int4_smallm_interp.py` compiled (`TRITON_INTERPRET=0`), 25 passed, none skipped.
- **Receipts** in `kernel/receipts-k34/5090/`; results in `kernel/RESULTS-k34-k16-wide-plan-census.md`.

Amendments, dated, go below this line before any data is read.
