# K33 — does the bandwidth-targeted NF4 decode GEMV (`GNF4_GEMV_BW=1`) run the single-row expert projections near the streaming ceiling, faster than the served route? Every layer of Qwen3-30B-A3B, Granite-3.1-3b-a800m and OLMoE-1B-7B in CUDA graphs, on one RTX 5090 (registered 2026-10-07, before any 5090 run)

Lane number claimed by `prereg/k33` (pushed 2026-10-07T17:42:12Z). Issue: experts4bit-qlora#1313 (single-stream decode on
the default NF4 server). The kernel and this bench: #500.

## Why now

**What e4b's censuses found.** At B=1 on Qwen3-30B-A3B, experts4bit-qlora's SV2 census
(`bench/hybrid-g9/sv2/RESULTS-sv2-device-census.md`, one RTX 5090) put the served NF4 expert GEMV,
`_gemv_nf4_dotpad`, at 2.469 ms of a 6.46 ms graphed decode step, 96 launches, at 3.8× its byte floor. The NF4
families whose shapes dot-pad does not cover run the scalar GEMV instead: e4b's P93 census reads 1.98 ms of a 3.9–4.3 ms
B=1 step on Granite and OLMoE, at ~18 % of the streaming ceiling (`RESULTS-m1-decode-config.md`).

**Where the time goes, from this repository's own lanes.**
- K7 (`RESULTS-k7-gemv-round2.md`): dot-pad at Qwen3's shapes is 27.2 µs (gate_up) + 18.8 µs (down) per 8-row launch in
  an isolated graph. Split-K did not help it.
- K4 (`RESULTS-k4-wide-loads.md`): the gate_up access pattern alone streams in 9.98 µs with 32-bit word loads. The rest
  of that kernel was shifts, the per-element codebook gather and the multiply-reduce.
- K26 (`RESULTS-k26-nf4-decode-ablation.md`): the codebook lookup is ~80 % of the NF4 small-M kernel; an exact
  4-level select tree is 2.6–2.7× faster.
- The int4-b32 GEMV reads the same bytes per parameter (0.5625 B) at ~66 % of the ceiling with an arithmetic decode.

**What #500 adds** (`GNF4_GEMV_BW=1`, opt-in). `_gemv_nf4_bw` loads one 16-byte-aligned `[BLOCK_N, KC/64, 8]` word tile
per K-step, the int4-b32 tile shape, and decodes the fp32 codebook exactly without a memory gather: `prmt32` uses PTX
byte-permute (`prmt`) lookups of the codebook's byte planes and `lop3` selects; `tree` is K26's select tree. It applies
`absmax` per 64-block as `dequant_ref` does and accumulates in fp32. `prmt32` is bitwise the tree, one-hot activations
read `dequant_ref` exactly, and the result never depends on the row count (#500's tests, run on an RTX A2000 for
parity). Nothing has timed it on a registered card.

## The bench (`kernel/k33_bench.py`)

**The work.** One expert projection's decode launches over every layer of a family, in one CUDA graph, at the family's
served shape: 8 activation rows, each the top-8 routed experts of that layer.

| family | layers | gate_up (N × K) | down (N × K) |
|---|---|---|---|
| Qwen3-30B-A3B | 48 | 1536 × 2048 | 2048 × 768 |
| Granite-3.1-3b-a800m | 32 | 1024 × 1536 | 1536 × 512 |
| OLMoE-1B-7B | 16 | 2048 × 2048 | 2048 × 1024 |

**Stores.** Each layer has its own synthetic NF4 stack of 16 experts (random packed bytes, absmax in [0.5, 1.5)),
seeded, so a graph streams every layer's bytes and nothing stays in L2 (K24's lesson); the 8 experts per layer come from
a seeded draw.

**Arms**, one graph each per (family, projection). Every arm runs with `GNF4_PDL=0` except `bw_pdl`.

| arm | what runs |
|---|---|
| `incumbent`, `incumbent2` | today's route (`GNF4_GEMV_BW=0`): dot-pad at Qwen3's shapes on this card, the scalar GEMV elsewhere; the repeat is the instrument |
| `scalar` | the certified scalar GEMV (`GNF4_GEMV_DOTPAD=0`) |
| `bw_tree`, `bw_prmt32` | `_gemv_nf4_bw` with each decode, at the plan this run selected |
| `bw_pdl` | `bw_prmt32` with `GNF4_PDL=1` (reported) |
| `int4` | `quant_x_rows` + `gemv_int4_b32` on int4-b32 stores of the same shape: the named comparator, same bytes per parameter, another format (reported, not ruled) |

**Plan selection** (K20's discipline), per projection. Every plan in `PLANS` (12: BLOCK_N 16 or 32, KC 256 to 1024,
4 or 8 warps, split-K 1 or 2) is captured with `bw_prmt32` and timed for 40 rounds. A plan that fails to compile or
launch is not selectable and is recorded. The fastest median wins. The arms are then timed afresh.

**Numerics before timing**, at the selected plan, on layer 0's 8 rows: `bw_prmt32` must be bitwise `bw_tree`, and both
must be within the tolerance contract of the fp32 reference (`dequant_ref` × fp32 activations): error at most 5 %
above the scalar route's, or at most 2^-9 of the output scale.

**Engagement.** `dispatch_counts()` across each capture must show the registered route on every layer and nothing else.

**Timing.** 20 warm rounds, then 200. Each round replays every arm's graph once, in an order that reverses every round,
timed by CUDA events. Medians and quartiles per arm.

**The floor.** Active bytes (layers × 8 rows × N × (K/2 + K/16)) over this box's own copy bandwidth (a 512 MiB device
copy, read plus write).

## The rule (`verdict`, self-tested on 14 cases)

The rule reads the Qwen3 projections. The first that applies:

1. **VOID.** Any of:
   - the card is not an RTX 5090;
   - the copy floor was not measured;
   - a Qwen3 arm's timing is missing;
   - no plan was selectable for a Qwen3 projection;
   - any capture's tally is not its registered route.
2. **NOISY.** `incumbent2 / incumbent` falls outside [0.98, 1.02] on either Qwen3 projection.
3. **FUNCTION_FAIL.** At any family's selected plan, `bw_prmt32` is not bitwise `bw_tree`, or a bw decode misses the
   tolerance contract.
4. **LEVER.** With bw = the faster contract-exact decode on the pair, all of:
   - bw / incumbent ≤ 0.77 on each Qwen3 projection;
   - bw / incumbent ≤ 0.667 on the pair (gate_up + down), that is ≥ 1.5×;
   - floor / bw ≥ 0.45 on each Qwen3 projection.
5. **PARTIAL.** The pair ≤ 0.80, but not LEVER.
6. **NO_LEVER.** Otherwise.

**Reported:** every arm's median and quartiles on every projection of every family; the family ratios against
`scalar`; `bw_prmt32 / bw_tree`; `bw_pdl / bw_prmt32`; `int4 / bw`; the selected plans, every plan's selection median and
every refused plan; the numerics' errors; the copy bandwidth.

**Why these bars.** 1.5× on the Qwen3 pair is what the served step needs from this slice to be worth a served lane:
roughly 1 ms of kernel time at B=1. Under the transfer K6b measured from kernel to step (about 0.55 of a kernel win
reached the step), that is 7–8 % at W1, comfortably above a served lane's self-pair band. 1.25× (PARTIAL) would read
3–4 %: readable, marginal. 0.45 of the floor is half of the int4-b32 GEMV's ~0.66 on the same bytes, so a kernel that
clears it has room left.

## Predictions (written before any data)

Basis: K7's dot-pad per-launch times, K4's wide-load floor, and an instruction-issue estimate (the 5090's ~5.7e13
thread-instructions per second against 1.8e9 weights per step: ~20 instructions per weight at the byte floor;
`prmt32` costs ~6–8 per weight with its FMA, the tree ~25). These were written in the lane's design, before any GPU ran
the kernel.

| # | prediction |
|---|---|
| Q1 | numerics and engagement hold everywhere; every plan in `PLANS` compiles |
| Q2 | per 8-row launch at Qwen3's shapes: `bw_prmt32` 13–17 µs (gate_up) and 7–9 µs (down); `bw_tree` 19–25 and 10–13 |
| Q3 | Qwen3 ratios: gate_up 0.48–0.63, down 0.36–0.48, pair 0.43–0.57; floor / bw 0.50–0.70 on each |
| Q4 | `bw_prmt32` is the faster decode at every projection |
| Q5 | the selected plans have split-K 1 and BLOCK_N 16 |
| Q6 | Granite and OLMoE: bw / scalar ≤ 0.50 at every projection |
| Q7 | `int4 / bw_prmt32` lies in [0.75, 1.0]: the arithmetic int4 decode stays somewhat faster |
| Q8 | `bw_pdl / bw_prmt32` lies in [0.97, 1.0] |
| Q9 | instrument within [0.99, 1.01] |
| Q10 | the verdict is LEVER |

## Consequence, registered now

- **LEVER:** the served lane in experts4bit-qlora (P116): the default NF4 `serve_paged` server with `GNF4_GEMV_BW=0`
  against `1`, W1 and W16, decode-only slopes, P110's teacher-forced quality bar (tokens are not expected to match
  dot-pad's bf16 MMA). On its DEFAULT read, this repository fills `_BW_SHAPES` and makes `auto` the default, with the
  `test_m3_defaults` trio and a release; the consumer floors on that release.
- **PARTIAL:** `GNF4_GEMV_BW` stays opt-in and a second round is registered, aimed by the reported arms.
- **NO_LEVER:** it stays opt-in; the ratios are recorded.
- **FUNCTION_FAIL:** the failing decode is withdrawn until explained; its timings are disclosed, not ruled.
- **NOISY or VOID:** no consequence; one rerun inside the ceiling, then an amendment.

## What was seen before this page (stated, not hidden)

- **On an RTX A2000** (sm_86, the NAS card; correctness only under the #1133 policy): #500's compiled tests passed
  (`prmt32` bitwise the tree at six plans and all six family shapes, one-hot readback exact, the tolerance contract, the
  PTX), and a `--quick` rehearsal of this bench (two plans, 20 rounds) ran end to end on all three families and read
  VOID by card class, as registered. Every capture ran its registered route and the numerics held. The rehearsal
  recorded timings. Under #1133 they are not speed evidence: they seeded no prediction above and filtered no plan, and
  they are not quoted.
- **The 5090 has not run any of this.**

## Box and cost

- **`k33-5090-<n>`:** one RTX 5090. **Guard 0.5 h at ≤ $0.75/h (≤ $0.375)**, so there is no proving rental. No model.
- **Driven from** experts4bit-qlora `bench/k33/` (K28's runner with the bench, the tripwire and the premise replaced):
  - the tripwire proves the installed gnf4 is the pinned commit and carries `_gemv_nf4_bw`, the `GNF4_GEMV_BW` switch,
    the `bw_*` tally keys and an empty `_BW_SHAPES`, and that `prmt32` is the decode on this card (rc 9);
  - the premise is `kernel/test_nf4_gemv_bw.py` compiled on the card: **27 passed, none skipped** (rc 23). On sm_90+ its
    two PDL tests run too.
- **Lane ceiling $0.75; hard stop $1.00.**
- **STOP rules:** a card that is not a 5090 is refused (rc 15); a failed premise stops before the bench (rc 23); a
  NOISY reading is rerun once inside the ceiling, then stops.

## What this lane cannot say

- **No served speed.** There is no e4b, no attention, router or glue: the served lane (P116) reads the step.
- **Nothing about B=16** (other kernels), other cards, or Mixtral (no census row; not run).
- **Nothing about quality beyond the tolerance contract.** The served lane carries the teacher-forced read.

## Receipts

`kernel/receipts-k33/5090/`: `k33.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `logs/` (the premise and the
bench), the teardown proof and `SHA256SUMS`. `RESULTS-k33-nf4-decode-gemv-bw.md` is written from those files.

Amendments, dated, go below this line before any 5090 data is read.
