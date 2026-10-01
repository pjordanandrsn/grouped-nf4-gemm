# K24 — K22 re-read: per-layer weight stores and K21's masked-tail plans, on gpt-oss-20b's recorded B=16 routing (registered 2026-10-01, before any data)

**K22** (`RESULTS-k22-gptoss-mxfp4-b16.md`, #424) read VOID by its instrument. Its bench's served `_gemm_nf4_grouped`
read 14.81 ms per step against the same box's in-model census of 17.86, outside the 15 % band. Descriptively it found:
- gpt-oss-20b's B=16 step is 79 % that one kernel (17.86 of 22.62 ms);
- K21 read 0.68× the served route, but at only 37 % of the MXFP4 byte floor;
- gpt-oss's K = 2880 capped K21 at KC 64.

Two changes answer it:
1. **Per-layer stores** (`kernel/k24_bench.py`). K22 shared one synthetic weight set across all 24 layers. The likely
   cause of the miss, inferred, is that L2 then serves experts that consecutive layers both touch. The model never gets
   that reuse.
   - Every layer now has its own NF4 and MXFP4 stores, generated on the GPU: about 21 GB, which fits a 32 GB 5090.
   - If the instrument still misses, the cause is elsewhere, and the read says so.
2. **K21's masked K tail** (#425). KC 128 and 256 now run on K = 2880. The grid is BLOCK_N {32, 64, 128} × KC {64, 128,
   256} × warps {4, 8} × stages {2, 3, 4} = 54 plans, and the default plan is K21's own (32 / 256 / 4 / 2).

**Unchanged from K22:**
- the arms (served NF4 route, MXFP4 GEMV at 64 rows, K21 × plans, floor);
- the select (steps 0–7) / read (steps 8–15) split;
- the rule (5-case self-test):
  - **VOID** unless the served `_gemm_nf4_grouped` is within ±15 % of a same-box census;
  - **PROMISING** if best K21 / served ≤ 0.77;
  - **MARGINAL** if ≤ 0.90;
  - **NO** otherwise.
- the routing: K22's recording (`receipts-k22/5090/eids_b16.pt`, `[128, 24, 16, 4]`), read from the clone at the
  registered commit rather than recorded again.
- the census arm: bo7's `store_r12` at B=16 with P42's replay census, run first on the same box.

**What follows a PROMISING read:**
- an experts4bit-qlora route for the MXFP4 store's batched rows to K21, opt-in;
- an end-to-end lane on gpt-oss-20b with the store's KL instrument as the quality gate (K21's weights are exactly the
  store's);
- only then a default.

## Predictions (written before the data)

- **The instrument holds** (within ±15 %) with per-layer stores. That would confirm K22's inferred cause.
- **K21 PROMISING, best / served 0.45–0.65,** with KC 256 on top, as on Qwen3. The masked 64-wide tail is one of 12
  chunks, so it costs little.
- **K21 reaches 60–80 % of the MXFP4 floor,** short of Qwen3's 89 %: gpt-oss's experts are 4.4× larger per row of the
  tile, so the same tile count streams more bytes per program.

## Budget

- One RTX 5090, any CPU vendor, no calibration.
- The runner is K22's minus the routing record: fetch, bake, census, bench.
- **Guard 1.25 h at ≤ $0.75/h (≤ $0.94).** That exceeds 1 h, so a proving rental runs first: K21's masked-tail contract
  compiled on sm_120 and the self-test, no model; 0.5 h, ≤ $0.375.
- **Hard stop $1.50.** Lane `k24-5090-<n>`, driven from experts4bit-qlora `bench/k24/`.
- Receipts in `kernel/receipts-k24/5090/`.

## Rehearsal (correctness only)

On the NAS A2000, `k24_bench.py --self-test` passed (5 cases). `--quick` ran end to end on K22's real recorded routing,
restricted to its first 4 layers, since a 12 GB card cannot hold 24 layers' stores:
- per-layer stores generated;
- every arm timed;
- the masked KC 256 plan bit-identical to KC 64, and 0.0030 from the oracle;
- the instrument VOIDed off the target, as it must.

No A2000 time is quoted.

Amendments, dated, go below this line before any data is read.
