# K19 A2000 probe (2026-10-01): exploratory, not a registered claim

Recorded B=16 routing from experts4bit-qlora lane P60 (`bench/p60/receipts/eids_b16.int16.bin`, 128 steps × 48 layers
× 16 rows × top-8), Qwen3-30B-A3B's expert shapes (gate_up N=1536 K=2048; down N=2048 K=768), synthetic int4-b32
weights, on the NAS RTX A2000 (sm_86, 26 SMs; idle before the runs). Each arm is one CUDA graph per decode step,
8 steps, 20 replays each. `k19_probe.py` imports experts4bit-qlora's `bench/p60/replay_gemv.py` (P60's served arm).

| arm | graph step (ms) | kernel time (ms) | launches / step |
|---|---:|---:|---:|
| served: `_gemv_int4_b32`, R = 128, + `_reduce_partials` | 79.28 | 78.99 | 96 |
| K16 once per distinct expert (`gemm_int4_b32_smallm`) | 73.31 | 69.62 (55.41 + 14.21 of row gathers) | 5,294 |
| **K19** (`gemm_int4_b32_grouped_smallm`), incl. one `_tile_table_r1` per layer | **41.88** | 43.15 (41.75 + 1.16 tile build) | 144 |

- `probe.json` is the first run (served and K16 arms), and `probe2.json` the second, with the K19 arm.
- Both logs (`run.log`, `run2.log`) carry the card's state before the runs.
- `run2.log` also records the contract tests: 20/20 under the interpreter and 20/20 compiled on the A2000. The
  packaging test's 4 failures there are the container's partial copy (no `pyproject.toml` or workflow); it passes in
  the full checkout (5/5).

This is the A2000, whose compute/bandwidth balance differs from an RTX 5090's: the ratio does not transfer, and no
speed claim is made from it. The 5090 measurement is a separate, pre-registered lane.
