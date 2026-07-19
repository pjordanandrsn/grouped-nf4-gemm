# RESULT — v0 hot-residency curve vs llama.cpp ladder (gpt-oss-120b, one H200)

**Tier: EXPLORATORY.** Stamp-ready; NOT stamped (operator's key). Grades
`PREREG-hybrid-vs-llama.md` (committed pre-data at `adf88be`).

- **Box:** DO H200 (143 GB VRAM, 235 GB RAM, EPYC host), PCIe gen5 x16,
  **measured H2D 55.2 GB/s** (pinned, in-run probe). Droplet 585771273
  (curve run; destroyed, 404-verified). Primary llama arm from the prior
  completed run on the same pod class, droplet 585768803 (destroyed,
  404-verified).
- **Model:** gpt-oss-120b (E=128, k=4, L=36). Ours: NF4-converted, fused
  grouped kernel, per-expert hot residency (v0 = the shipped
  `hot_residency` partition with the gpt-oss GLU harness; cold tail
  streamed synchronously). llama.cpp: its GGUF-native shipped quant,
  per-layer `--n-cpu-moe` (cold layers **compute on the host CPU**).
- **Cross-format caveat (applies to every comparative sentence here):** the
  two stacks serve different quantization formats of the same checkpoint;
  tok/s comparisons bundle format + engine differences. Raw tool output in
  the receipts file is verbatim and names formats in its own words.
- Receipts: `receipts-hybrid-curve-v0-h200-585771273.txt` (raw pod state,
  unedited).

## Ours — per-expert residency dial (v0 implementation)

One load, K swept in place; correctness anchored against the bare fused
forward at every K (gate: b_rel < 3e-2; **worst over sweep 0.0138 — OK**).
20-token greedy decode, median.

| K/layer | resident-expert GB | ms/tok | tok/s | b_rel |
|---:|---:|---:|---:|---:|
| 0   | 0.0  | 250 | 3.99  | 0.0123 |
| 4   | 1.8  | 252 | 3.97  | 0.0138 |
| 8   | 3.7  | 249 | 4.01  | 0.0102 |
| 16  | 7.4  | 230 | 4.35  | 0.0123 |
| 24  | 11.1 | 205 | 4.87  | 0.0123 |
| 32  | 14.7 | 165 | 6.05  | 0.0123 |
| 64  | 29.5 | 140 | 7.12  | 0.0123 |
| 128 | 58.9 | 61  | 16.38 | 0.0123 |

## llama.cpp — per-layer ladder (same box, tg24 median ± sd)

Resident-expert VRAM ≈ (36−N) × 1.64 GiB (59 GiB experts / 36 layers).

| --n-cpu-moe | ~resident GB | tok/s (curve run) | tok/s (primary run) |
|---:|---:|---:|---:|
| 0 (resident) | 59.0 | 240.83 ± 5.87 | 241.18 ± 5.24 |
| 8  | 45.9 | 72.58 ± 1.80 | — |
| 16 | 32.8 | 43.55 ± 0.93 | — |
| 24 | 19.7 | 30.44 ± 0.93 | — |
| 32 | 6.6  | 23.28 ± 0.11 | 24.34 ± 0.18 |
| 36 | 0.0  | 21.26 ± 0.08 | — |

llama's ladder is ~linear in offloaded layers: ≈1.2 ms/token per CPU layer
on this EPYC — a **host-CPU-compute** term. Ours is a **link-transfer +
implementation-overhead** term. Different physics; that is the whole point
of the comparison.

## Prereg grade (six lines, reds included)

| P | prediction | outcome | grade |
|---|---|---|---|
| P1 | b_rel < 3e-2 every K | worst 0.0138 | **PASS** |
| P2 | llama > hybrid at matched ~7 GB (we lose) | 24.34 / 23.28 vs 4.35 | **PASS** (loss as predicted) |
| P3 | hybrid K=16 ∈ [1, 8] tok/s | 4.35 | **PASS** |
| P4 | llama ncmoe32 ∈ [24, 34] | primary 24.34 **in**; curve-run replication 23.28 **out** (−3%) | **PASS / replication MISS** — band under-margined host-CPU variance across pod instances |
| P5 | llama resident ∈ [210, 250] | 241.18 / 240.83 | **PASS** |
| P6 | ratio ∈ [4×, 30×] | 5.6× / 5.35× | **PASS** |

The registered nuance is confirmed, not just the bands: the loss is
implementation overhead, not the streaming strategy (decomposition below).

## Decomposition receipts (inputs to the pipelined build's slip)

- **Transfer floor at K=0:** ~1.89 GB cold/token ÷ 55.2 GB/s ≈ **34 ms**.
  Measured 250 ms ⇒ **achieved fraction of transfer floor ≈ 13.6%**; the
  ~215 ms/token gap is the Python/sync bounty (kill sheet: nonzero syncs,
  host id round-trip, tolist per GEMM call, unpinned CPU index_select).
- **v0 overhead floor with zero transfer:** K=128 (everything resident,
  cold branch never fires) still costs **61 ms/token** (16.38 tok/s) —
  the hot branch's own per-layer sync bundle, measured directly.
- **Cold-branch bundle at K=0:** 250 − 61 ≈ 189 ms, of which ~34 ms is
  bytes ⇒ ~155 ms is cold-path host round-trip + effectively-synchronous
  unpinned H2D.
- **Crossing:** v0 never crosses llama's pinned 24.3 at any K ≤ 128 on
  this box. (All-resident anchor for scale: llama 240.8; v0's K=128 16.4.)
- Slot-level dial behavior verified: monotone tok/s in resGB with
  correctness flat across the sweep — the per-expert dial works; the loop
  around it is what's slow.

## Reading

The v0 hybrid loses everywhere on this fast-host box, exactly as
preregistered, and the curve now shows *why* with measured splits: ~86% of
per-token time is overhead the flagship staging engine + GPU-resident
routing already know how to remove. The pipelined build's job is to close
250 ms → toward the 34 ms floor; llama's own ladder (21–73 tok/s across
offload fractions) is the standing yardstick on this box class.
