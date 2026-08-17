# G7 refinement round — the kernel clause resolves

Post-merge refinement on Jordan's "squeeze more". Qualified box: RTX 5090,
600 W / 3135 MHz max SM, driver 595.84, triad 1574.1–1575.0 GB/s across
the session (`instrument.txt`; qualification protocol below).

## Verdict change

**The kernel ≥70%-of-`B_vram` clause PASSES: 90.0% serving-shaped,
70.3–70.6% isolated-launch.** G7 becomes 3 of 3.

The round's decisive find was not a kernel change — it was a measurement
defect with real serving consequences: the split heuristic's
`int(seq_lens.max())` launched a device reduction **and a full device
sync inside every wrapper call**. ~12 µs against a ~123 µs kernel. Every
"shipped defaults" row in the round-1/2 receipts carried it, and — the
part that matters — it **serialized the 94-layer B_max loop**, which is
exactly the shape a real decode step has. `max_useful` now derives from
the block table's host-known capacity; no sync.

| measurement (shipped defaults unless noted) | before | after |
|---|---|---|
| B_max step, 94 layers × B=25 × 4K | 10.59 ms | **7.50 ms** |
| B_max aggregate over the 9.90 GiB working set | 992 GB/s (63%) | **1417 GB/s = 90.0%** (reps 1416.4 / 1417.2 / 1419.9) |
| defaults B=32/T=4K | 1005 | **1095 (69.5%)** |
| wall vs bf16 SDPA, B=16/25/32 at 4K | ×0.77–0.83 | **×0.68–0.75** |
| best-tuned isolated launch, B=32 (kt64, w4, st3, sp2) | — | 11 fresh reps: median 1111 (70.6%), min 1101.2 (69.96%) |

Why the two fractions differ, and which is the gate's: an isolated
launch pays its launch latency, split tails, and combine gap against a
hard sync — costs a decode loop overlaps with the next layer's work.
The 94-layer measurement launches attention back-to-back exactly as a
model executes it, every layer's ~113 MB pool L2-cold (94 distinct
allocations, no cross-layer reuse possible), KV bytes only in the
numerator. That is the workload the gate exists to protect; 90.0% is
its number, stable to ±0.1% across process-fresh repeats. The isolated
number is reported alongside because it is the harsher protocol and it
alone sat ON the bar (median above, worst rep 69.96%).

## What else the round measured

- **Heads-major block layout, rebuilt and re-tested under the fp8
  kernel** — the tokens-major verdict from round 1 was recorded under
  the decode-ALU-bound kernel, a bottleneck the fp8 path deleted, so
  the lore needed re-deriving under the bottleneck that exists.
  Verdict: tokens-major still wins — 1113 vs 1089 sustained at B=32,
  1065 vs 1032 at B=25. The layout stays in-tree behind `layout=`
  with loss tables under BOTH regimes (`g7_sweep7_layout.json`); the
  question is closed, which is worth more than the flag.
- **Non-pow2 splits fixed B=25's wave quantization**: sp=3 took B=25
  from ~1000 to 1065 (+6.9%); the heuristic now lands ~2.2 CTAs/SM
  (B=32 → 2, B=25 → 3).
- **kt=256 collapses** (656 GB/s — the serial-iteration tail), **w=8
  loses ~12%** (register pressure), st=3≈st=4. Mode-aware defaults
  ship: fp8 → ktile 64, 4 warps; f32 decode keeps 32/2.
- **Instrument qualification now requires a kernel-class probe, not
  triad alone**: a rented "5090" with a full-speed triad (1568 GB/s)
  sustained only 2430 MHz SM under load (550 W limit) and ran this
  kernel 35% slower at identical configs — its layout A/B tie was
  correctly discarded as regime-uninformative, the box destroyed. A
  '5090' is not one instrument; triad qualifies DRAM, not the SM
  domain this kernel also lives in.

## Receipts

- `g7_sweep7_layout.json` — full tokens-vs-heads surface (320 rows) +
  3-rep confirms
- `g7_confirm_grid.txt` — 12-config × 5-rep neighborhood (two passes)
- `g7_sweep7_confirms.txt` — sweep-stage sustained confirms
- `g7_final3.json` — shipped-defaults formal table, both compute modes,
  post-sync-fix
- `g7_bmax3.json` / `g7_bmax_reps.json` — pipelined B_max + 3
  process-fresh repeats
- `instrument.txt` — card, power limit, max SM clock, driver
