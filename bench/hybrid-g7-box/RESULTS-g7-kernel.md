# G7 kernel + B_max receipts — quiet calibrated box (RTX 5090)

Formal measurement of Phase 7's two performance clauses on an uncontended
RTX 5090 (sm_120, 32 GB), torch 2.8.0+cu128, calibration blob
`calib.json` in this directory (`--tag g7box`). The quality clause's
receipts live in experts4bit-qlora `bench/hybrid-g7/RESULTS-g7-quality.md`.

**Calibrated ceiling: `B_vram` = 1573.4 GB/s** (fp32 STREAM triad,
`calibrate.py --skip-cpu`). The honest comparator arm, bf16 SDPA with
`enable_gqa=True` on contiguous tensors, achieves 1447 GB/s (92% of
triad) at B=25/T=4096 — the headroom the gate points at is real on this
card.

## Verdicts

| clause | result | number |
|---|---|---|
| kernel ≥70% of measured `B_vram` | **MISS** | 52.9% best-tuned (832 GB/s), 49.5% shipped defaults (778 GB/s) |
| `B_max` ≥25 at 4K on the 235B-class geometry | **PASS** | 94 layers × B=25 × 4K resident in **9.90 GiB**, attention over all of it 13.5 ms/step, 19.7 GiB VRAM left |

With quality (PASS on `fp8_kg32`, both probe models, both metrics), G7
lands **2 of 3 clauses**. The miss is reported as a miss; nothing below
argues it away.

## What the bytes buy (the framing that matters for serving)

At the serving shapes the FP8 kernel reaches **wall-clock parity with
bf16 SDPA while reading half the bytes**: ×1.05 at B=25/T=4096, ×0.99 at
B=32/T=4096. The byte halving is therefore realized as *capacity* — the
B_max table below — at zero latency cost at batch. It is NOT yet realized
as *speed*, which is exactly what the 70% clause measures, and why it
misses.

## Shipped-defaults table (`g7_kernel_final.json`)

Defaults: ktile 32, 2 warps, 3 stages, ~8-CTAs/SM split heuristic —
no hand-picked config; shape (H_q 64, H_kv 4, D 128), k_groups 4.

| B | T | kernel GB/s | frac B_vram | wall vs SDPA |
|---|---|---|---|---|
| 1 | 4096 | 75.0 | 0.048 | ×2.71 |
| 8 | 4096 | 406.4 | 0.258 | ×1.78 |
| 16 | 4096 | 612.7 | 0.389 | ×1.19 |
| 25 | 4096 | 739.1 | 0.470 | ×1.05 |
| 32 | 4096 | 778.2 | 0.495 | ×0.99 |

(1024-token rows in the JSON; same shape, lower fractions.)

## How the kernel got here, and where the remaining gap is

Design iterations, each measured (dev card A2000 for direction, this box
for decisions — the shared A2000 shows 40% run-to-run variance at
identical configs and cannot discriminate fine configs):

1. (B, H_kv) grid: 4 CTAs at batch 1 — occupancy-starved, 36.6 GB/s.
2. Split-K + combine, bit-assembly E4M3 decode: the box's first sweep
   found 711 GB/s but capped splits at 4.
3. **Split axis extended** (`g7_sweep2.json`): the B=1..8 wall time was
   *flat* (~108 µs) — a latency pedestal from each CTA's serial tile
   loop, not a bandwidth wall. splits=8..16 clears most of it: 806 GB/s
   at B=25 (32,2,3,8), 832 at B=32 (32,2,3,16), B=1 2.5× to 103 GB/s.
   Marginal bytes above B=16 stream at ~940 GB/s.
4. **Head-packed variant** (`g7_sweep3_packed.json`): one CTA per
   (sequence, split) reading every 512 B token line whole, block-diagonal
   tensor-core scores. Built, tested (24/24 both paths), swept — and
   LOST everywhere: 574 vs 806 at B=25, 66.8 vs 103.5 at B=1. With a
   96 MB L2 over a ~106 MB pool the scattered sibling-quarter reads of
   the split kernel mostly hit L2 anyway, and packing pays 4× dot work
   plus one heavyweight CTA's register pressure. Kept behind
   `pack_heads=` (launch-time choice, same format) for cards where the
   L2:pool ratio inverts the trade.

Where the remaining ~45% goes, from the surface: ~10% is split-K partial
traffic (m/l/acc write + combine read — inherent to high split counts);
the rest is the serial per-CTA tile walk through a gather (block-table
indirection → payload → scales) that pipelines worse than SDPA's
contiguous streams, which is also why the gap closes as B grows and the
grid widens. The next lever that could plausibly reach the bar is a
persistent-CTA schedule with software-pipelined table prefetch — a
redesign, not a tuning pass; costed in the phase writeup rather than
attempted past the gate deadline.

## B_max at the 235B-class geometry (`g7_bmax.json`)

94 layers × B=25 × T=4096, `fp8_kg32` (grouped-32 key scales — the
quality-passing format), all pools resident, per-layer paged decode over
the whole set with shipped defaults:

- KV bytes: **10,626,662,400 (9.90 GiB)** — the bf16 equivalent is
  19.3 GiB; allocator-measured VRAM delta 9.88 GiB (pool rows are the
  only allocation).
- Attention step over all 94 layers: **13.49 ms** (143.5 µs/layer),
  aggregate 788 GB/s.
- VRAM free after: **19.7 GiB** on the 32 GB card — the headroom the
  hybrid tier spends on hot experts and activations.
- Pool content is one packed layer cloned per layer: every layer is a
  distinct real allocation the kernel really reads; identical content
  changes nothing (different addresses, and one layer's ~113 MB exceeds
  L2, so cycling 94 layers leaves no cross-layer reuse).

`B_max` here is the KV-capacity + attention half of the clause — batched
*expert* dispatch does not exist until Phase 8/9, so a full-model batched
decode cannot be measured yet. The clause's arithmetic closes: at
9.9 GiB KV + 19.7 GiB free, batch is not the binding constraint at 25
sequences; the expert path is, which is Phase 8's gate.

## Round 2 — the redesign pass ("hold and go")

The 52.9% surface carried its own diagnosis: stages flat (not
latency-bound), 2 warps beating 4, ktile 32 beating 64 by 65% — per-SM
**issue throughput**, i.e. the ~40 integer ops per payload element of
E4M3 bit-assembly decode + scale expansion. `compute="fp8"` deletes that
work instead of scheduling around it: payload bytes BITCAST into fp8
tensor-core dots, scales folded onto post-dot `[G, KTILE]` score tiles
(grouped keys = one sub-dot per group, exact because a group's scale is
constant across the reduction axis; V scales fold into P under a
per-tile range factor riding the online-softmax rescale; q scaled once
into e4m3's range, divided out of `sm_scale`).

Second box, same card class, own calibration blob (`round2/calib.json`,
B_vram **1574.3** — matching round 1's 1573.4; warm-state triad
re-verified identical, so no thermal correction applies).

| number (B=32/T=4K unless noted) | GB/s | frac B_vram |
|---|---|---|
| f32-decode path (round 1 best) | 832 | 52.9% |
| **fp8-compute, best-tuned sustained** | **1087–1095** | **69.1–69.6%** |
| fp8-compute, best-tuned, B=25 | ~1000 | 63.6% |
| fp8-compute, shipped defaults | 1004.7 | 63.8% |
| single-shot transient (sweep 4, unreproduced) | 1141.5 | (72.6%) |

**Gate verdict unchanged in letter: MISS — sustained best 69.4% vs the
70% bar, from 52.9%.** The 1141.5 (72.6%) single measurement did NOT
reproduce across eight re-runs on both kernel versions with the triad
stable; it is recorded as a transient and not quoted as the gate number.
The bar is missed by ~0.6 points on a reproducible protocol (median of
repeated 30–50-iter runs, fresh allocations, triad re-verified).

What DID change decisively is the serving consequence:

- **wall vs bf16 SDPA at serving shapes: ×0.77–0.83** — the FP8 kernel
  is now 17–23% FASTER than bf16 SDPA while reading half the bytes
  (round 1: ×0.99–1.05). Byte halving is now realized as capacity AND
  speed, which is what the 70% clause was a proxy for.
- **B_max full-model step (94 layers, B=25, 4K, shipped defaults):
  10.59 ms** (round 1: 13.49), aggregate **1003 GB/s** over the whole
  9.90 GiB working set with every layer L2-cold.
- fp8-compute numerics vs the f32 oracle (K/V bytes identical; q and p
  each pay one e4m3 rounding): probed EXACT at T=1, worst-element 0.087
  at T=2 (one dominant token), serving-shape distribution mean 0.005 /
  p99 0.020 / max 0.152. Documented serving tolerance (invariant 4′);
  model-level quality of fp8 COMPUTE is explicitly owed if it becomes
  the default — the G7 oracle certified storage, and `compute="f32"`
  (52.9%-class, decode-exact) remains the default pending that run.

Also measured and closed this round:

- **Packed + fp8** (`_fp8_paged_decode_packed_f8`, whole-line loads
  without the f32-decode register pressure): 23/23 correct, LOSES 609
  vs 1090 at B=32 — with the decode gone the one-block tile is
  iteration-bound on a small grid, and line utilization never pays past
  what the 96 MB L2 already provides. Kept default-off, loss recorded
  (`round2/g7_sweep6_pf8.json`).
- **Fused last-CTA combine** (stream-k arrival counter, self-cleaning
  slots): worth ~1% at the winning split counts (sp=2–4 leaves little
  to fuse) — but building it exposed that per-call `torch.zeros` for
  the counters cost ~4% (a memset launch per call); counters are now
  cached per (device, size) and reset by the combining CTA itself.
- Cache-policy hints: ptxas rejects `.cg`+`evict_first`; `evict_first`
  alone measured ~nothing. The kernel is at the memory system's
  effective ceiling for this access pattern on this card.

Remaining headroom analysis: at 1090 counted GB/s the kernel moves
~103.5% of counted bytes (split partials + q reloads + tables), i.e.
~1128 GB/s of actual traffic = 71.7% of triad, against SDPA's 92% on
contiguous bf16 with zero indirection. The residual is the paged
gather's DRAM-page behavior itself; the two structural alternatives
that could in principle beat it (head packing, block-tile fusion) were
both built and both measured slower. This design is done on this card.

Round-2 receipts in `round2/`: own calib blob, f8dot sweep
(`g7_sweep4_f8dot.json` — contains the unreproduced 1141 row), fused
sweep (`g7_sweep5_fused.json` — polluted by the per-call counter
memset, superseded), pf8 sweep, shipped-defaults formal table
(`g7_kernel_final2.json`, both compute modes), fp8-defaults B_max
(`g7_bmax.json`).

## Raw receipts

- `calib.json` — calibration blob (GPU lanes; `--skip-cpu`)
- `g7_sweep.json` / `g7_sweep2.json` — split-kernel config surface
- `g7_sweep3_packed.json` — packed-variant surface
- `g7_kernel_bench.json` — sweep-1-best formal table (superseded)
- `g7_kernel_final.json` — shipped-defaults formal table
- `g7_bmax.json` — B_max run
- `env.txt` / `smi.txt` — torch/driver/card identification
