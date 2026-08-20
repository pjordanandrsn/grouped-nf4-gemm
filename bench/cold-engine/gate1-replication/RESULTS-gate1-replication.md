# Gate 1 window defect: independent replication, and the diagnosis closed

Receipts: [`gate1_rerun.json`](gate1_rerun.json), [`calib.json`](calib.json),
[`gate1_rerun.log`](gate1_rerun.log). Box destroyed after the pull.

**gnf4#137 already corrected gate 1's read counts** on a box matched to the
original (EPYC 9655, 394 GB/s triad). This run was executing at the same time
on different hardware and is **not** a second correction — #137's numbers
stand, measured on the better-matched host. Two things here are additive:

1. **The diagnosis is now closed by measurement, not asserted.** A second
   snapshot at model load shows the published counts are *reproduced* by
   `reads_since_load`.
2. **A cross-box replication** of the corrected counts on deliberately
   different silicon.

## Host — deliberately unmatched

RTX 5090 · **AMD EPYC 7B12 (Zen 2)** · driver 580.159.03 · torch 2.13.0+cu130
· `numactl --interleave=all` (2 NUMA nodes). DRAM triad **148.5 GB/s** and
NVMe seq **2.42 GB/s**, against the original's 380.1 and 5.51 and #137's 394
and 3.19. Same model, arena geometry (L=16, E=64, row 3,538,944 B) and config
(`steps=128`, `warmup=8`, `seq=64`, `hot_rows=384`, `vram_frac=0.25`,
`order=tail`, `source=dram`).

This box is a poor match for wall-clock work, which is why **no wall number
from it is used anywhere below**. It is a fine box for counting reads, because
a read count is a property of the routing trace and the placement.

## 1. The published counts reproduce as `reads_since_load`

`run_gate1.py` now snapshots `cold_stats` twice — at model load and at the
measurement boundary. The load snapshot is exactly the quantity the pre-#132
harness was unknowingly differencing against.

| cold | arm | **published "win reads"** | `reads_since_load` here | agreement |
|---|---|---|---|---|
| 1% | cold-GPU | 105 | 112 | +7% |
| 1% | cold-CPU | 106 | 113 | +7% |
| 5% | cold-GPU | 238 | 241 | +1% |
| 5% | cold-CPU | 238 | 249 | +5% |
| 10% | cold-GPU | 340 | 335 | −1% |
| 10% | cold-CPU | 335 | 347 | +4% |
| 20% | cold-GPU | 3400 | 3437 | +1% |
| 20% | cold-CPU | 2025 | 1597 | −21% |

Seven of eight within 7%, on hardware that shares nothing with the original
but the model. The published figures are not *approximately*
warmup-inclusive — they **are** the since-load counts.

That distinction matters. "Those numbers counted warmup" was, until this run,
an inference from a code path. It is now a measurement with a residual, and
the residual is small.

## 2. Cross-box replication of the corrected counts

| cold | arm | #137 (EPYC 9655) | here (EPYC 7B12) |
|---|---|---|---|
| 1% | cold-GPU | 19 | 25 |
| 1% | cold-CPU | 19 | 26 |
| 5% | cold-GPU | 28 | 27 |
| 5% | cold-CPU | 26 | 36 |
| 10% | cold-GPU | 28 | 26 |
| 10% | cold-CPU | 29 | 38 |
| 20% | cold-GPU | 237 | 255 |
| 20% | cold-CPU | 301 | 536 |

The GPU-destination arm replicates tightly (19/25, 28/27, 28/26, 237/255).
**The CPU-destination arm does not**, and drifts with cold mass: +38% at 5%,
+31% at 10%, +78% at 20%.

That asymmetry is not noise and is worth stating plainly. Cold-CPU execution
holds tier rows for the duration of a CPU-side matmul, so on a box with 2.7×
less DRAM bandwidth the residency window is longer and eviction order differs
— the count stops being purely a property of the trace once the consumer is
slow enough to change what is resident. **Read counts transfer across boxes
for the GPU path; for the CPU path they transfer only in order of magnitude.**

Applying either column to the original box's wall deltas gives the same
conclusion — storage at low single-digit percent of cold cost — so this
does not disturb #137's finding. It does mean a corrected cold-CPU read count
is a per-box quantity, not a universal one.

## What this does not claim

- **Not a correction to gate 1.** #137 holds; it was measured on the matched
  host. Nothing here supersedes it.
- **No wall-clock claim.** Δ ms/step on this box runs 24–42× the published
  values (699 ms at 10% cold-GPU against 16.66 ms), because Zen 2 is far
  slower at the per-call software cost that dominates the cold path. Gate 1's
  clauses were **not** re-scored here.
- **The device row cache was not exercised.** It is wired into
  `Mxfp4NvmeResidency`; gate 1 runs the NF4 hybrid tier. That measurement
  needs an MXFP4 model and remains open.
