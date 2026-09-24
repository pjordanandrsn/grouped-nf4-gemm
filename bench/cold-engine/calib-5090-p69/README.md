# `bench/calibrate.py` (schema /2) on a rented RTX 5090 — lane P69's receipt (2026-09-24)

The kernel package's own calibration script at `54ac2a44` (#402; the fetched file's sha256 is in `calibrate.sha256`
and equals this tree's), run **twice back to back** with `--skip-cpu` on one Vast verified/secure RTX 5090 (driver
580.95.05, 600 W; host AMD EPYC 7663, 224 CPUs, 2 NUMA nodes; PCIe gen 4 × 16, `nvidia-smi` reading gen 1 *current*
at the idle pre-run read; image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, torch 2.8.0+cu128). Registered before
the rental as experts4bit-qlora `bench/p69/P69-PREREG.md` (#741); the drive script and its exit codes are
experts4bit-qlora `bench/p69/p69_drive.sh`. Run `p69-5090-2`, $0.0146, 93 s, teardown proven (`teardown-proof.json`;
the launcher's `receipt.json` is in the private store, `receipts/experts4bit-qlora/2026-09-24/p69-5090-2/`).

| run | `h2d_64mb` (40 back-to-back, GB/s) | `h2d_64mb_single` (one synchronized copy, GB/s) | `link_eff` = single / back-to-back |
|---|---|---|---|
| 1 | 20.58 | 17.97 | **0.873** |
| 2 | 20.72 | 19.90 | **0.960** |
| spread | 0.7 % | 9.7 % | 9.1 % |

**Against the registration.** P2 (back-to-back 20–30 GB/s) **HELD**. P4 (the two runs within 10 %) **HELD**, at 9.7 %
on the single-copy figure. **P1 REFUTED:** `link_eff` was registered at [0.55, 0.75] from lane P66's census probe
on another RTX 5090 (14.72 / 23.07 = 0.638, host AMD EPYC 7C13); this host reads 0.873 / 0.960. P3 (single 12–18)
refuted on run 2 (19.90). The decision rule's refuted branch applies: the difference is stated as a difference
between two gen 4 × 16 hosts — 0.64 on one, 0.87–0.96 on the other, with the back-to-back rate itself 23.07 there and
20.6 here — and is not explained. Candidates the registration named (the link's idle state, NUMA placement of the
pinned buffer, the DMA engine's behaviour) were not measured.

**What it means for `cold_deadline`.** `link_eff` is a **per-host** measurement, not a constant of the 5090 class,
which is exactly why #402 reads it from the box's own blob rather than fixing it. `Costs.from_blob` on `run1/calib.json`
gives 0.873 (`kernel/test_cold_deadline.py::test_the_committed_5090_blob_reads_as_schema_2`). No default moves.
