# `bench/calibrate.py` at the #402 head, on the NAS RTX A2000 (2026-09-24) — a rehearsal receipt, not a reading

The first schema-`gnf4-hybrid-calib/2` blob: GPU benches only (`--skip-cpu`), the shared RTX A2000 12 GB (sm_86,
PCIe gen 3 x8, host Xeon W-1250, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, torch 2.8.0+cu128), run under
the A2000 lock. `job.log.txt` is the container's output; `calib.json` the blob.

What it shows: the new single-copy probe reads **5.67 GB/s** against the 40-deep back-to-back **5.46 GB/s**, so
`Costs.from_blob` derives `link_eff` = 1.0 (capped) — on a gen 3 x8 link the two rates agree, exactly as
experts4bit-qlora lane P66's rehearsal found (6.14 vs 6.12 there; the shared card's clock state moves the absolute).
The gen 4 x16 figure (P66: 14.72 single vs 23.07 back-to-back on an RTX 5090, link_eff 0.64) is the registered probe's
to measure with this script; this blob is the /2 round-trip and the gen-3 control, never quoted as a 5090 number.
`kernel/test_cold_deadline.py::test_the_committed_a2000_blob_reads_as_schema_2` loads it.
