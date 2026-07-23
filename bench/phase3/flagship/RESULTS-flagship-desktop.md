# Flagship desktop-cell replication — the projected gen4-×16 row, measured (virtualized host; bare-metal + energy run pending)

**MEASURED (replication) tier.** Receipts in `receipts-desktop-20260721/`
(driver copy included). **Host #8** of the flagship series: RunPod SECURE
virtualized pod `c52i94rmuewpzc`, NVIDIA **L40S 46 GB** (driver 550.127.05),
AMD EPYC 9354 host, 650 GB container disk, triton 3.4.0 / bnb 0.49.2, run at
`1a60c39`, torn down after receipts (404-verified). This fills the tier
table's one **projected** row — "235B on a gen4-×16 desktop, ≈3 tok/s by the
waterfall arithmetic" — on a rented box of the same link class. It is **not**
the bare-metal confirmation: that run (with RAPL/GPU energy per token) has
armed lanes waiting on stock, gets its own doc, and only then does the
README/site tier row flip.

## The box class, verified before anything else

Measured pinned H2D: **26.98 GB/s** — inside the gen4-×16 band [18, 32].
(`nvidia-smi` reads link gen 1 ×16 at idle — power-state downclock; the
bandwidth measurement is the ground truth.) Property suite on this stack:
**44/44** (`suite.txt` — the DO-replication triton anomaly does not recur;
triton 3.4.0 has `tl.gather`).

Waterfall for the real checkpoint: 7.984 GB/token ÷ 26.97 GB/s =
**3.38 tok/s** ceiling.

## Phase A — synthetic-weight stream (64 tokens)

| mode | tok/s | per-mode ceiling | fraction |
|---|---|---|---|
| none (pure stream) | 3.296 | 3.378 | **97.6%** |
| fused | 3.269 | 3.228 | **101.3%** |

The fused row's own link microbench read 25.77 GB/s (vs 26.97 for `none`),
so its per-mode ceiling is lower; landing 1.3% above it is microbench-vs-
sustained-stream jitter, not free performance. Both modes sit in the
**projected 3.0–3.5 band**, i.e. the row's waterfall arithmetic holds on
this box class.

## Phase B — real Qwen3-235B-A22B-Instruct-2507, coherent decode

438 GB bf16 pulled to **disk cache** (`--cache /root/hf`, the
desktop-realistic configuration — a desktop does not hold the checkpoint in
tmpfs; the stamped phaseB5 run used /dev/shm), stream-quantized to NF4 in
pinned host RAM, decoded through the fused kernel:

| prompt | off | gpu | gpu_early | paired gpu | paired gpu_early | hit (positional) | distinct-2 |
|---|---|---|---|---|---|---|---|
| MoE-vs-dense | 2.603 | 2.628 | 2.191 | ×1.009 | ×0.842 | 0.780 | 0.87 |
| haiku | 2.609 | 2.631 | 2.196 | ×1.008 | ×0.842 | 0.790 | 1.00 |
| quantization | 2.607 | 2.630 | 2.204 | ×1.009 | ×0.845 | 0.793 | 0.98 |

Greedy identity: **6/6 byte-identical** across arms. VRAM peak **15.14 GB**.

## What replicates (against the stamped flagship runs)

- **The off-mode waterfall fraction is 0.771–0.772×** — the same **0.771×**
  the stamped phaseB5 H100 run measured at 45 GB/s
  (`RESULTS-flagship-phaseB5.md`). The fraction is the box-invariant; the
  link sets the absolute number: 2.60–2.61 tok/s here.
- **`gpu` mode is again the best configuration** (×1.008–1.009 paired), and
  **speculation again loses**: gpu_early ×0.842–0.845 at hit 0.78–0.79.
  The wire law 1/(2−H) predicts ×0.820–0.829 from these hits; observed runs
  ~2% above prediction — the same magnitude and direction as phaseB5
  (predicted 0.826–0.833, observed 0.838–0.851). No prefetch win exists in
  this regime; run `gpu`, not `gpu_early`.
- **VRAM peak 15.14 GB** matches every flagship host (15.1–15.2 GB) — the
  cell's "fits a 16 GB desktop card" premise survives the box-class change.
- **Disk-cache quantization is a non-event for decode** — once weights are
  quantized into pinned RAM, decode streams from RAM; the cache tier only
  paces the one-time load.

## What this doc does not claim

- Not bare metal: virtualized pod, no RAPL, no dmidecode chain — **no energy
  numbers here**. The bare-metal + J/token run (Vultr 8×A100 or Latitude
  H100, lanes armed) is the confirmation gate for the tier-table flip.
- The 2.60–2.61 real-decode figure is not the projected "≈3 tok/s" — that
  band is the waterfall/synthetic arithmetic (Phase A lands in it); real
  decode carries the universal 0.77× serialization fraction, exactly as on
  every stamped host. The row should be read as: ceiling ≈3.4 measured,
  real decode 2.6, fraction 0.77 — no surprises in either direction.

## Receipts

`receipts-desktop-20260721/`: `link.txt`, `suite.{txt,log}`,
`flagA_none.json`, `flagA_fused.json`, `flagB_real.json`, `gpu.txt`,
`cpu.txt`, `ram.txt`, `disk.txt`, `versions.txt`, `gnf4_sha.txt`,
`SUMMARY.txt`, `driver-flag-desktop-run.sh` (the on-pod driver, verbatim).
Checksums: `SHA256SUMS.desktop`.

## Driver notes (review disclosures — the driver is preserved as run, not repaired)

Three properties of the verbatim driver, flagged in review and disclosed
here rather than edited away (editing the driver copy would falsify the
receipt):

1. **The clone is unpinned** (`git clone --depth 1` at floating HEAD); the
   commit actually exercised is recorded in `gnf4_sha.txt` (`1a60c39`). To
   re-run against the same harness code: `git clone … && git checkout
   1a60c39` (a full clone, or `--depth 1` + `fetch --depth 1 origin
   1a60c39`). Future drivers pin at clone time.
2. **`AB-DONE` is an evidence-collection marker, not a success gate** — the
   driver is continue-on-fail by design (receipts of failure are still
   receipts). Ground truth is the per-phase artifacts: `suite.txt` carries
   the pytest rc, and each `flag*.json` exists only if its phase completed.
   All are present and parse here.
3. **`SUMMARY.txt` is cosmetically broken**: the driver's summary step looks
   up `tok_per_s`/`tokens_per_s`/`decode_toks` but the harness emits
   `toks_per_s`, so it dumped raw dicts instead of one-line rates. Every
   number in this doc reads from the `flag*.json` receipts directly;
   `SUMMARY.txt` is retained only as the as-pulled artifact.

## Addendum 1 (2026-07-22) — the fraction is NOT the box-invariant; additive law supersedes

**Forward-only correction.** The sections above are stamped and untouched
(pre-addendum proof: `RESULTS-flagship-desktop.md.pre-addendum1.ots`); two of
their claims are superseded by a measurement:

- *"The fraction is the box-invariant; the link sets the absolute number"*
  (What-replicates §1) — **retired.**
- *"real decode carries the universal 0.77× serialization fraction, exactly as
  on every stamped host"* (What-this-doc-does-not-claim §2) — **retired.**

The gen5 bare-metal run ([`RESULTS-flagship-gen5-metal.md`](RESULTS-flagship-gen5-metal.md),
Latitude H100 PCIe, measured 56.69 GB/s link) achieved **3.924 tok/s = 0.553× of
its waterfall** with a correct, greedy-identical kernel. The 0.77 agreement
between this doc and phaseB5 was a coincidence of two hosts whose per-box
overheads happened to scale with their links — and the wider record (phaseB3
0.628, phaseB4 0.671, DO H200 0.639) never supported a constant. The standing
model is the additive form registered in `PROJECTIONS-multiarch.md` Addendum 1:

```
t_token ≈ c_box + cold_bytes / L
```

with `c_box` measured per box (this host: **87.4 ms**; seven-host range
**53.5–114.0 ms**, not ordered by link speed — full receipt-derived table in the
gen5 doc). The fraction is a *derived output* that falls as the link gets
faster. This doc's own numbers are unaffected (they are measurements, not the
law); note 3's "cosmetically broken SUMMARY.txt" typo class became load-bearing
on the gen5 run (a `None`-printing summary under an evidence-complete banner)
and is retired by the committed schema-aware reducer
([`ab_reduce.py`](ab_reduce.py), CELL-VOID rule).
