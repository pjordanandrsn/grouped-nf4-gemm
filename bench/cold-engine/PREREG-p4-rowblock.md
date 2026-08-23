# PREREG — P4: row-chunk-outer blocking kills the batch rows-penalty

Registered before any measurement. P4 of the e4b objective revision
(`experts4bit-qlora/docs/hybrid/OBJECTIVE-REVISION-2026-08-23.md`),
confirmed as the B=16 lever by the offline P1/P2 scoring there
(`bench/hybrid-g9/p1p2offline/`): the CPU grouped GEMV's column-outer
order re-streams the whole activation block through L3 once per column —
`N × rows × K × 4` bytes per call (~1.6 GB at the B=16 serving shape)
against 76 MB of weight reads — and that traffic is the measured
rows-penalty (achieved weight-rate 124.3 → 83.5 GB/s when rows double at
near-equal weight bytes) and a large share of the 3.5–3.8× in-executor
loss.

## The change (this PR)

`gnf4_native/cpu_kernels.c`, both grouped range functions: row-chunk
OUTER, columns inner, within the existing 32-column tile. Activation
chunks (~64 KB) are read once per tile instead of once per column; the
tile's weights (~32 KB) stay L2-resident across row chunks. Only the
(row, column) visit order changes; each output element's k-descent is the
locked tree unchanged — bit-exactness is a test
(`test_batch_rows_bit_exact_after_row_outer_blocking`, multi-chunk ×
multi-tile shapes, both formats), not an argument.

## Registered claims, one box, matched shapes (e4b law 5)

[p4_rowscale.py](p4_rowscale.py): one grouped NF4 call, Qwen3-30B-class
geometry (N = 768, K = 2048, 29 uniques), rows ∈ {64, 128}, medians of
50, 32 threads. Old arm = the parent commit's binary; new arm = this
commit's; same box, same harness, back-to-back.

* **C1**: new/old achieved GB/s at rows = 128 ≥ **1.25×**.
* **C2**: rows = 64 unchanged within **±10%** (the swap must not tax the
  B=8 regime that already meets its bar).
* **C3**: new rows-scaling ratio (achieved₁₂₈ / achieved₆₄) ≥ **0.85**
  (b16close measured 0.67 — the penalty must mostly close).
* **Correctness voids walls**: the bit-exact test must pass on the box
  before any timing is read.

Box gates: AVX-512, triad ≥ 100 GB/s, the burst-clock gate is NOT
required (no GPU in the measurement); n\* is NOT required (no promotion
economics in the measurement) — this is a CPU-only kernel A/B.

**Hard stop**: one box, one A/B. REFUTED at any clause ⇒ the L3-restream
theory is wrong or incomplete; the finding is reported and the next
kernel hypothesis is registered separately. The e4b end-to-end G8 B=16
re-run happens only after a PASS here, as its own registration.
