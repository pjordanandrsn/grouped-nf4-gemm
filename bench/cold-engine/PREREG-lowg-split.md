# PREREG — the low-G item split: un-starving small-G calls

Registered before any measurement. The interaction hunt
([RESULTS-interaction-hunt.md](RESULTS-interaction-hunt.md)) measured the
finding this fixes: the (group × column-tile) work grid starves the pool
when `G × tiles < threads` — a G=1, rows=128 call ran **slower than a
G=64 call carrying 64× the weight bytes** (820.7 vs 733.7 µs), because 24
serial items cannot feed 32 threads.

## The change (this PR)

`gnf4_native/cpu_kernels.c`, both grouped GEMV ranges: when — and only
when — `G × tiles_n < threads`, the item grid gains a row-slice dimension
(`rsplit`, power of two ≤ 64), each slice aligned to whole
`NF4_CELL_ROWS` cells. Splitting a group's rows re-reads its weights per
slice — the Phase-8 amortization warning — which is the right trade only
while threads sit idle, so **well-fed calls keep the Phase-8 single-item
path verbatim (`rsplit = 1`)**: every previously measured configuration
is bit-for-bit and instruction-for-instruction unchanged. Slices
partition rows; each output element is computed by exactly one item with
its k-descent untouched — the retiling bit-exact guard covers the split
path on-box (its G=2 × 3-tile shape arms the split at 32 threads).

## Registered A/B ([interaction_hunt.py](interaction_hunt.py), the committed grid, A/B/A, staged harness, clean rebuilds)

* **B1 (the win)**: new/old median ≤ **0.55** at (G=1, rows=128, N=256)
  and (G=1, rows=128, N=768) — the starved cells must at least ~1.8×.
* **B2 (no harm)**: every grid cell with `G × tiles ≥ 32` and both
  G = 29 serving shapes within **±10%** of old (the guard predicts ~0%).
* **B3**: the multi-row bit-exact test passes on-box before any timing
  (correctness voids walls).

Box gates: AVX-512, triad ≥ 100 GB/s, ≥ 32 effective cores (the
starvation threshold under test presumes them). **Hard stop**: one box,
one A/B/A. Refuted at any bar ⇒ the change is reverted (the P4
precedent: no unproven churn in the hottest kernel) and the finding
reported.
