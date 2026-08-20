# R2, scored — resurrections reach 15.7% of routed work and move wall by nothing

Registered (`PREREG-tribrid-stage3`, R2):

> VRAM resurrection is disproportionately valuable — **even 2–5% of routed
> invocations moves wall time** — **refuted by no measurable wall effect**.

**REFUTED.** Rates of **5.37% to 15.72%** were measured — every
nonzero point above the band R2 conditions on — and wall time is explained by
row transfers with nothing left over for resurrection.

First run of `Mxfp4NvmeResidency` on a real model rather than a fixture.
Arena: **openai/gpt-oss-20b**, baked with `nvme_arena.bake` — 24 layers × 32
experts, 13.22 MB per expert row, 10.15 GB. One layer, decode-shaped (T=1),
256 timed steps after 16 warmup, 2 repeats per point, RTX 5090. Receipts in
`r2-wall-2026-08-20/`.

## The sweep

| rows | prot | wall (ms) | resurrections | per routed | transfers | cache hits |
|---|---|---|---|---|---|---|
| 48 | 36 | 0.385 | 0 | 0.00% | 5 | 1019 |
| 48 | 44 | 0.378 | 0 | 0.00% | 5 | 1019 |
| 32 | 28 | 0.376 | 55 | 5.37% | 61 | 908 |
| 32 | 24 | 0.391 | 105 | 10.25% | 144 | 775 |
| 48 | 24 | 0.402 | 105 | 10.25% | 144 | 775 |
| 24 | 20 | 0.872 | 86 | 8.40% | 274 | 664 |
| 24 | 18 | 0.875 | 98 | 9.57% | 314 | 612 |
| 32 | 16 | 0.883 | 102 | 9.96% | 363 | 559 |
| 16 | 12 | 1.344 | 128 | 12.50% | 466 | 430 |
| 24 | 12 | 1.347 | 128 | 12.50% | 466 | 430 |
| 48 | 12 | 1.351 | 128 | 12.50% | 466 | 430 |
| 12 | 8 | 1.355 | 133 | 12.99% | 597 | 294 |
| 16 | 8 | 1.355 | 133 | 12.99% | 597 | 294 |
| 32 | 8 | 1.360 | 133 | 12.99% | 597 | 294 |
| 12 | 6 | 1.819 | 149 | 14.55% | 672 | 203 |
| 24 | 6 | 1.821 | 149 | 14.55% | 672 | 203 |
| 12 | 3 | 1.822 | 161 | 15.72% | 730 | 133 |
| 16 | 4 | 1.823 | 161 | 15.72% | 730 | 133 |

## Wall is transfer-bound, and resurrection adds nothing after that

| relationship | correlation |
|---|---|
| wall vs **transfers/step** | **+0.9748** |
| wall vs resurrection rate | +0.8404 |
| transfers vs resurrection rate | +0.9028 |
| **residual after transfers, vs resurrection rate** | **-0.1778** |

Fitting wall against transfers per step gives **545.9 µs per 13.22 MB
row = 24.2 GB/s**, against a measured PCIe H2D ceiling of ~28 GB/s on
this box class. That the cost model lands on the link's real bandwidth is the
check that the counter means what it is being treated as; an early pass
implied 6 TB/s, which is how a units error (per-step wall regressed on
*total* transfers) surfaced.

Once transfers are accounted for, the residual correlation with resurrection
rate is **-0.1778**, spread -0.162..+0.144 ms against a wall
range of 0.376–1.823 ms. That is not a measurable effect.

**Resurrection rate correlates with being SLOWER (+0.8404)**, because
both it and the transfer count are driven by capacity pressure. The
configurations producing high rates are the ones performing worst: at
15.72% the cache does 730 transfers and 133 hits; at 0.00% it
does 5 transfers and 1019 hits and runs **4.8× faster**.

## What is refuted, precisely, and what is not

R2 predicts resurrection is *disproportionately valuable* — that reaching a
few percent shows up in wall time. Reaching **three times** the top of that
band shows up only as harm, and the harm is attributable to the pressure
that produced the rate rather than to the resurrections.

What this does **not** show is that a resurrection is worthless. A
resurrection is an avoided transfer by construction, so holding a
configuration fixed and disabling them could only make it slower. **No such
control exists**: `protected = rows` leaves nothing demotable and
`VramSlots._claim` raises *"no slot available"*, so there is no configuration
with the cache otherwise identical and resurrection off. The counterfactual
is unmeasurable in this engine and is not estimated here.

R2 as registered is a claim about the observable, and the observable is flat.

## The counters describe the timed window, after a correction

The first version of this measurement paired a wall timed over 256 steps
with **lifetime** cache counters — `traffic()` totals include `_prime` and
every warmup forward — so the rates and transfer counts described a
different window than the wall beside them (Bugbot, gnf4#152). This is the
same defect gnf4#132 fixed in `run_gate1.py` earlier the same day, written
fresh into a new harness.

Counters are now snapshotted at the measurement boundary and differenced.
The correction moved every number and changed no conclusion: the top rate
went 16.80% → 15.72%, the slope 533 → 545.9 µs, the residual
correlation −0.144 → -0.1778. The lowest rate is unchanged at
5.37%, so every nonzero point still sits above R2's band.

A third, low-severity one followed the fix itself: the windowed `diff`
blacklisted non-counters by NAME, and `bytes` is a GAUGE (`rows *
row_stride`), so it differenced to a constant 0 and the committed receipt
stores 0 where the cache footprint is 317 MB at 24 rows. **No published
number uses that field**, and it is derivable as `rows x row_stride`, so the
receipt is left as taken rather than re-run for a column nothing cites. The
harness now WHITELISTS the monotone counters instead: a gauge wrongly
differenced reads 0 and looks like a measurement, while a counter wrongly
carried through reads as a lifetime total and is obvious beside a windowed
neighbour. Whitelisting fails in the visible direction.

A second defect surfaced with it: the transfer column had been reading
`overwritten`, an eviction counter, rather than `host_to_cache_rows`. The
two are near-collinear here, which is why it went unnoticed until the
implied bandwidth came out at 6 TB/s.

## A gap this measurement had to work around

**A gpt-oss arena cannot currently be served by `Mxfp4NvmeResidency`
without a shape fix-up that nothing in the shipped path performs.**

`bake` records the checkpoint's own `[5760, 90, 16]`, which is what makes
`sha256(arena) == sha256(source)` provenance hold. `engine_segment_map`
requires `[n, k]` — for a real reason: it distinguishes blocks from scales by
the invariant that blocks width is exactly 16× scales width, and on the 3D
shape both read 90 so the discriminator collapses. Flattened to
`[5760, 1440]` it is 1440 against 90 and works. The bytes are identical;
only the metadata differs.

**Settled in gnf4#153: the engine.** The arena keeps recording what the
checkpoint says — that fidelity is the whole point of `sha256(arena) ==
sha256(source)` and of `verify --against-source` — and
`engine_segment_map` normalises onto `[n, k]` on a local copy, which is
where the seam map always described the flattening. Rewriting shapes at bake
time would have bought a simpler validator at the cost of the arena's
fidelity to its source, and put newly-baked arenas at odds with every arena
already on disk. The harness workaround here predates that fix and is left
as taken.

## What this does not establish

- One layer (layer 0), one model, one routing seed, `hot_ids=()`,
  `hot_rows` fixed at 32 — the DRAM tier's own pressure is not varied.
- Routing is a fixed pseudorandom top-4, not a captured trace. Real routing
  has locality this does not, and a burstier stream would raise hit rates at
  every capacity. **The sweep carries the argument, not any single point.**
- The zero-rate arms still show 5 transfers, not 0: a handful of rows land
  in the timed window regardless. They are the floor, not a clean baseline.
