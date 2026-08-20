# R2, scored — resurrections reach 16.8% of routed work and move wall by nothing

Registered (`PREREG-tribrid-stage3`, R2):

> VRAM resurrection is disproportionately valuable — **even 2–5% of routed
> invocations moves wall time** — **refuted by no measurable wall effect**.

**REFUTED.** Rates of **5.37% to 16.80%** were reached, comfortably above the
band R2 conditions on, and wall time is explained by row transfers with
nothing left over for resurrection.

This is the first time this program has run `Mxfp4NvmeResidency` on a real
model rather than a fixture. Arena: **openai/gpt-oss-20b**, baked with
`nvme_arena.bake` — 24 layers × 32 experts, 13.22 MB per expert row,
10.15 GB. One layer, decode-shaped (T=1), 256 steps, 2 repeats per point,
RTX 5090. Receipts in `r2-wall-2026-08-20/`.

## The sweep

Capacity (`rows`) and ownership (`protected`) both swept; every point is a
real forward through the real engine.

| rows | prot | wall (ms) | resurrections | per routed | transfers | cache hits |
|---|---|---|---|---|---|---|
| 48 | 36 | 0.376 | 0 | 0.00% | 32 | 1057 |
| 48 | 44 | 0.374 | 0 | 0.00% | 32 | 1057 |
| 32 | 28 | 0.383 | 55 | 5.37% | 88 | 946 |
| 32 | 24 | 0.402 | 105 | 10.25% | 172 | 812 |
| 48 | 24 | 0.420 | 105 | 10.25% | 172 | 812 |
| 24 | 20 | 0.882 | 87 | 8.50% | 303 | 699 |
| 24 | 18 | 0.883 | 99 | 9.67% | 344 | 646 |
| 32 | 16 | 0.893 | 102 | 9.96% | 394 | 593 |
| 16 | 12 | 1.353 | 135 | 13.18% | 499 | 455 |
| 24 | 12 | 1.354 | 135 | 13.18% | 499 | 455 |
| 48 | 12 | 1.365 | 135 | 13.18% | 499 | 455 |
| 12 | 8 | 1.361 | 144 | 14.06% | 636 | 309 |
| 16 | 8 | 1.363 | 144 | 14.06% | 636 | 309 |
| 32 | 8 | 1.366 | 144 | 14.06% | 636 | 309 |
| 12 | 6 | 1.827 | 159 | 15.53% | 717 | 213 |
| 24 | 6 | 1.831 | 159 | 15.53% | 717 | 213 |
| 12 | 3 | 1.829 | 172 | 16.80% | 778 | 139 |
| 16 | 4 | 1.827 | 172 | 16.80% | 778 | 139 |

## Wall is transfer-bound, and resurrection adds nothing after that

| relationship | correlation |
|---|---|
| wall vs **transfers/step** | **+0.976** |
| wall vs **resurrection rate** | +0.868 |
| transfers vs resurrection rate | +0.922 |
| **residual after transfers, vs resurrection rate** | **−0.144** |

Fitting wall against transfers per step gives **533 µs per 13.22 MB row =
24.8 GB/s**, against a measured PCIe H2D ceiling of ~28 GB/s on this box
class. That the cost model lands on the link's real bandwidth is the check
that the counter means what it is being treated as; a slope implying 6 TB/s
would have meant the units were wrong (an earlier pass of this analysis had
exactly that, from regressing per-step wall on total transfers).

Once transfers are accounted for, the residual correlation with resurrection
rate is **−0.144**, with a spread of ±0.16 ms against a wall range of
0.374–1.831 ms. That is not a measurable effect.

**Resurrection rate correlates with being SLOWER (+0.868)**, because both it
and the transfer count are driven by capacity pressure. The configurations
that produce high resurrection rates are the ones performing worst: at 16.80%
the cache is doing 778 transfers and 139 hits; at 0.00% it does 32 transfers
and 1057 hits and runs **4.9× faster**.

## What is refuted, precisely, and what is not

R2 predicts resurrection is *disproportionately valuable* — that reaching a
few percent is enough to show up in wall time. Reaching 3× the top of that
band shows up in wall time only as **harm**, and the harm is attributable to
the pressure that produced the rate rather than to the resurrections.

What this does **not** show is that a resurrection is worthless. A
resurrection is an avoided transfer by construction, so holding a
configuration fixed and disabling them could only make it slower. **No such
control exists**: `protected = rows` leaves nothing demotable and
`VramSlots._claim` raises *"no slot available"*, so there is no
configuration with the cache otherwise identical and resurrection off. The
counterfactual is unmeasurable in this engine, and this document does not
estimate it.

R2 as registered is a claim about the observable, and the observable is
flat. That is what "no measurable wall effect" means.

## A gap this measurement had to work around

**A gpt-oss arena cannot currently be served by `Mxfp4NvmeResidency`
without a shape fix-up that nothing in the shipped path performs.**

`bake` records the checkpoint's own `[5760, 90, 16]`, which is what makes
`sha256(arena) == sha256(source)` provenance hold. `engine_segment_map`
requires `[n, k]` — and for a real reason: it distinguishes blocks from
scales by the invariant that blocks width is exactly 16× scales width, and
on the 3D shape both read 90 so the discriminator collapses. Flattened to
`[5760, 1440]` it is 1440 against 90 and works. The bytes are identical
either way; only the metadata differs.

The harness flattens a **copy** of the index (`flatten_block_shapes`).
Whether that belongs in the bake or in the engine is a real decision with
provenance consequences on both sides, and a measurement harness is the
wrong place to settle it. Recorded here as the gap it is.

## What this does not establish

- One layer, one model, one routing seed, `hot_ids=()`. Layer 0 of
  gpt-oss-20b, not a full forward.
- Routing is a fixed pseudorandom top-4, not a captured trace. Real routing
  has locality this does not; a burstier or more repetitive stream would
  raise hit rates at every capacity. The *sweep* is what carries the
  argument here, not any single point.
- `hot_rows` is fixed at 32 throughout, so the DRAM tier's own pressure is
  not varied.
