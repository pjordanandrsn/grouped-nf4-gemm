# R10 — REFUTED, and the control it implies for R1's read claim

Receipt: [`r10.json`](r10.json). Harness: [`score_r10.py`](score_r10.py).
Trace: [`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl). No box.

## As registered

> **R10** — reclaimable residency reduces promotion churn, NVMe rereads and
> H2D refills without reducing effective hit rate. **Refuted if churn
> unchanged or hit rate drops.** (`PREREG-tribrid-stage3.md`)

Unlike R1–R3, R10 is stated in the metric that survives scrutiny — physical
refills rather than resurrection rate (see [`RESULTS-r3.md`](RESULTS-r3.md)).
It is answerable as written, **provided the arms are matched on capacity**.

## Matched on capacity, soft eviction is slightly worse

Both arms hold the same number of physical rows. The only difference is
whether ownership is capped below that number, leaving the remainder
readable-but-unowned.

| rows | protected | hard reads | soft reads | Δ reads | Δ churn | verdict |
|---|---|---|---|---|---|---|
| 128 | 64 | 44,298 | 44,939 | **+1.4%** | +1.5% | REFUTED |
| 128 | 120 | 44,298 | 44,965 | +1.5% | +1.5% | REFUTED |
| 192 | 96 | 38,086 | 38,607 | +1.4% | +1.4% | REFUTED |
| 256 | 128 | 32,169 | 32,569 | +1.2% | +1.3% | REFUTED |
| 384 | 192 | 21,890 | 22,129 | +1.1% | +1.1% | REFUTED |
| 512 | 256 | 13,616 | 13,706 | +0.7% | +0.7% | REFUTED |

**10 of 10 refuted.** Reads and churn both move the wrong way, consistently
and by a small margin. Capping ownership below capacity costs about 1% and
buys nothing.

The direction makes sense once stated: if you hold 128 rows, owning all 128
is strictly more retention than owning 96 and hoping the other 32 survive
until they are wanted. Reclaimable rows lose every allocation contest — that
is what makes them reclaimable — so they are the first overwritten.

## The control this implies for R1's read-reduction clause

`RESULTS-tribrid-reclaimable.md` records a preregistered clause as confirmed:

> ≥10% fewer physical NVMe reads (Arm A vs Arm B) — **CONFIRMED** — −14.9%
> and −29.6%

**Those arms are not matched on capacity.** Arm A is `hot_rows == protected
== P`, a pool of **P** rows. Arm B is a **128**-row pool with P protected. B
simply has more memory.

Reproducing that arm *shape* on this trace reproduces the result. Holding
capacity fixed reverses it:

| P | A rows | A reads | B rows | B reads | Δ |
|---|---|---|---|---|---|
| 96 | 96 | 47,771 | 128 | 44,948 | **−5.9%** |
| 64 | 64 | 51,932 | 128 | 44,939 | **−13.5%** |
| 32 | 32 | 57,741 | 128 | 44,932 | **−22.2%** |

| P | hard reads (128 rows) | soft reads (128 rows) | Δ |
|---|---|---|---|
| 96 | 44,298 | 44,948 | **+1.5%** |
| 64 | 44,298 | 44,939 | **+1.4%** |
| 32 | 44,298 | 44,932 | **+1.4%** |

The unmatched shape produces R1's numbers — including −13.5% almost exactly.
The matched control produces the opposite sign. **On this trace the read
reduction is attributable to Arm B holding more rows, not to reclaimable
residency**, and the decisive comparison is the bottom table: given 128 rows,
owning all of them beats capping ownership at 96.

## What I am and am not claiming

**I am not claiming R1's measurements are wrong.** They were taken on real
hardware with a real model and they reproduce here. What does not survive is
the **attribution** — that the reads fell *because* rows were reclaimable.

**This control should be re-run on R1's own setup** before its clause is
changed. Mine is a different trace, a synthetic 224-byte-row arena, and
crucially **uncontended**. R5 reports soft eviction *faster than* hard under
contention, and contention is exactly where a ghost row that survives to be
resurrected could pay for itself. Nothing here tests that.

`RESULTS-tribrid-reclaimable.md` is **not edited by this PR.** Rewriting
another document's verdict from a different trace on a synthetic arena would
be the same over-reach this measurement is pointing at.

## Limits

- One model, one prompt, 512 decode steps, uncontended.
- Synthetic 16×64 arena, 224-byte rows. Bytes do not matter to a residency
  question; no I/O timing is claimed.
- `ColdTier` only. The VRAM side cannot run the hard arm at all —
  `protected == rows` leaves `VramSlots` nothing demotable and the allocator
  raises, which is itself a structural asymmetry between the two tiers.
