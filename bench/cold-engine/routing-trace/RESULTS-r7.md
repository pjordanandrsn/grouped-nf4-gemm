# R7 — REFUTED: the knee does not move, and on this trace there is barely a knee

Receipt: [`r7.json`](r7.json). Harness: [`score_r7.py`](score_r7.py). Trace:
[`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl). No box.

## As registered

> **R7** — reclaimable residency moves the NVMe knee outward by 20–50% on
> workloads with temporal locality. **Refuted if knee unmoved.**
> (`PREREG-tribrid-stage3.md`)

The knee is in **capacity** space: the pool size below which shrinking starts
costing reads sharply. "Outward" means reaching the cheap regime with fewer
rows. Operationalised as the smallest pool whose reads stay within 10% of the
best on the curve.

## The curve, both arms

| rows | hard reads | soft reads | Δ |
|---|---|---|---|
| 64 | 51,932 | 53,023 | +2.1% |
| 128 | 44,298 | 44,964 | +1.5% |
| 192 | 38,086 | 38,620 | +1.4% |
| 256 | 32,169 | 32,605 | +1.4% |
| 384 | 21,890 | 22,149 | +1.2% |
| 512 | 13,616 | 13,720 | +0.8% |
| 640 | 7,389 | 7,440 | +0.7% |
| 768 | 3,340 | 3,338 | −0.1% |
| 896 | 1,311 | 1,308 | −0.2% |
| 1024 | 989 | 989 | 0.0% |

**Hard knee: 1024 rows. Soft knee: 1024 rows. Movement: +0.0%.** R7 needed
+20%. **REFUTED.**

Soft is worse at every capacity below 768 and indistinguishable above it.
This is the same ~1% penalty R10 measured, now across the whole curve.

## There is barely a knee to move

R7 presupposes a knee. On this trace the read-vs-capacity curve is smooth and
concave — the marginal value of a row *falls* monotonically as the pool grows:

| range | reads saved per row added |
|---|---|
| 64 → 128 | 119 |
| 128 → 256 | 95 |
| 256 → 512 | 72 |
| 512 → 768 | 40 |
| 768 → 1024 | 9 |

No sharp bend anywhere. The only abrupt feature is the working set running
out at full capacity. A prediction about *moving* a knee is hard to score
against a workload that does not present one, and that is worth recording
alongside the verdict: OLMoE's top-8-of-64 decode routing has enough churn
that capacity buys reads at a smoothly diminishing rate.

## The R1 arm shape, and why it looks like a huge knee shift

Running the comparison R1 used — soft always holding the **full 1024-row
pool** while `protected` varies — the soft "knee" lands at **protected = 64**,
with reads at the compulsory floor of 989.

That reads as a spectacular outward move. It is not a move at all. With a
1024-row pool the entire working set is resident regardless of the ownership
cap, so `protected` can fall to 64 without costing a single read — the other
960 rows are still holding everything. The knee did not move outward; the
axis changed from *capacity* to *ownership cap*, and capacity was never the
binding constraint.

This is the same confound [`RESULTS-r10.md`](RESULTS-r10.md) identifies in
R1's read clause, in its clearest form.

## Re-verified under the qd pin

`ColdTier`'s read queue depth defaulted to the host CPU count, which made
counters non-reproducible above `qd=1` (gnf4#169). This document's numbers
were taken at that default, like `score_r3/r8/r9/r10`'s were.

Re-run at the pinned `qd=1`: **three cells of twelve differ, by one or two
reads** — 41,646→41,644 at 160 rows, 38,620→38,619 at 192, 22,149→22,150 at
384. Every other cell is identical. The knee is unmoved at **+0.0%** and the
verdict is unchanged.

That is the same small-blast-radius ordering effect #169 measured, and it
does not reach any conclusion here. `score_r7.py` inherits the pin through
`score_r10.run`, whose default is now 1.

## Limits

- One model, one prompt, 512 decode steps, **uncontended**. R5 reports soft
  eviction faster than hard under contention; nothing here tests that, and
  contention is where a ghost row could pay for itself.
- Synthetic 16×64 arena of 224-byte rows. Reads are counted, not timed.
- `ColdTier` only. The VRAM side cannot express the hard arm —
  `protected == rows` leaves `VramSlots` nothing demotable.
- The knee threshold (within 10% of best) is a choice. The verdict is not
  sensitive to it here: the two curves are within 2% of each other
  everywhere, so no threshold separates them.
