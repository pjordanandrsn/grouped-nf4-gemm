# RESULTS — K5: M-tile tensor-core probe at M=1 (STRUCTURE-REFUTED)

Measured 2026-08-24 under PREREG-k5-mtile-probe + AMENDMENT-k5-stack +
AMENDMENT-k5-graph-timing. Receipts: `receipts-k5-probe/attempt4-graph/`
(box 4: RTX 5090, driver 595.71, EPYC host with triad 105 GB/s
recorded-advisory — GPU-pure cycle; instance destroyed, account
verified zero). Verdict calculator output, both receipts:

```
K5 VERDICT: STRUCTURE-REFUTED   (k5_probe_new.json, t28-floor)
  ratio 1.303 >= 0.9
  gemv_sum=76.6us mtile_sum=99.9us
K5 VERDICT: REFUSE              (k5_probe_old.json, t27-t33)
  no successful M-tile config at gate_up   [expected: path absent]
```

## The registered outcome

On the floor stack (torch 2.8.0 / triton 3.4.0 — **all four compile
diag cells OK**, so the pyproject `triton>=3.4` floor is exactly
right), the best M-tile cell-sum is **99.9 µs vs the GEMV winners'
76.6 µs: ratio 1.303 ≥ 0.9 ⇒ the compute-structure theory is REFUTED
for the existing kernels.** The existing tensor-core path does not
beat the scalar-reduce GEMV at M=1. Per the registered consequence
map, the kernel lane's remaining option is a bespoke `tl.dot` GEMV,
admissible only under a fresh prereg with this probe as its baseline;
the elementwise-fusion lane takes priority.

## The K5 graph anchor (and the amendment-2 vindication)

Graph-basis GEMV pair: **76.9 µs** (t27-t33) / **76.6 µs**
(t28-floor) — cross-stack delta **0.4%**. Eager read 95.7 and
112.2 µs on the same box/stacks: the eager–graph gap (19–36 µs/pair)
is the host-enqueue cost amendment 2 identified, and it moved with
the python stack while the graph basis held still. Attempts 2–3
(drivers 590/595, eager pair 94.5/94.6 µs, destroyed by the
since-retired eager anchor gate) are consistent with the same graph
number plus larger host gaps; their JSONs predate the graph
instrument and exist as task-log numbers quoted in amendment 2.

## Receipt-visible observation (NOT a registered outcome)

The sum hides opposite cells: gate_up M-tile best 72.0 vs GEMV 36.1
(2.0× worse — 15/16 of every 16-row MMA tile is padding at M=1), but
**down M-tile best 27.8 vs GEMV 40.6 (0.68×)** — the down GEMV's
sk=1/512-program launch underfills the 170-SM part, and the M-tile
covers it better. A down-only mixed routing would read 63.9/76.6 =
0.83 — inside the registered pause band even if it had been
registered, which it was not. Recorded here as the seed fact for any
future routing prereg; no product change follows from it now.

## Stack notes

- triton 3.3.0 (below the declared floor): M-tile path absent
  (`getMMAVersionSafe` / gather-layout asserts; AMENDMENT-k5-stack).
  The REFUSE receipt on the healthy box completes that record.
- The GEMV winners' graph times (gate_up 36.2/36.1, down 40.7/40.6)
  are within 0.4% across torch 2.7/triton 3.3 vs torch 2.8/triton
  3.4: **no re-baselining of the certified decode path is needed for
  a floor-stack bump.**
