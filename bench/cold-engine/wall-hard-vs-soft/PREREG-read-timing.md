# Preregistration: is the residual INSIDE the read, or around it?

Registered before the instrument was run on real hardware.

## Where the residual stands

Soft costs ~6.4% more wall on ~1.35% more reads. About **5 points** are
unexplained, and the gap does not move with queue depth. Five candidates are
eliminated: the demote sort, the whole demote path, "lower effective
bandwidth" (a restatement), CPU–I/O overlap, and read locality — the last
refuted by showing both arms read the arena in the *same order*, medians
identical at four capacities.

What survives is a **per-read** cost that is not explained by which rows are
read, how many, or in what order. Every remaining explanation is either inside
the read syscall or outside it, and nothing so far has measured which.

## Instrument

`wall_hard_vs_soft.py --time-reads` wraps `reader.read_row` and accumulates
nanoseconds spent inside it, partitioning the wall into `read_ns` and
`non_read_ns`. The reader is wrapped, not edited — the same non-invasive
capture `score_locality.py` used, where it was validated by the captured read
count matching the tier's own miss counter exactly.

The wrapper costs both arms the same per read, so it inflates `read_ns`
equally and cannot manufacture a difference beyond the read-count difference
already being accounted for.

Config is the qd probe's own: `rows=256`, `protected=248`, 5 repeats, A/B/A,
`drop_caches` between arms, at **both** qd=4 and qd=1 (the residual is
queue-depth invariant, so both must show the same partition).

## Predictions, registered

Let `Δwall ≈ +6.4%` and `Δreads ≈ +1.35%`.

- **INSIDE** — `Δread_ns` tracks `Δwall` (materially above `Δreads`), and
  `Δnon_read_ns` is small. The residual is in the storage path: the same rows,
  in the same order, at the same size, cost more to fetch in the soft arm.
  That would be a genuinely surprising result and would need a device-level
  explanation.
- **OUTSIDE** — `Δread_ns` tracks `Δreads` (~1.35%, i.e. reads cost what their
  count says), and `Δnon_read_ns` carries the ~5 points. The residual is in
  our own path around the read, despite the demote path having been bounded at
  ≤0.8 of 4.1 points.
- **NEITHER** — both deltas come in near `Δreads` and the wall gap does not
  reproduce. Then the residual is not robust on this box and the earlier
  measurement needs re-examining before anything else is attributed to it.

## Control, run first

The same configuration **without** `--time-reads`. The wall residual must
reproduce unchanged. If instrumenting the reader moves the residual, the
partition measures the instrument and not the effect, and nothing else here is
usable.

## Stated in advance

This partitions the residual; it does not name a mechanism. "Inside" would
narrow it to the storage path and rule out our bookkeeping for good; "outside"
would do the reverse. Either outcome is a real constraint on the sixth
candidate — which I am deliberately not naming before the split is measured.
