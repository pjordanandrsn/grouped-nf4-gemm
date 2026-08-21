# The residual is OUTSIDE the read, by about 6 to 1

**Registered outcome: OUTSIDE.** At qd=1, where the partition is
interpretable, the soft arm's reads cost what their count says. Its work
*around* the reads costs 26.6% more.

Registered in [`PREREG-read-timing.md`](PREREG-read-timing.md) before the box
was rented. Third independent box, chosen in #153's slow-disk class rather
than by peak bandwidth — the axis #163 identified as the wrong one.

## It replicates first, and the box is the sharpest yet

| | #153 | #163 | here (leading control) |
|---|---|---|---|
| wall Δ | +5.8% | +6.3% | **+6.69%** |
| reads Δ | +1.4% | +1.3% | **+1.35%** |
| residual | +4.4 | +5.0 | **+5.33** |
| per-arm spread | 0.34% | 3–7% | **1.0–1.4%**, and **0.34%** at qd=1 |

The slow disk did what it was picked to do: 65–75 s arms let the signal
accumulate, and the qd=1 hard arm lands at 0.34% spread — matching #153, the
tightest instrument in the campaign — against an effect of 6.4%.

## Preconditions, verified rather than assumed

- `reader_mode: O_DIRECT`, **and confirmed to bypass**: the same 64 MB read
  twice cost 41.3 ms then 40.8 ms, ratio 0.99. Disk ≈1.5 GB/s.
- `arena_row_bytes 3342336` — byte-identical geometry to #163's receipt.
- reads 32169 / 32605 — matching `RESULTS-r10.md` at this capacity.
- `drop_caches` unavailable (read-only `/proc` in the container) and **not
  load-bearing under O_DIRECT**; recorded in every receipt regardless.

## Drift bracket

The timed runs are bracketed by a control before and after:

| | Δwall | hard spread | soft spread |
|---|---|---|---|
| leading control | +6.69% | 1.37% | 1.02% |
| trailing control | +6.01% | 1.42% | 1.49% |

0.68 points apart, inside the combined spread. The box did not drift across
the session, so the timed runs sit on stable ground.

## The partition, at qd=1

One worker, so `read_ns` is directly comparable to wall.

| | hard | soft | Δ |
|---|---|---|---|
| wall | 75411.4 ms | 80271.4 ms | **+6.44%** |
| `read_ns` | 62252.0 ms | 63613.2 ms | **+2.19%** |
| `non_read_ns` | 13159.4 ms | 16658.2 ms | **+26.59%** |
| reads | 32169 | 32605 | +1.36% |

Decomposing the 4859.9 ms of extra wall:

| term | ms | share |
|---|---|---|
| explained by 436 more reads, inside reads | 843.7 | 17.4% |
| explained by 436 more reads, outside reads | 178.4 | 3.7% |
| **residual (unexplained)** | **3837.8** | **79.0%** |
| — of the residual, inside reads | 517.4 | 13.5% |
| — of the residual, **outside reads** | **3320.4** | **86.5%** |

Per read: the soft arm's reads are **0.82%** slower (1.935 → 1.951 ms), while
its non-read work per read is **24.9%** heavier (0.409 → 0.511 ms).

qd=4 agrees on the half that is interpretable there: `read_ns` +1.9% against
reads +1.4%, i.e. total device work still tracks read count. Its
`non_read_ns` is **not** interpretable and is not used — with four workers,
summed worker time is 224% of wall and the subtraction goes negative. That is
arithmetic, not a finding, and it is why the verdict rests on qd=1.

## What this rules in and out

A per-read *storage* cost exists but is small: 517 ms, 13.5% of the residual.
It is also not locality — `RESULTS-read-locality.md` showed both arms read the
same rows in the same order, medians identical at four capacities.

The remaining 86.5% is in the tier's own work around the read. That does not
contradict #161/#163 bounding the **demote path** at ≤0.8 of 4.1 points:
`non_read_ns` covers the whole `ensure()` path — planning, slot reservation,
pending-event handling, `as_completed` dispatch — of which demote is one part
already measured and found small.

So the sixth candidate is not a guess any more. It is a named region, with a
size (3.3 s of a 4.9 s gap), a per-read magnitude (+24.9%), and a queue-depth
independence that matches the residual's known behaviour.

## The first instrument measured the wrong thing

The first `--time-reads` wrapped `reader.read_row`, which is
`self._pool.submit(self._read, ...)` — it hands the read to a thread pool and
returns a Future **without blocking**. It measured submission, not I/O, and
reported 0.85 s to move 107 GB: **126 GB/s on a 1.5 GB/s disk.**

That run produced "OUTSIDE the read" as well. The conclusion here is the same
one, and that is a coincidence, not corroboration: an instrument that puts all
I/O into `non_read_ns` by construction will say OUTSIDE whatever is true. It
was withdrawn and re-run against `reader._read`, the function the pool
actually executes. Verified before spending the second run: qd=1 `read_ns` is
now 82.2% of wall implying 1.73 GB/s, consistent with the disk.

The first attempt also drifted — its hard arm fell monotonically 65.3 → 61.0 s
across five repeats. That was transient: the corrected run's hard arm holds
0.34%, so the drift was the box, not the instrumentation.

## What this does not show

One trace, one model, one capacity (`rows=256, protected=248`), one box. The
partition is only read off qd=1; qd=4 corroborates the `read_ns` half but
cannot speak to the rest.

"Outside the read" is a region, not a mechanism. It does not say *which* part
of `ensure()`, and the honest next step is a profile of that path rather than
another guess at it.

## Receipts

`read-timing/` — `warm_replication`, `ctl_qd4` (leading control), `sanity`
(instrument verification), `v2_timed_qd4`, `v2_timed_qd1`, `v2_ctl_trailing`.
Box destroyed; total spend **$0.27**.
