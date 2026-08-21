# Preregistration: does #153's residual survive the tier optimisations?

Registered before renting the box.

## Why this has to be re-run

`RESULTS-read-timing.md` put **86.5% of the residual outside the read**, on a
tier whose CPU has since been cut **41–49%** (#175, #176). That document now
describes code that no longer exists. Either the residual moved with the
optimisation — which would close a six-candidate investigation — or it did not,
which locates it somewhere tier CPU never reached.

## The baseline it is measured against

Same box class, same arena geometry, same configuration (`rows=256`,
`protected=248`, `qd=1`, 5 repeats, `--pinned`), from `read-timing/`:

| quantity | before |
|---|---|
| wall Δ | +6.44% |
| reads Δ | +1.36% |
| `read_ns` Δ | +2.19% |
| `non_read_ns` Δ | **+26.59%** |
| residual | **5.08 points** |
| `non_read` share of hard wall | 17.5% |

## Predictions, registered

- **DISSOLVED** — the residual falls to ≲2 points and `Δnon_read_ns` drops
  well below +26.59%. The extra work the soft arm did around its reads was
  largely the victim sweep, and removing it removed the effect. This is what I
  expect if `_victim` was the mechanism: the soft arm called it ~1.4% more
  often *and* each call was linear in `hot_rows`.
- **SURVIVES** — the residual stays near 5 points and `Δnon_read_ns` stays
  high, even though `non_read_ns` itself is much smaller in absolute terms.
  Then the cause is a per-read cost that tier CPU never explained, and the
  thread-dispatch machinery (~28% of tier CPU after the heap: a Future, an
  Event and several lock round-trips per read) becomes the prime suspect.
- **PARTIAL** — the residual shrinks materially but not to noise. Then the
  sweep was *a* contributor, not *the* mechanism, and what remains is the part
  worth naming.

## Stated in advance

`non_read_ns` will shrink in absolute terms no matter what — that is what the
optimisation did, and it is not the question. **The question is the soft-minus-
hard DELTA**, which is what the residual is made of. A smaller `non_read_ns`
with an unchanged +26.6% delta means the optimisation did not touch the
mechanism.

Guards carried over from the last run and non-negotiable: O_DIRECT confirmed
to bypass (not merely to open), arena row bytes 3342336, read counts matching
`RESULTS-r10.md`, a leading and trailing control bracketing the timed run, and
the instrument's own control — the residual must reproduce **without**
`--time-reads` before any partition is read.
