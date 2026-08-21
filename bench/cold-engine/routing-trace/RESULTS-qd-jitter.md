# ColdTier is not reproducible above qd=1, and four scorers ran at the default

**No verdict changes. Four scorers become reproducible, and one instrument
property that the campaign already depended on is now written down.**

`ColdTier(...)` takes `qd: int | None = None`, and `None` **sizes the read
queue from the host's CPU count**. A counter comparison run at the default is
therefore not reproducible run-to-run on one box, and not comparable between
boxes — the queue depth is a property of the machine, not of the experiment.

`qd=1` was already chosen for the demote probe and for R6 *because* higher
depths were not reproducible. That reason was never written down and the
spread was never quantified, so nothing stopped four other scorers from
running at the host default.

## The spread, measured

`qd_jitter.py`, 7 repeats of one replay (rows=512, protected=461) on the
committed OLMoE trace. Values are max−min across the 7 runs:

| qd | misses | evictions | resurrections | reclaimable_overwritten | hits | reproducible |
|---|---|---|---|---|---|---|
| **1** | 0 | 0 | 0 | 0 | 0 | **yes** |
| 2 | 1 | 1 | 1 | 1 | 1 | no |
| 4 | 1 | 1 | **2** | 1 | 1 | no |
| 8 | 1 | 1 | 1 | 1 | 1 | no |

Mechanism: with `qd>1` several reads are in flight and are serviced in
completion order, which the kernel and device do not guarantee matches issue
order. Completion order decides which row lands first, which decides what the
next demote sees. At `qd=1` exactly one read is outstanding, ordering is
forced, and the replay is a pure function of the trace.

The spread does not grow with depth — qd=8 is no worse than qd=2. This is an
ordering effect with a small blast radius, not an accumulating error.

## Which scorers were exposed

`ColdTier` constructions in `bench/`, before this change:

| construction | queue depth | exposed? |
|---|---|---|
| `score_r3.py`, `score_r8.py`, `score_r9.py`, `score_r10.py` | **host default** | **yes** — counters compared between arms, nothing checking reproducibility |
| `bench_direct.py` | host default | no — self-checks. It asserts `reads_match` across all three A/B/A legs, which is exactly the invariant a reordering flip would break, so a perturbation surfaces instead of hiding |
| `wall_hard_vs_soft.py`, `score_r6.py`, `qd_jitter.py` | pinned | no |

The four exposed scorers now take `--qd`, defaulting to **1**.
`bench_direct.py` is deliberately left alone: it measures *wall time* on a
real arena, where pinning would cost 50% (below), and it already detects the
failure it would be pinned to prevent.

## No verdict moves

Re-run at `qd=1` against the committed default-qd baseline:

- **R10** — 10 of 10 still **REFUTED**; Δ reads +0.7%…+1.5% unchanged. Six
  cells move, across three of the ten configurations (128/120, 256/248,
  384/376), by −2/+1/−1 — exactly the measured jitter, two orders of magnitude
  below the effect being claimed. In every case **reads and evictions moved by
  the same amount**, which is what a single ordering flip predicts: one row
  refilled instead of resurrected is also one more eviction. The five `half`
  budgets did not move at all.
- **R8** — **byte-identical** to the baseline at every cell; ratios 1.8×–18.2×.
- **R9** — unchanged; both-rate 0.0% at the four VRAM-starved points and
  23.5% / 27.5% at the two where the caches can overlap. The dominance
  argument that refuted R9 is untouched.
- **R3** — still **UNDETERMINED as registered**, and for the same reason: 6
  holds / 4 refuted, splitting cleanly by a budget the prediction never pins
  (`half` 5–0 holds, `rows-k` 1–4 refuted). Pinning the queue depth does not
  rescue a prediction whose verdict is decided by an unpinned *configuration*.

All four now reproduce exactly across repeated runs.

This is the expected result, and worth stating plainly: these conclusions rest
on differences of hundreds to thousands of reads, and the instrument's spread
is 1–2. **The jitter was never large enough to threaten a verdict.** What it
threatened was reproducibility — a reader re-running `score_r10.py` got
different digits than the doc quoted, with nothing explaining why.

## Cost, and why the default differs by harness

I expected `qd=1` to be expensive — it serialises reads — and on the real
harness it is. On these scorers it is **free**, and that gap is the whole
reason the two defaults differ:

| harness | rows | qd=1 | qd=4 | cost |
|---|---|---|---|---|
| `score_r8.py` (toy arena) | 224 B | 1.4 s | 1.6 s | none |
| `score_r10.py` (toy arena) | 224 B | 58.8 / 57.6 s | 54.6 / 58.9 s | **none — ranges overlap** |
| `wall_hard_vs_soft.py` (real NVMe, #163) | 3.3 MB | 36303 ms | 24279 ms | **~50%** |

The `score_r10` row is why it was worth repeating: a single pair read as a
clean ~8% penalty, and the repeat put a qd=4 run (58.9 s) above both qd=1 runs.
At n=2 the two depths are **indistinguishable** on this arena, and the 8% was
noise. Reported as a range rather than a mean because two runs do not support
one.

A 224 B read is served from page cache, so there is essentially no queue to
lose by draining it one at a time; a 3.3 MB read against a real device has a
great deal. Hence: **the offline scorers default to `qd=1`** — reproducibility
is free there, and they emit numbers that go into documents — while
`wall_hard_vs_soft.py` keeps its default of 4 and exposes `--qd`, since
pinning would distort the wall time it exists to measure.

`--qd` is on all of them either way, so neither default is load-bearing.

## What this does not show

The probe builds the **same synthetic arena the four scorers build** — 16
layers x 64 experts, **224 B rows** — so for the scorers' own reproducibility
this is not an approximation: it is their operating condition, measured
directly.

It is *not* representative of the real-NVMe harness. `wall_hard_vs_soft.py`
reads a baked OLMoE arena with **3.3 MB rows**, ~15,000x larger, on a rented
box rather than a laptop filesystem. A 224 B read is served from page cache
almost immediately, so there is very little window in which reordering can
occur; MB-scale reads against a real device have a far wider one. The
*existence* and *mechanism* of the ordering dependence carry over — they follow
from completion order not being issue order — but **the magnitude does not**,
and could be considerably larger there. That is the argument for pinning
rather than for tolerating a known spread.

One trace, one geometry, one host. `qd=1` removes the dependence rather than
bounding it, which is why no claim here rests on the spread staying small.

## Receipts

`qd_jitter.json`. Scorer `qd_jitter.py`; offline, no GPU, no spend.
