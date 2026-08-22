# The cache is LRU, and LRU is ~1.9× off optimal — where the remaining wall is

Receipt: [`oracle-headroom.json`](routing-trace/oracle-headroom.json). Harness:
[`oracle_headroom.py`](routing-trace/oracle_headroom.py). 48 cells — four
models × four prompts × three capacities. No box.

The gpt-oss traces are read from
[`wall-real-routing-2026-08-21/`](wall-real-routing-2026-08-21/) where they
were captured, via `--dir`, rather than copied.

[`RESULTS-wall-real-routing.md`](RESULTS-wall-real-routing.md) established
that wall is **transfer-bound** (r = +0.9872 on real routing, replicated on
two hosts). So the only lever on wall is transfers, and the question that
decides whether more policy work pays is how far the shipped cache is from the
best any policy could do at the same capacity.

Belady's MIN answers it exactly: evict the resident key whose next use is
furthest away. No online policy can beat it, and it needs the whole trace —
which offline replay has.

## The gap

| | median | min | max |
|---|---|---|---|
| **cache ÷ optimal** | **1.90×** | 1.33× | 4.05× |
| LRU ÷ optimal | 1.96× | | |
| **cache ÷ LRU** | **0.995×** | | |

The shipped `DevRowCache` makes about **twice** the transfers an optimal
policy would, on every model measured. And it is, to within half a percent,
**plain LRU**:

| steps_held | cache ÷ LRU |
|---|---|
| 1.0 | **0.870×** |
| 1.5 | 0.997× |
| 2.0 | 0.996× |

Its entire advantage over LRU lives at *exactly* one step, where resurrection
and LFU-then-LRU victim choice buy 13%. At any larger capacity the two are
indistinguishable. That is not a criticism of the implementation — it is a
statement about where the remaining opportunity is, and it is not in the
mechanisms this program has been measuring.

## What the gap is worth in wall

Fitting the captured-routing arms of the wall receipt:

```
wall_ms = 0.357 × transfers/step + 0.405
```

A fixed per-step floor of **0.405 ms** and **0.357 ms** per transfer. Closing
the transfer gap entirely would be worth:

| regime | transfers/step | wall now | wall at optimum | saving |
|---|---|---|---|---|
| tight (730 fills) | 2.85 → 1.50 | 1.424 ms | 0.941 ms | **−33.9%** |
| mid (469 fills) | 1.83 → 0.96 | 1.059 ms | 0.750 ms | −29.2% |
| loose (72 fills) | 0.28 → 0.15 | 0.505 ms | 0.458 ms | −9.4% |

**Up to a third of wall, in the regime where the tier is actually under
pressure** — which is the regime the whole cold-tier design exists for. At
loose capacity the fixed floor dominates and there is little to win, which is
also why the wall experiment saw ~1% differences at rows=32 despite a 72%
transfer cut.

## Three things this is not

**Belady is not a proposal.** It cannot be implemented online; it is here to
size the opportunity. How much an implementable policy could recover is a
separate question this does not answer, and the honest expectation is a
fraction, not all of it.

**The 4.05× outlier is the degenerate trace.** Qwen's math prompt is the
period-2 repetition loop documented in
[`routing-trace/RESULTS-third-model.md`](routing-trace/RESULTS-third-model.md);
its extreme ratio should not be read as headroom on real decoding.
Excluding all three of its cells moves the median from **1.90× to 1.85×**
over the remaining 45, so it is not what carries the result.

**This does not contradict the R10 or wall results.** R10 asked whether
*reclaimable residency* cuts refills — refuted on four models. This asks
whether *any* policy could cut them, and answers yes, by about half. Those are
different questions and the second is where the measured wall is.

## What it implies for direction

The mechanisms audited so far — the one-step threshold, the zero-hit region,
the headroom rule — are
[structurally forced](RESULTS-verdict-audit.md); no routing can
refute them, so no capture can test them. Reclaimable residency is refuted on
four models. Resurrection adds nothing to wall once transfers are accounted
for.

What is left, measured rather than argued, is a **~2× transfer gap against a
policy that is currently LRU**, worth up to a third of wall under pressure.
That is the largest quantified lever remaining in this line of work, and it is
a replacement-policy question, not a residency-mechanism one.
