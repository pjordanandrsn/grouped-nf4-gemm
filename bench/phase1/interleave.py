#!/usr/bin/env python3
"""Iteration-level interleaved pairing.

Every leg before this one timed a whole block of A, then a whole block of B,
and divided. That exposes the ratio to any drift on a timescale longer than a
block, and two legs died of it:

  * leg 1 lost a whole device when the cell after a fixture build measured the
    GPU clocking back up (amendment 1);
  * leg 2's amendment 2 made blocks LONGER to meet the registered 250 ms target
    and made the instrument WORSE -- 32/32 live at 89 ms blocks became 26/32 at
    258 ms, and median |1 - self-pair| doubled from 0.0012 to 0.0027 -- while
    the ratios themselves moved by a median of 0.999. Block length was never
    the binding constraint. Drift BETWEEN blocks is.

This module pairs at the iteration, not the block: one call of A, one call of
B, ratio, repeat. Each ratio is formed from two calls a few milliseconds apart,
so any drift slower than a single pair cancels inside the pair instead of
accumulating across a cell.

Two details that make it honest rather than merely finer-grained:

  * ORDER ALTERNATES. Pairs run A,B then B,A. If A always ran first it would
    always eat whatever the first call of a pair costs -- cache state, clock
    ramp, launch-queue position -- and every pair would carry the same bias.
  * NO PER-CALL SYNC. Events are recorded on the stream around each call and
    read after ONE synchronize at the end. Synchronising per call would inject
    a stall into every measurement; recording on the stream measures device
    time and leaves the pipeline alone.

The statistic is the MEDIAN OF PER-PAIR RATIOS, not the ratio of medians. For
paired data those differ, and only the former has the drift-cancelling property
this module exists for.
"""
from __future__ import annotations

import statistics as st


def interleaved_pairs(fa, fb, pairs: int, warm: int = 10, torch_mod=None):
    """Time `fa` and `fb` alternately, one call each, `pairs` times.

    Returns (ta, tb, orders) -- per-call device milliseconds and the order each
    pair ran in. `torch_mod` is injectable so the reduction below can be tested
    without a GPU."""
    torch = torch_mod
    if torch is None:  # pragma: no cover - trivial import indirection
        import torch as torch

    for _ in range(warm):
        fa()
        fb()
    torch.cuda.synchronize()

    ev = [tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
          for _ in range(pairs)]
    orders = []
    for i in range(pairs):
        a0, a1, b0, b1 = ev[i]
        if i % 2:
            b0.record()
            fb()
            b1.record()
            a0.record()
            fa()
            a1.record()
            orders.append("ba")
        else:
            a0.record()
            fa()
            a1.record()
            b0.record()
            fb()
            b1.record()
            orders.append("ab")
    torch.cuda.synchronize()
    ta = [e[0].elapsed_time(e[1]) for e in ev]
    tb = [e[2].elapsed_time(e[3]) for e in ev]
    return ta, tb, orders


def pair_stats(ta, tb, orders=None):
    """Reduce per-call times to the paired statistics the gates read.

    `ratio_median` is the median of per-pair b/a. `halves_ratio` compares the
    median of the first half of the pairs to the second: it is the drift check,
    but applied to the RATIO rather than to a raw timing, because after
    interleaving a drifting box no longer threatens the ratio and only a
    drifting *ratio* does. `order_bias` is the a-first median over the b-first
    median -- a control on the alternation, which should read ~1."""
    r = [b / a for a, b in zip(ta, tb) if a > 0]
    n = len(r)
    if n < 4:
        return {"n": n, "ratio_median": None}
    h = n // 2
    q = sorted(r)
    out = {
        "n": n,
        "ratio_median": st.median(r),
        "ratio_iqr": q[int(0.75 * n)] - q[int(0.25 * n)],
        "ratio_p05": q[int(0.05 * n)],
        "ratio_p95": q[int(0.95 * n)],
        "halves_first": st.median(r[:h]),
        "halves_second": st.median(r[h:]),
        "ms_a_median": st.median(ta),
        "ms_b_median": st.median(tb),
    }
    out["halves_ratio"] = out["halves_second"] / out["halves_first"]
    if orders:
        ab = [x for x, o in zip(r, orders) if o == "ab"]
        ba = [x for x, o in zip(r, orders) if o == "ba"]
        if len(ab) >= 2 and len(ba) >= 2:
            out["order_bias"] = st.median(ab) / st.median(ba)
    return out


def block_ratio(ta, tb):
    """The OLD statistic, for contrast: median of all A against median of all
    B, as if each had been timed as its own block. Kept so a receipt can show
    both and a reader can see what the pairing bought on that cell."""
    return st.median(tb) / st.median(ta)
