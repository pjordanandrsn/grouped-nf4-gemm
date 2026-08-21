"""Microbenchmark _victim alone, against a realistic tier state.

End-to-end CPU timing could not resolve this change on a laptop: spreads of
6-25% against an expected ~10% effect, with configurations disagreeing on the
sign. Isolating the function removes the reader pool, the arena, the page
cache and the rest of ensure().

Reports MIN of many trials, the standard microbenchmark estimator: noise only
ever adds time, so the minimum is the least-contaminated sample. Median is
shown too, so a change that only moves the median is visible as such.

This measures the cost of ONE FULL SCAN. The batched path that serves most
real calls is measured by count_victim_scans.py instead, because its cost is
a 16-element list walk whose interest is how OFTEN it replaces a scan, not
how long it takes.
"""
import argparse
import os
import statistics
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "kernel"))
from nvme_residency import ColdTier  # noqa: E402


class FakeTier:
    """Just enough state for _victim, borrowed unbound from ColdTier."""
    _victim = ColdTier._victim

    def __init__(self, rows, reclaimable_frac, seed=7):
        import random
        rng = random.Random(seed)
        self.hot_rows = rows
        self._key_of = [(i // 64, i % 64) for i in range(rows)]
        self._reserved = set()
        self._freq = {k: rng.randint(1, 40) for k in self._key_of}
        self._last_use = {k: rng.randint(1, 100000) for k in self._key_of}
        n = int(rows * reclaimable_frac)
        self._reclaimable = {k: 0 for k in rng.sample(self._key_of, n)}
        # _victim consults the batched ranking first. Without this attribute
        # the borrowed method raises AttributeError; WITH it but never reset,
        # every call after the first is served from the cache and this would
        # time a list scan of 16 rather than the full O(hot_rows) sweep it
        # claims to measure -- reporting a large fake speedup. bench() clears
        # it per call for that reason. (Bugbot, gnf4#175.)
        self._victim_batch = None


def bench(rows, frac, calls, trials):
    t = FakeTier(rows, frac)
    excluded = set(range(8))
    best = None
    times = []
    for _ in range(trials):
        c0 = time.process_time()
        for _ in range(calls):
            t._victim_batch = None        # force the full scan, see FakeTier
            best = t._victim(excluded)
        times.append(time.process_time() - c0)
    return min(times), statistics.median(times), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=20000)
    ap.add_argument("--trials", type=int, default=7)
    a = ap.parse_args()
    print(f"{'rows':>5} {'recl%':>6} {'min us/call':>12} {'median':>9} "
          f"{'victim':>7}")
    for rows in (128, 256, 512):
        for frac in (0.0, 0.25):
            mn, md, v = bench(rows, frac, a.calls, a.trials)
            print(f"{rows:>5} {frac*100:>5.0f}% {mn/a.calls*1e6:>12.3f} "
                  f"{md/a.calls*1e6:>9.3f} {v:>7}")


if __name__ == "__main__":
    main()
