# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage 3 / gate 3 — observed reuse, and the four classes it distinguishes.

Gate 3 closes the loop the earlier gates left open:

    placement -> execution -> observed cost/reuse -> new placement

To move a placement boundary from evidence you first have to name the
evidence. The directive names four behaviours a cold expert can show, and
they want different answers:

    one-shot     routed once and not again      leave it on NVMe
    burst        clustered reuse, then silence  reclaimable DRAM is enough
    warm         reused across the whole run    deserves protected DRAM
    hot          reused constantly              deserves VRAM

**R4 is the load-bearing question and this module is built to answer it, not
to assume it.** The prediction is that SHORT-WINDOW RECURRENCE predicts
reuse-before-overwrite better than long-run frequency — that an expert which
is globally cold but locally hot is worth retaining, and one that is
uniformly warm is not. So both predictors are computed and kept side by
side; `classify` uses recency, and `predictor_scores` reports what each one
would have said, which is what makes the comparison a measurement rather
than a design assumption.

Pure counting over an event stream: no torch, no clock, no tier coupling.
The tier calls `observe`; anything that can replay a routing trace can drive
it offline.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import NamedTuple

ONE_SHOT, BURST, WARM, HOT = "one-shot", "burst", "warm", "hot"


class Stats(NamedTuple):
    picks: int                   # total times routed
    first_tick: int
    last_tick: int
    resurrections: int           # reused after losing capacity ownership
    max_run_picks: int           # most picks inside one recency window
    windows_seen: int            # distinct windows it appeared in


class ReuseProfile:
    """Per-expert reuse evidence, accumulated from tier events.

    ``window`` is the recency horizon in ticks — the span over which
    "locally hot" is judged. It is the one tuning knob and it is explicit
    rather than implied: a burst is only a burst relative to a window, and a
    module that hid that choice would be smuggling the answer to R4 into a
    constant.
    """

    def __init__(self, window: int = 64):
        if window < 2:
            raise ValueError("window must be >= 2 ticks")
        self.window = int(window)
        self._picks: dict = defaultdict(int)
        self._first: dict = {}
        self._last: dict = {}
        self._res: dict = defaultdict(int)
        self._recent: dict = defaultdict(deque)   # ticks inside the window
        self._max_run: dict = defaultdict(int)
        self._windows: dict = defaultdict(int)
        self._last_window: dict = {}
        self.ticks = 0

    # ------------------------------------------------------------ events --
    def observe(self, key, tick: int, *, resurrected: bool = False) -> None:
        """One routing of ``key`` at ``tick``. ``resurrected`` marks that the
        row was reused after losing capacity ownership — the event R1
        counts, and the ground truth R4's predictors are scored against."""
        self.ticks = max(self.ticks, int(tick))
        self._picks[key] += 1
        self._first.setdefault(key, tick)
        self._last[key] = tick
        if resurrected:
            self._res[key] += 1
        d = self._recent[key]
        d.append(tick)
        while d and tick - d[0] >= self.window:
            d.popleft()
        if len(d) > self._max_run[key]:
            self._max_run[key] = len(d)
        w = tick // self.window
        if self._last_window.get(key) != w:
            self._last_window[key] = w
            self._windows[key] += 1

    def stats(self, key) -> Stats:
        return Stats(picks=self._picks.get(key, 0),
                     first_tick=self._first.get(key, -1),
                     last_tick=self._last.get(key, -1),
                     resurrections=self._res.get(key, 0),
                     max_run_picks=self._max_run.get(key, 0),
                     windows_seen=self._windows.get(key, 0))

    # ----------------------------------------------------------- classify --
    def classify(self, key, *, burst_min: int = 3, hot_frac: float = 0.5,
                 warm_windows: int = 3) -> str:
        """Which of the four behaviours this expert has shown.

        Ordered most-specific first, and deliberately conservative: an
        expert is only called HOT if it appears in at least ``hot_frac`` of
        all windows elapsed, because promoting on thin evidence is how a
        feedback loop starts thrashing.
        """
        s = self.stats(key)
        if s.picks <= 1:
            return ONE_SHOT
        total_windows = max(1, self.ticks // self.window + 1)
        if s.windows_seen >= max(warm_windows, hot_frac * total_windows):
            return HOT
        if s.max_run_picks >= burst_min and s.windows_seen <= warm_windows:
            return BURST
        return WARM

    # --------------------------------------------------------- the R4 test --
    def predictor_scores(self, keys=None) -> dict:
        """How well each predictor ranks experts by observed resurrections.

        R4 says recency (``max_run_picks``) should beat long-run frequency
        (``picks``). Reported as Spearman-style rank agreement against the
        resurrection count each expert actually accumulated, so the claim is
        settled by the trace rather than by argument.

        Returns ``None`` for a correlation that is undefined (fewer than two
        experts, or no resurrections at all) rather than a misleading 0.0.
        """
        ks = list(keys if keys is not None else self._picks)
        if len(ks) < 2:
            return {"n": len(ks), "recency": None, "frequency": None}
        truth = [self._res.get(k, 0) for k in ks]
        if not any(truth):
            return {"n": len(ks), "recency": None, "frequency": None,
                    "note": "no resurrections observed; nothing to rank"}
        return {"n": len(ks),
                "recency": _spearman([self._max_run.get(k, 0) for k in ks], truth),
                "frequency": _spearman([self._picks.get(k, 0) for k in ks], truth)}

    # -------------------------------------------------------- promotion --
    def candidates(self, *, tier: str) -> list:
        """Experts whose observed behaviour argues for ``tier``.

        Ranked by the evidence that justifies the move, not by raw
        frequency: DRAM promotion is argued by resurrections (bytes that
        were retained and reused), VRAM promotion by sustained presence.
        """
        if tier not in ("dram", "vram"):
            raise ValueError("tier must be 'dram' or 'vram'")
        want = HOT if tier == "vram" else BURST
        out = [(k, self.stats(k)) for k in self._picks
               if self.classify(k) == want]
        out.sort(key=lambda t: (-(t[1].resurrections if tier == "dram"
                                  else t[1].windows_seen), -t[1].picks))
        return out


def _spearman(a, b) -> float:
    """Rank correlation. Ties get average ranks, which matters here because
    routing counts tie constantly."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (num / (da * db)) if da and db else 0.0
