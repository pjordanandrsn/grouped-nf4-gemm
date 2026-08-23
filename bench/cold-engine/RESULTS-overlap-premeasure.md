# RESULTS — the routing-overlap pre-measurement: CLOSED, 0/4 — the tail draw is independent

Registered in [PREREG-overlap-premeasure.md](PREREG-overlap-premeasure.md)
(#232; bar frozen before the statistic existed). Run 2026-08-23 on the 16
committed rank traces, deterministic, offline. Receipt:
[routing-trace/overlap-2026-08-23.json](routing-trace/overlap-2026-08-23.json).

**Verdict: CLOSED by the registered bar — 0 of 4 models reach R ≤ 0.90 at
f = 0.66, m = 4.** The measured R across every (model × fraction) cell:

| model | f=0.50 | f=0.66 | f=0.75 |
|---|---|---|---|
| granite | 0.997 | 0.998 | 0.998 |
| olmoe | 0.999 | 0.998 | 1.001 |
| qwen | 1.001 | 1.001 | 0.999 |
| gptoss | 0.997 | 0.996 | 0.993 |

Twelve cells, R ∈ [0.993, 1.001]: **once per-request popularity is
conditioned away — and placement already exploits popularity fully — the
DRAM-tail routing of different requests is statistically independent** to
within a fraction of a percent. The m = 2 pair spreads (0.988–1.003) say
choosing WHICH requests to co-batch is worth ≤ ~1% of tail uniques; the
m = 4 composition value lands at −6 to +32 µs/step against a ~23 ms wall.
No scheduler can group what does not correlate; the bar anticipated
exactly this closure and it fired.

## What this closes, and the state of the B=16 wall

The uniques-reduction tree is now fully disposed, each branch by its own
registered method: **placement** (monotonicity proof — the greedy already
minimizes expected uniques), **re-ranking** (the same proof — a
distinction without a difference), **batch composition** (this
measurement — the correlation it needs does not exist). The 58 µs/unique
decode bill (~10.9 of the 22.9 ms wall) is therefore irreducible by
scheduling at fixed VRAM; what remains against the wall, both previously
priced: **VRAM slot growth** (~17–35 µs/step per slot — the FP8
workstream's serving payoff) and the **2.4 µs/row multi-expert
interaction term** (bounded, three candidates named in
[RESULTS-p4-cellmodel.md](RESULTS-p4-cellmodel.md)) — its own
registration or none.
