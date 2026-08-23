# RESULTS — P2-G2'': the offline gate closes REFUTED on the throttle clause — with the law's substance at 100%

Registered in [PREREG-p2-g2pp.md](PREREG-p2-g2pp.md) (#215). Run 2026-08-23
offline, repo at `f983dd1`. Receipt:
[p2-g2pp-2026-08-23-registered.json](p2-g2pp-2026-08-23-registered.json).

**Scored verdict: REFUTED — and per this prereg's own hard stop, the
offline G2 gate closes here; no third metric correction is registered
against these traces.**

* **(a) convergence: 16/16 traces PASS** — with the equilibrium-metric
  scoring, every capacity-adequate arm reaches §5's sustained predicate at
  step ~0 and holds it through the trace.
* **(b) plateau vs ideal-LRU: 73/73 arms PASS.**
* **(c) equilibrium churn: 16/16 PASS.**
* **Spoilers: 16/16 both fail their must-fail conditions** — informative.
* **(d) throttle gracefulness: 191/219 — REFUTED.** All 28 failures are
  throttled arms; the clause is what closes the gate.

## The (d) anatomy — a measured admission law, and a bound that ignored it

With unthrottled convergence at ~0, the registered bound
(`throttled ≤ 2 × max(unthrottled, 16)`) collapses to **32 steps flat** —
independent of throttle depth. But a PROMO_FRAC throttle admits at most
`frac · m` rows per step, so admitting a working set of `pairs` rows takes

```
t_admit ≈ pairs / (frac · m)
```

steps *by arithmetic*. The measured throttled convergence times track this
law across every failing arm — qwen at 1/16: predicted 240, measured
257–277; granite at 1/16: predicted 67–78, measured 114–117 (the EWMA
settles ~1.2–1.6× past raw admission); olmoe at 1/8: predicted 55–62,
measured 74–89. Failures cluster exactly where the law says they must:
frac = 1/16 (16 arms), 1/8 (7), 1/4 (5, all qwen — the largest
`pairs/m`). Eight of the 28 also fail the fill half: deep throttles push
the deferred backlog into the 128–512 eval window (qwen_code at 1/16:
663 throttled fills vs 20 unthrottled — the same admissions, displaced).

The bound embedded an assumption that unthrottled convergence is
O(discovery) — true under G2's plateau metric, false under the corrected
equilibrium metric where unthrottled converges immediately. (d) was pure
prereg invention (§5 defines no throttle bound), its refutation is the
scored outcome, and the hard stop forbids a third correction. What the
clause *bought* is the measured cost function: **PROMO_FRAC trades
per-step inflation (ε, §5) against admission time `pairs/(frac·m)`,
settle-inflated ~1.2–1.6×** — the calibration input §5's ε-budget choice
was missing.

## What the G2 arc leaves validated

Across G2 → G2' → G2'' the single-pool reuse law's substance ended at
100%: fills ≤ 1.10× ideal-LRU at all 73 arms, `EWMA(fills) = EWMA(novelty)`
at every adequate arm, immediate sustained equilibrium unthrottled, both
spoilers failing everywhere, and the throttle's cost now a formula rather
than a hope. The gate's REFUTED close is a statement about clause (d)'s
bound, and the registered discipline is that it stands. Any future
throttle bound must be derived from the admission law before a G3-era
registration uses one.

No box; program spend unchanged (~$0.77).
