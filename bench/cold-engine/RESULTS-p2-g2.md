# RESULTS — P2-G2: REFUTED — the partitioned persistent pool, not retention, is what fails

Registered in [PREREG-p2-g2.md](PREREG-p2-g2.md) (#212); bars frozen in
spec §11 (#211) before the harness existed. Run 2026-08-23 offline on the
16 committed rank traces, repo at `0b72cc2` + the set-path correction, cold
start, real `DevRowCache`. Receipt:
[p2-g2-2026-08-23-registered.json](p2-g2-2026-08-23-registered.json).

**Verdict: REFUTED** — (a) 5/16 traces converge ≤ 64 steps at every
capacity; (b) 16/56 arms within 1.10× ideal-LRU; (c) 4/16 traces meet the
churn bound; (d) 166/168 throttle arms graceful. **Both spoilers failed
their must-fail conditions on all 16 traces** (margin-trap thrash;
no-retention at all-miss), so the instrument distinguishes and the
refutation is informative.

## The anatomy — three mechanisms, all in the parameterization

**1. The static persistent reservation starves the transient half.** The
registered split reserves rows/4 for the persistent pool. At
granite_code @ 1024 rows (working set 1,067 pairs — nearly fits), ideal-LRU
makes **20** fills over the eval window; the law makes **1,243 — 62×** —
because the transient half has only 768 rows against ~800 unprotected-hot
keys and churns continuously. qwen_math @ 1024: 25.7×. The (b) failures
concentrate exactly where the reservation binds (mid/high capacities,
ratios 1.2–3.2); at capacities where the persistent pool stays empty the
law runs at **1.00–1.06× LRU** (qwen ≤ 512: 1.00/1.04/1.03/1.06) — the
transient law itself is LRU-class, consistent with the program's earlier
cache result (0.995× LRU).

**2. Bulk PERIOD promotions destabilize late.** Every convergence failure
(236–361 steps) sits at a capacity whose persistent pool actually fills
(`pers = 127–256`); promotions land in batches of up to 256 keys at PERIOD
boundaries, shifting the hit pattern each time, and θ-demotions dump keys
back cold two periods later. Where persistence never engages, convergence
is 0–43 steps.

**3. Clause (c) was mis-registered.** "Largest applicable capacity" was
meant to be capacity-adequate, but 12/16 traces have working sets larger
than the 1,024-row sweep ceiling — their max-cap churn measures capacity
misses (EWMA fills 20–44/step), not the law's equilibrium. The 4 passes are
exactly the arms near adequacy. A registration flaw, disclosed as such: the
clause should read *capacity-adequate arms only* (pairs ≤ 0.9 × rows).

## What is clean

Retention economics and the fast half work as derived: the transient pool +
SMOOTH_CAP budget + burst accounting matches ideal-LRU wherever the
partition doesn't bind, converges in single-digit steps there, throttles
gracefully (166/168), and both falsifiability arms fail exactly as
registered on every trace.

## Disclosures

* **A wrong-set run happened first and is void**: the prereg's path
  parenthetical pointed at the older 12-trace `routing-trace/` set while
  naming "the 16 committed rank traces" twice; the run against the wrong
  set ([p2-g2-2026-08-23-wrongset-void.json](p2-g2-2026-08-23-wrongset-void.json))
  showed the same clause-level shape (REFUTED a/b/c, spoilers fail) before
  the error was caught. The named set governs; the registered run above is
  the scored one.
* §11's spoiler shorthand "`protected = rows`" was uninstantiable (the
  constructor rejects it) and named I1's *unservable* side; the runnable
  thrash spoiler (`protected = 1`, margin ≫ k) was corrected in the prereg
  before any run (#212).

## The derived fix (to re-register, not tuned here)

Per the registered miss protocol, the fix is derived from the anatomy and
goes back through spec → prereg → run:

1. **No partition.** The persistent pool becomes a *protection attribute*
   inside one physical pool — which is exactly the semantics the shipped
   `DevRowCache` already has (`protected`) — so unused persistence is
   transient capacity, and the 62× regime cannot exist.
2. **Trickle promotion.** Persistent status accrues ≤ 1–2 keys/step (age-
   and hits-qualified as now) instead of PERIOD-boundary batches, removing
   the late-trace pattern shifts.
3. **Clause (c) scoped to capacity-adequate arms** (pairs ≤ 0.9 × rows),
   with the sweep extended so every trace has at least one adequate arm.

No box was rented for this gate; total program spend remains ~$0.77.
