# PREREG — P2-G2'': the equilibrium-metric scoring of the single-pool law

Registered before any G2''-scored run, from G2''s anatomy and §5's frozen
text alone ([RESULTS-p2-g2p.md](RESULTS-p2-g2p.md); spec §11 G2'' entry).
Identical replay to G2' — same traces, sweep with the above-pairs adequacy
rule, PROMO_FRAC arms, spoilers, parameters — with the scoring corrected to
the spec's own definitions:

* **(a) Convergence = time-to-sustained-equilibrium**: the first step t at
  which §5's predicate `EWMA(fills) ≤ (1 + η)·EWMA(novelty) + 1.0`
  (α = 1/16, EWMAs from step 0) holds **and holds at every later step of
  the trace**. Bar: ≤ 64 at every swept capacity, on ≥ 14/16 traces.
* **(b)**: eval-window fills ≤ 1.10 × same-capacity ideal-LRU **+ m** at
  every arm (m = one step's routed set — the registered absolute guard for
  microscopic denominators; G2''s sole (b) failure was 23 vs 20 fills).
* **(c)**: unchanged from G2' (steady-state predicate over steps 256–512 at
  every capacity-adequate arm).
* **(d)**: convergence half re-based on the corrected (a) (throttled ≤ 2 ×
  max(unthrottled, 16)); fill half ≤ 1.05 × unthrottled **+ m**.

PASS iff (a) ∧ (b) ∧ (c) ∧ (d); spoilers as in G2' (margin trap and
no-retention), both must fail their unchanged conditions. Refuted at any
clause ⇒ per-clause report and the offline gate closes REFUTED — no third
metric correction will be registered against these traces; the law would
return to the spec for re-derivation instead.

The replay is deterministic, so this is a re-scoring discipline exercise:
the harness re-runs fresh at this registration's commit rather than
re-reading G2''s receipt, so the receipt and its scoring share provenance.
