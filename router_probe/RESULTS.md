# Router-Predictability Probe — RESULTS (EXPLORATORY TIER)

**Exploratory, NOT a confirmatory. No bar here is stamped.** Per CHARTER §2, every
number that could enter a law is printed by `reduce/reduce_ceiling.py` reading
`procedure.yaml` — this document quotes the reducer, it does not adjudicate.

## Phase 0 — instrument gate (CHARTER §3.4)

The gate proves the *pipeline* recovers a known answer before any real checkpoint
is touched (CHARTER §5 bright line 1). It is not a result about any model.

- Fixtures: 3 planted (temperature-noise, signal-as-logits-stream; see
  `fixtures/planted.py` DESIGN HISTORY for the two dead constructions) at analytic
  H ∈ {0.70, 0.85, 0.95} + 1 null (labels ⊥ features, chance = k/E = 0.125).
- Full ladder (linear → MLP-d → MLP-4d → attn2) on each; recovery band ±0.02;
  null margin +0.02.
- Verdict below is copied verbatim from the committed reducer
  (`reduce_ceiling.py gate`), the only thing permitted to print it.

```
gate: 4/4  exit_phase0: true  (committed reducer, reduce_ceiling.py gate)
  planted_70   target=0.7002  best_heldout=0.6996  pass=True
  planted_85   target=0.8500  best_heldout=0.8470  pass=True
  planted_95   target=0.9500  best_heldout=0.9376  pass=True
  null         target=0.1250  best_heldout=0.1267  pass=True  leakage_alarm=False
```

**Gate PASSED 2026-07-16** — pipeline recovers the planted levels within ±0.02 and the null reads chance (no leakage). Phase 1 unlocked.

Reducer self-check (CHARTER §3.3, done before first real data): the ceiling
reducer was exercised on four crafted ladders spanning the plateau criterion —
model-limited, probe-limited, runtime-viable, and plateau-without-gap — and
classified each per the frozen arithmetic. Logged in the commit message.

## Phase 1 — audit

NOT STARTED. Requires the gate at 4/4 (this document's Phase 0 block) and, per
CHARTER §6, per-launch human approval for any GPU capture — default cloud budget
for Phases 0–1 is $0. Capture instrument (`capture/hooks.py`) is built and
fixture-smoke-tested through the shared serialization format, but has touched no
real checkpoint (CHARTER §5 bright line 1).

When it runs: H(capacity, Δ ∈ {1,2,4}, family) across the serving census,
flagship 235B first; per-family reducer verdict against the CHARTER §7
interpretation table; receipts carry the EXPLORATORY label in-band.
