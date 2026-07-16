# Router-Predictability Probe Audit — Agent Exploration Charter

**2026-07-16 · Exploratory program spec — NOT a preregistration.** Nothing in this
document is stamped and no bar in it is confirmatory. The charter exists to let a
coding agent build the instruments while the humans keep the testimony: every
decision that could become a number in a published law is frozen here, before the
agent writes a line.

## 1. The question

Is **H = 0.93** (set-agreement, one-layer lead) the router's conditional entropy,
or the probe's capacity limit?

The wire law's operative break-even is H ≈ 0.95; the measured predictor ceiling
is 0.93 — measured with *some* probe. If a deliberately oversized probe finds
0.95 already sitting in the observables, the wire law stands unchanged on gen4
(the bytes arithmetic doesn't care who predicts) but the fine-tune program
becomes unnecessary: the path forward is a runtime predictor, not surgery. If the
ceiling is the model's, router-consistency fine-tuning is the only route to
H ≥ 0.95, and that program gets its own stamped protocol later.

Either way, the prize is not gen4 throughput — B5 already priced the recoverable
idle at the +1.5% neighborhood. The prize is law 4's regime: lead-time routing is
the mechanism that deletes t_s from outside the max before faster links make the
serial tax tens of percent, and a tunable H converts the wire law from a boundary
into a curve the program gets to grade out-of-sample on its other branch.

## 2. Division of labor

**The agent builds instruments; it does not testify.**

| Frozen by human (this document) | Owned by the agent |
|---|---|
| Primary metric and its definition | Capture hooks, serialization, dataloaders |
| Feature contract (what the probe may see) | Probe implementations and the capacity sweep |
| Ceiling-inference procedure | Fixture construction, to spec |
| Fixture pass bands | Orchestration, plots, logs |
| Model census and phase gates | RESULTS drafting (labeled exploratory) |

Standing rule: any output that would be quoted in a law is printed by a committed
reducer reading a committed procedure file — the confirmatory reducer pattern,
applied at exploratory tier. The agent generates the reducer's *inputs*, never
its verdicts.

## 3. Frozen decisions

### 3.1 Primary metric

**Set-agreement at lead Δ = 1**: |predicted top-k ∩ realized top-k| / k, per
token per layer, k as served. Chosen for comparability with the B-series 0.93 —
an agent choosing between set-agreement (0.93) and positional (0.79) would be
choosing the answer.

Logged secondaries, never headlined: positional agreement; **byte-weighted hit
rate** (expert byte sizes as weights — set-agreement approximates it only when
experts are uniform); and the **H = 1 fraction** (shared / always-resident
experts), reported separately because those bytes prefetch unconditionally today.

### 3.2 Feature contract (law 11)

The probe is billed only what a runtime predictor could see at decode step t,
*before* layer l+Δ's router executes:

- post-block hidden state at layer l
- layer-l router logits
- token id / embedding

Enforced **in the dataloader** — features and labels assembled from separately
serialized streams; the eval script never touches raw activations. Decode-time
tokens only (bs1); prefill excluded. Lead distance Δ is a dataloader parameter,
not an eval-time slice.

### 3.3 Ceiling-inference procedure (mechanical)

Capacity ladder per (family, layer band, Δ): linear → 2-layer MLP (width d) →
2-layer MLP (width 4d) → 2-layer attention probe. Verdict printed by
`reduce/reduce_ceiling.py` reading `procedure.yaml`:

- **model-limited** — held-out H gains < 0.005 absolute across each of the top
  two capacity doublings, AND top-rung train H exceeds held-out H by ≥ 0.02
  (capacity demonstrably not binding).
- **probe-limited** — held-out H still gaining ≥ 0.005 at the top rung. Verdict
  is "ceiling not established"; the ladder extends before anything is concluded.
- **runtime-viable** — any rung reaches H ≥ 0.95 held-out at Δ = 1. Its
  inference cost is logged against the serving host's measured router cost
  (logged, not barred) and the runtime-predictor fork is flagged.

The reducer is smoke-tested on fixtures sitting on both sides of the plateau
criterion before the first real capture — verdicts are arithmetic, not judgment.

### 3.4 Fixture gate (Phase 0 exit)

Synthetic router: a planted map from layer-l signal to top-k assignment with
controlled label noise, so Bayes-optimal set-agreement is analytically known.

- Planted levels **0.70 / 0.85 / 0.95** — pipeline must recover each within
  ±0.02 held-out.
- **Null fixture** — labels independent of features; pipeline must read ≤ chance
  + 0.02. This is the leakage alarm: if the null runs hot, the dataloader is
  showing the probe the future, and every downstream receipt is void.

No real checkpoint is touched until the gate reads 4/4.

## 4. Phases

**Phase 0 — instruments** (agent). Hooks, streams, probes, fixtures, reducers.
Exit = fixture gate 4/4, mechanically graded.

**Phase 1 — audit** (agent runs, human reads). H(capacity, Δ ∈ {1, 2, 4},
family) surfaces across the serving census, flagship 235B first. Deliverable:
`RESULTS.md` at exploratory tier plus a per-family reducer verdict. Report-only —
no pass/fail exists at this tier.

**Phase 2 — the fork** (human decision, separate documents).

- *Probe-limited or runtime-viable* → runtime predictor path; the wire-law curve
  gets measured out-of-sample on the zero-copy mechanism (B5), where break-even
  sits lowest.
- *Model-limited* → router-consistency fine-tune (StableMoE-style distillation
  term; routers + predictor head trainable; trunk frozen 4-bit via e4b). The
  agent may build the harness — smoke-tested on toy models only. **The H and
  quality measurements on any fine-tuned checkpoint are outside this charter**:
  they wait for a stamped protocol carrying twin registered bars (H at Δ = 1;
  quality delta on a fixed eval set — the adaptivity being removed may be
  capacity, which is exactly why quality gets a bar instead of an assumption).

## 5. Registration-window protection (law 14)

Bright lines, enforced by the agent's command allowlist:

1. **No H measurement on any non-fixture fine-tuned checkpoint, ever, under this
   charter.** Receipts generated by an eager agent close registration windows
   permanently.
2. Phase-1 receipts carry the exploratory label in-band (filename and header), so
   no future confirmatory can be back-graded onto them.
3. Candidate Phase-2 bars are **deliberately absent** from this document.

## 6. Operations (laws 12, 16, 18)

- **No provisioning authority.** Existing runners only; any cloud launch requires
  per-launch human approval; detached hardcap from create; teardown verified by
  the 404-checking collector. Default cloud budget for Phases 0–1: **$0**.
- **One change per commit.** The sweep matrix varies data and probe capacity,
  never code, within a run.
- Home-card jobs (activation capture, probe training) run **02:00–06:00 only**;
  remote monitors carry ~20-failure tolerance.

## 7. Interpretation table — written before the data

| Phase-1 outcome | Reading | Next |
|---|---|---|
| A fat probe reaches ≥ 0.95 at Δ = 1 | 0.93 was the probe's limit; gen4 wire law unchanged; prefetch viability becomes a runtime-predictor question | Runtime path + wire-curve out-of-sample |
| Held-out plateau at ~0.93 | Model-limited; the entropy is the router's | Draft the fine-tune protocol for stamp |
| Plateau materially **below** 0.93 under the contract-legal feature set | The B-series 0.93 saw features outside the runtime contract — a law-11 correction to a published number | Errata + re-derive break-even |
| H(Δ = 2) within 0.02 of H(Δ = 1) | Prediction horizon is deeper than one layer; the prefetch window can widen | Pipeline-design note, no claim |
| Null fixture hot at any point | Leakage; all downstream receipts void | Stop, fix, re-run Phase 0 |

## 8. Layout

```
router_probe/
  CHARTER.md            # this file
  procedure.yaml        # frozen metric, contract, ladder, criteria
  fixtures/             # planted + null routers, analytic H targets
  capture/              # hooks, stream serialization
  probes/               # ladder implementations
  reduce/reduce_ceiling.py
  receipts/<date>/      # exploratory-labeled, hashed
  RESULTS.md            # exploratory tier
  SHA256SUMS
```

---

The one-line contract: **the agent may generate anything except a verdict.**
Verdicts are arithmetic, printed by committed reducers from frozen procedures —
at every tier, including this one.

---

## 9. Operator amendments (forward-only; the base text above is not edited)

**A1 — 2026-07-16, operator instruction: "cross them autonomously."**

1. **§6 per-launch approval → pre-granted for Phase 1.** The flagship 235B
   capture (~$7, still first per §4) and Qwen3-30B (~$0.50) are approved in
   advance; standard ops rails stay (hardcap from create, 404-verified teardown,
   email on fire/completion). Budget ceiling for Phase-1 captures: $15.
2. **§4 Phase-2 fork → delegated.** The fork is selected mechanically from the
   per-family reducer verdict per the §7 interpretation table; the agent
   proceeds down the selected branch without waiting.
3. **§5 bright line 1 → satisfied by registration, not by waiting.** If the
   model-limited branch is selected, the agent authors the stamped protocol
   (twin registered bars: H at Δ=1; quality delta on a fixed eval set),
   OTS-stamps it BEFORE any fine-tune data exists, and only then runs and
   measures. The prohibition's purpose — never closing a registration window
   with unregistered receipts (law 14) — is preserved; the waiting is removed.
4. Status emails continue at every phase transition (operator instruction of
   the same date).
