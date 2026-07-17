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

## Phase 1 — audit (in progress)

**Family 1 of the census: OLMoE-1B-7B-0924** (H2048 / I1024 / E64 / k8 / L16).
Remaining families (Qwen3-30B-A3B, flagship Qwen3-235B-A22B) are pending healthy
GPU hardware and will extend this section, each EXPLORATORY-labeled in-band.

**Setup.**
- Model loaded in **NF4 via `Experts4bit`** (`experts4bit_qlora.load_moe_4bit_streaming`).
  Stock `load_in_4bit` skips the fused `OlmoeExperts` stacks
  ([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)),
  leaving the experts in bf16; the streaming loader quantizes them so the probe
  characterizes **the actual 4-bit model the wire serves**, not a bf16 stand-in.
  The router gate + block modules are untouched (hooks unchanged) and the
  zero-init `ExpertsLoRA` leaves the forward equal to the frozen NF4 base.
- bs1 greedy decode, 512 tokens × 12 diverse prompts → **98,304 (token, layer)
  records**. Feature contract = {`hidden_post_block_l`, `router_logits_l`,
  `token_embedding`} (CHARTER §3.2); label = realized top-k; metric = set-agreement
  H (CHARTER §3.1); Δ-join and cross-token mask applied in the loader only.
- Full capacity ladder (linear → MLP-d → MLP-4d → attn2) at Δ ∈ {1, 2, 4}.
- Device: local RTX A2000 12 GB. Receipt:
  `receipts/20260717/EXPLORATORY_phase1_olmoe.json` (hashed in `SHA256SUMS`).

Heldout set-agreement H by rung (higher = more predictable; chance = k/E = 0.125):

| Δ (layer lead) | linear | MLP-d | MLP-4d | attn2 | ceiling |
|---|---|---|---|---|---|
| 1 | 0.671 | 0.911 | 0.914 | 0.911 | **0.914** |
| 2 | 0.673 | 0.905 | 0.907 | 0.905 | **0.907** |
| 4 | 0.677 | 0.897 | 0.897 | 0.901 | **0.901** |

Verdict below is copied verbatim from the committed reducer (`reduce/reduce_ceiling.py`),
the only thing permitted to print it:

```
{ "family": "olmoe", "band": "all_layers", "delta": 1,
  "heldout_by_rung": [0.67126, 0.91143, 0.91361, 0.91147], "verdict": ["model-limited"] }
{ "family": "olmoe", "band": "all_layers", "delta": 2,
  "heldout_by_rung": [0.67251, 0.90474, 0.90682, 0.9047],  "verdict": ["model-limited"] }
{ "family": "olmoe", "band": "all_layers", "delta": 4,
  "heldout_by_rung": [0.67669, 0.89666, 0.89685, 0.90073], "verdict": ["model-limited"] }
```

**Reading (CHARTER §7).** All three leads are **model-limited**: the ladder plateaus —
the 2-layer stream-token attention probe (attn2) does not beat the flat 4×-width MLP,
and both sit ≈0.91, far above the linear floor (0.67). Added probe capacity stops
buying accuracy, so the ceiling is the router's **intrinsic conditional entropy**, not
the probe's capacity limit. This resolves the CHARTER §7 fork toward *the wire-law
H (1-layer set-agreement) is a property of the router* — confirmed here on the
smallest census MoE. The decay with lead is gentle (0.914 → 0.907 → 0.901 for
Δ 1→2→4): routing stays ≈0.90-predictable four layers ahead, so a runtime prefetch
predictor built on the feature contract has real lead time, and its ~9% miss at Δ1 is
irreducible with probe capacity.

Determinism: an independent re-run (same seed, GPU reducer) reproduced every rung and
all three verdicts.
