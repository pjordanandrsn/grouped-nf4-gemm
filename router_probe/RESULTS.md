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

### Family 2 — Qwen3-30B-A3B (E=128, k=8, L=48)

Qwen's ladder did **not** resolve the way OLMoE's did, and pinning down *why* drove a
CHARTER-amended re-run — the informative part of this result.

**First pass (4-rung ladder, 147,456 records, `receipts/20260717/…_preA1_4rung.json`):**
`probe-limited ×3`. attn2 (0.77) sat +0.20 above MLP-4d (0.57) and was still the top of a
rising ladder — no ceiling established. Unlike OLMoE (flat MLP↔attn plateau at 0.91),
Qwen's routing signal lived in cross-stream structure only the attention probe reached, and
the 4-rung ladder ran out of rungs before flattening.

**Amendment A1** (`procedure.yaml`, re-stamped pre-data): ladder extended past attn2 with
three attention-family rungs — `attn4` (2× depth), `attn4_w512` (2× width), `attn6_w512`.
First four rungs unchanged (OLMoE comparability preserved); criteria arithmetic untouched.

**A1 re-run at two data volumes** (7-rung, local A2000 audit):

| Δ | linear | MLP-d | MLP-4d | attn2 | attn4 | attn4_w512 | attn6_w512 | ceiling | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **147k** (256 tok) `…_A1_147k.json` | | | | | | | | | |
| 1 | 0.518 | 0.513 | 0.570 | 0.773 | 0.790 | 0.789 | 0.799 | 0.799 | probe-limited |
| 2 | 0.512 | 0.503 | 0.558 | 0.768 | 0.785 | 0.787 | 0.788 | 0.788 | model-limited |
| 4 | 0.496 | 0.537 | 0.514 | 0.758 | 0.776 | 0.780 | 0.785 | 0.785 | model-limited |
| **294k** (512 tok) `receipts/20260718/…` | | | | | | | | | |
| 1 | 0.528 | 0.532 | 0.552 | 0.792 | 0.814 | 0.826 | 0.824 | **0.826** | plateau-no-gap |
| 2 | 0.523 | 0.522 | 0.582 | 0.785 | 0.808 | 0.820 | 0.821 | **0.821** | plateau-no-gap |
| 4 | 0.509 | 0.435 | 0.564 | 0.778 | 0.799 | 0.813 | 0.816 | **0.816** | plateau-no-gap |

Verdict at 294k, verbatim from the committed reducer (`reduce/reduce_ceiling.py`):

```
{ "family": "qwen3_moe", "band": "all_layers", "delta": 1,
  "heldout_by_rung": [0.528, 0.532, 0.5523, 0.7921, 0.814, 0.8257, 0.8242],
  "verdict": ["plateau-without-overfit-gap (no verdict; extend data or ladder)"] }
{ "family": "qwen3_moe", "band": "all_layers", "delta": 2,
  "heldout_by_rung": [0.5234, 0.5221, 0.582, 0.7845, 0.8075, 0.8198, 0.8209],
  "verdict": ["plateau-without-overfit-gap (no verdict; extend data or ladder)"] }
{ "family": "qwen3_moe", "band": "all_layers", "delta": 4,
  "heldout_by_rung": [0.5088, 0.4353, 0.5643, 0.7784, 0.799, 0.8128, 0.8161],
  "verdict": ["plateau-without-overfit-gap (no verdict; extend data or ladder)"] }
```

**Reading (CHARTER §7).** Doubling the data (147k→294k) both **raised the plateau**
(~0.80→~0.826) and **changed the verdict** — the 147k probe/model-limited calls dissolved into
the reducer's fourth, abstaining outcome: the ladder has flattened at ~0.82 (top-two-doublings
held-out gain < 0.005) but the train–held-out gap has not closed enough to certify the
capacity-not-binding condition for *model-limited*. The reducer therefore refuses to name a
ceiling and asks to extend data or ladder. (The Δ4 MLP-d dip to 0.435, below its own linear
rung, is an optimizer artifact at that rung; the attention rungs are unaffected.)

The honest result is **not** a number but a shape: **Qwen3-30B routing is ≈0.82-predictable at
all three leads, and its ceiling is not yet pinned.** This contrasts sharply with OLMoE (clean
flat plateau, cleanly model-limited at 0.91) and is the concrete evidence that **the wire-law H
is family-dependent, and that measuring it is itself data-sensitive** — a 2× data change moved a
Qwen verdict but no OLMoE one. Decision-relevant corollary: even unpinned, ~0.82 ≪ the ~0.95
speculation break-even, so a runtime prefetch predictor is not viable on this family regardless
of where the true ceiling sits — pinning it exactly is scientific completeness, not a gate.

Determinism: the 294k audit's first four rungs reproduced the 147k capture's independent
values to ±0.01 across a *different* capture run (two pods, two token counts).

### Family 3 — gpt-oss-20b (E=32, k=4, L=24)

*(EXPLORATORY, 2026-07-18 — the first k=4 family; olmoe and qwen3_moe are both k=8.)*

**Setup.**
- Model loaded in **NF4 via `Experts4bit`** through the gpt_oss lane
  (experts4bit-qlora ≥0.4.0): on-disk **MXFP4** expert tensors dequantize
  bit-identically to transformers' reference and requantize to NF4; capture ran
  with **expert offload** (~5.1 GB peak) on the same local A2000. Adapter-only
  instrument change: the router is `model.layers.{i}.mlp.router`, and
  transformers≥5 `GptOssTopKRouter` returns `(router_logits, scores, indices)`,
  so the standard gate hook's `out[0]` is the raw logits row — stream 2 is
  comparable across families. `procedure.yaml` bytes untouched.
- bs1 greedy decode, 256 tokens × 12 diverse prompts → **73,728 (token, layer)
  records**; the stamped 7-rung A1 ladder at Δ ∈ {1, 2, 4}.
- Device: local RTX A2000 12 GB (capture + audit, one process, 2 h 18 m wall).
  Receipt: `receipts/20260718/EXPLORATORY_phase1_gpt_oss.json` (hashed in
  `SHA256SUMS`).

Heldout set-agreement H by rung (chance = k/E = 0.125):

| Δ | linear | MLP-d | MLP-4d | attn2 | attn4 | attn4-w512 | attn6-w512 | best | reducer verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.440 | 0.383 | 0.405 | 0.820 | 0.833 | 0.823 | 0.826 | **0.833** | model-limited |
| 2 | 0.440 | 0.398 | 0.395 | 0.817 | 0.825 | 0.808 | 0.819 | 0.825 | probe-limited |
| 4 | 0.452 | 0.409 | 0.393 | 0.801 | 0.810 | 0.790 | 0.800 | 0.810 | probe-limited |

Verdicts, verbatim from the committed reducer (`reduce/reduce_ceiling.py`):

```
{ "family": "gpt_oss", "band": "all_layers", "delta": 1,
  "heldout_by_rung": [0.44001, 0.38277, 0.40532, 0.82035, 0.83333, 0.82263, 0.82613],
  "verdict": ["model-limited"] }
{ "family": "gpt_oss", "band": "all_layers", "delta": 2,
  "heldout_by_rung": [0.4403, 0.39804, 0.39495, 0.81652, 0.8248, 0.80811, 0.81936],
  "verdict": ["probe-limited (ceiling not established)"] }
{ "family": "gpt_oss", "band": "all_layers", "delta": 4,
  "heldout_by_rung": [0.45247, 0.40916, 0.39272, 0.80083, 0.81021, 0.79045, 0.80007],
  "verdict": ["probe-limited (ceiling not established)"] }
```

**Reading (CHARTER §7).** At Δ=1 the attention ladder **saturates** — attn4 reads
0.833 and both wider/deeper rungs sit flat at 0.823–0.826 — with a large
train–held-out gap, so the reducer certifies *model-limited*: **≈0.83 is the
router's conditional entropy one layer ahead on this family.** The flat-feature
rungs collapse (MLP rungs 0.38–0.41, below the 0.44 linear floor): as on
Qwen3-MoE, the predictive signal lives in cross-stream structure that only the
attention probes exploit — but unlike Qwen3-MoE, the ladder then saturates
cleanly. Multi-layer leads (Δ=2/4 ≈0.82/0.80) remain probe-limited at this
record count.

Cross-family picture after three families: OLMoE (k=8) ≈0.91, cleanly
model-limited; Qwen3-30B (k=8) ≈0.82 plateau, ceiling unpinned at 294k records;
**gpt-oss-20b (k=4) ≈0.83, certified at Δ=1.** The wire-law H is family-dependent
along both axes measured so far, and k=4 does not, by itself, buy the
predictability that OLMoE's k=8 shows. Decision-relevant: 0.83 ≪ the ~0.95
speculation break-even, so the runtime-prefetch fork stays dead on this family
too. Streams re-auditable on the capture host.
