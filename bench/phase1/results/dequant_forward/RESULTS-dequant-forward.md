# RESULTS — the training axis vs DEQUANT-ON-FORWARD

**Grades `kernel/prereg_dequant_forward.json`** (OTS-stamped pre-data at commit
`9b48483`, stamp `6d2d02b`). Reduced mechanically by
`bench/phase1/reduce_dequant_forward.py`; the verdict below is that reducer's
output, not a reading of the table.

**VERDICT: NOT CONFIRMED.** `S1` fails. `M1`, `F1`, `Q1`, `Q2`, `Q3` pass.
Nothing was re-tuned, re-run or re-banded after seeing data.

---

## What was measured, and against what

The training axis had only ever been measured against Unsloth's grouped GEMM.
The baseline practitioners with this problem actually run is different: **per
routed expert, call `bitsandbytes.functional.dequantize_4bit` on the packed
weight inside the forward, then `F.linear` on the result.** It is published at
743B by GenON — [`genonai/nf4moe`](https://github.com/genonai/nf4moe),
`QuantizedNaiveMoe`, and the writeup at
[`mncai/nf4moe`](https://huggingface.co/mncai/nf4moe) — and re-derived by every
hand-rolled fused-MoE QLoRA path. This leg measures it.

The arm was written from that source. Same `quant_type="nf4"`, same blocksize
64, dequant per **hit** expert inside the forward, weight materialized and
dropped in the loop body. The module's routing plumbing — one-hot hit mask, the
single `.tolist()` host sync, the per-expert `torch.where` gather, `index_add_`
accumulation — is **not** charged to the baseline, because every arm in this
harness is measured at the KERNEL_CONTRACT op boundary with rows pre-assembled.
Hoisting it makes the baseline *faster* than what GenON published, which is the
conservative direction; its cost is measured separately as `D_routed` and
reported below rather than assumed.

Both arms carry the identical LoRA (`lora_delta_grouped`, r=16, `lora_B`
zero-init, applied pre-activation). Neither uses gradient checkpointing — a
cell is one projection's forward and backward, so "configured identically"
means off in both. Neither applies routing weights. Weights are synthetic.

---

## Verdict against the registered criteria — H100 80GB (sm_90)

| id | criterion | bar | got | |
|---|---|---|---|---|
| **S1** | speed at `decode_m8`, paired `d/g` ≥ 1.0 | ≥ 6 of 8 cells | **5 of 7 live** (1 cell VOID) | **FAIL** |
| S1 band | predicted median 1.3–3.0× | — | **1.070×** | **MISS** |
| **M1** | transient memory D > G at `tokbudget_11800` | ≥ 7 of 8 cells | **8 of 8** | **PASS** |
| M1 band | predicted median 1.5–20× | — | **1.19×** | **MISS** |
| **F1** | `b_rel` G ≤ D, every cell | 24 of 24 | **24 of 24**, median 0.763 | **PASS** |
| F1 band | predicted ratio 0.5–0.9 | — | **0.761–0.765** | **HIT** |
| **Q1** | self-pair in [0.97, 1.03] per cell | ≤ 25% void | **1 of 24 void (4%)** | **PASS** |
| **Q2** | wiring, positive-controlled | no failures | **0 failures** | **PASS** |
| **Q3** | drift in [0.95, 1.05] per cell | — | folded into Q1 | **PASS** |
| S2 | M-axis monotone decay *(report-only)* | — | **not monotone** | **MISS** |
| E1 | J/step *(report-only)* | — | median D/G 1.14 / 1.72 / 1.48 | — |
| P1 | GenON's plumbing *(report-only)* | — | 1.12–1.74× on top | — |

Property suite **49/49**. Wiring smoke passed before any timing. The optional
Unsloth arm was **deliberately not run** — report-only under the prereg, and
`import unsloth` patches torch globally, which is a measurement risk to the two
arms that do carry the verdict.

`DQF_CONFIRMED` also requires two devices; see **Second device** below.

---

## The four things this measured

**1. The M-axis answer is not the one I registered, and it is not monotone.**
Median `d/g` across the census: **1.070× at `decode_m8` → 1.709× at 2048 tokens
→ 1.468× at 11 800 tokens.** I predicted monotone decay, reasoning that GenON's
dequant tax is fixed per step while the GEMM grows with T, so their pattern
should get relatively stronger with batch. Registered as monotone, measured as a
rise then a fall: **MISS.**

The absolute timings say the registered mechanism was right about the top of the
axis and blind to the bottom. Two effects, in opposite directions:

*Below 2048 tokens*, a **near-constant ~0.68–0.82 ms host-side floor** — the
identical `lora_delta_grouped` call both arms make — sits under measurements
that are themselves only 1.15–1.62 ms. Its absolute size barely varies across
four models and two projections, so at `decode_m8` it is roughly half of both
arms and pins the ratio near 1.0. That is why the advantage is *smallest* at
small batch, and it is a property of this harness, not of either kernel.

*Above 2048 tokens*, the registered mechanism does appear, and it is
measurable. Splitting each arm's time into a T-independent part and a per-token
part across the 2048 and 11 800 cells (**post-hoc and descriptive — not a
registered criterion, and it rescues nothing: S1 still fails and S2 is still a
miss**):

| cell | G fixed ms | D fixed ms | D/G |
|---|---:|---:|---:|
| OLMoE gate_up | 1.71 | 4.11 | 2.4× |
| OLMoE down | 1.82 | 4.01 | 2.2× |
| Qwen3-30B gate_up | 2.91 | 7.80 | 2.7× |
| Qwen3-30B down | 3.08 | 6.86 | 2.2× |
| gemma-4 gate_up | 3.20 | 8.30 | 2.6× |
| gemma-4 down | 3.28 | 6.89 | 2.1× |
| gpt-oss gate_up | 5.89 | 9.32 | 1.6× |
| gpt-oss down | 4.61 | 8.29 | 1.8× |

**The dequant-on-forward arm's per-step fixed cost is 1.6–2.7× the fused arm's
in every cell** — 33–54% of its total at 2048 tokens, falling to under 10% at
11 800. That is GenON's ~2.5 s/step dequant tax, in miniature, on a single
projection. Because their fixed cost is the larger one, it amortizes faster, and
the fused arm's advantage compresses as the token budget grows — which is
exactly what S2 predicted, operating on the half of the axis where the floor
does not drown it.

**2. Shape decides the sign at GenON's own token budget.** At 11 800 tokens the
fused arm ranges **0.797× to 1.825×**. OLMoE — the smallest-expert model in the
census, and this repo's long-standing known-loser class — **loses on both
projections** (0.797, 0.861). Every other shape wins (1.078–1.825). Reported
per-cell, not folded into the median, because the split is the useful statistic.

**3. The memory trade is real, and it also decays with batch.** Transient bytes
for one forward+backward, D over G: **2.95–7.73× at `decode_m8`**, falling to
**1.07–1.74× at 11 800 tokens**. The mechanism is not subtle — `F.linear` saves
its weight for backward, so the dequant-on-forward arm holds every hit expert's
materialized bf16 weight across the forward-to-backward window, while gnf4's
autograd Function re-decodes one expert at a time and stores none. At large T
the activations dominate and the held weights stop mattering, which is why M1
passed its bar (8/8) and missed its band (1.19× vs a predicted 1.5–20×).

**4. Fidelity is flat, and it is the cleanest number here.** `b_rel` for the
fused arm is **0.761–0.765× the baseline's across all 24 cells** — measured
against the fp64 exact GEMM on identical dequantized values, subsampled to 16
rows × 32 groups with the caps recorded per cell. The spread across four models,
two projections and three token budgets is 0.004. This is the standing
fp32-accumulation claim, and it survives contact with a comparator it had never
been measured against.

### Two measurement facts that belong in the table, not a footnote

**The shared LoRA floor is 18–66% of the fused arm's time.** Both arms call the
identical `lora_delta_grouped` with identical arguments; whatever it costs is
added to both and **compresses every ratio toward 1.0**. It is largest exactly
where the ratios are smallest (median 0.54 of the fused arm's time at
`decode_m8`, 0.22 at 11 800 tokens). It is measured per cell and reported. It is
**not subtracted from anything** — arithmetic on ratios after the fact is how
instruments get talked into saying what you wanted. A floor-free comparison is a
different experiment and would need its own registration.

**GenON's published routing plumbing costs 1.12–1.74× on top of the same
compute** (median 1.56 at `decode_m8`, 1.19 at 11 800 tokens). That is the
one-hot mask, the `.tolist()` host sync, the gather and the `index_add_` — the
part this leg deliberately did *not* charge to the baseline. The headline
comparison therefore runs against a baseline meaningfully faster than the
published module.

---

## Per-cell — every cell, including the losses and the void

`d/g` > 1 means the fused kernel is faster. `b_rel G/D` < 1 means fused is the
more accurate one. `mem D/G` is transient bytes for one fwd+bwd. VOID means the
cell failed its self-pair and **has no measurement** — the number is withheld,
not reported small.

<!--PER_CELL_TABLE-->

The one void cell (Qwen3-30B `down` at `decode_m8`) failed on `d_selfpair`
1.1145 and `g_drift` 1.1002 — the instrument moved 10% across that cell, so
nothing measured beside it is a measurement. The band was not widened.

---

## Second device

<!--SECOND_DEVICE-->

---

## What can now be claimed that could not before

**Before:** the training axis had one comparator — Unsloth's grouped GEMM at
1.11× median (0.48–1.65, 9/16 cells), one device, exploratory, with per-cell
self-pairs 0.618–1.088 that voided the individual cells.

**Now, on the H100:** against the dequant-on-forward pattern as published,
with an adjacent self-pair that holds in 23 of 24 cells, the fused training path
is **1.47× at the token budget GenON actually trains at** (median, 8/8 cells
live), **1.71× at 2048 tokens**, and **1.07× at batched decode** where it loses
outright on the widest expert. Fidelity is **0.76×** the baseline's error on
every cell, and transient memory is **1.07–7.73×** lower depending on batch.

That is a statement about the thing practitioners are running, not a
translation from a comparison they are not.

**What did not survive:** the registered speed bar at small batch (S1), and both
predicted magnitude bands. The claim is narrowed accordingly, per the prereg's
no-tune clause.

---

## What this does and does not license

- **Licensed:** a comparison against the dequant-on-forward pattern as
  published, at the stated token budgets, on the stated device(s), on synthetic
  weights, at the projection boundary.
- **NOT licensed: any statement about a tuned grouped GEMM.** This baseline is
  not Unsloth's kernel, not `torch._grouped_mm`, not marlin, and not a grouped
  GEMM of any kind.
- **NOT licensed: superseding the Unsloth training comparison.** The 1.11×
  median (0.48–1.65, 9/16 cells, EXPLORATORY, per-cell self-pair 0.618–1.088)
  **remains separate and is not superseded by this leg.** They measure different
  things and must not be conflated or divided into one another.
- **NOT licensed: any end-to-end training-throughput claim.** A cell is one
  projection's forward and backward, not a training step. GenON's 5.5× is an
  end-to-end pipeline number and nothing here is comparable to it.
- **NOT licensed: any claim about real-checkpoint numerics.** Weights are
  synthetic (seeded per-expert `randn * 0.02`). NF4 error is data-dependent; the
  speed comparison is not, and both arms consume identical bytes.
- **Device count, token budgets, and rep count are as stated.** This is a
  **single-rep** leg per device — no cell is replicated within a device, and the
  per-cell self-pair is the instrument check that stands in for replication.

## Receipts

- `H100/dequant_forward_H100.json` — 24 cells, gates, fidelity, memory, energy
- `H100/dequant_forward_smoke.json` — the pre-timing wiring smoke
- `H100/suite.txt` — property suite 49/49 + this leg's 20 CPU tests
- `H100/run.log` — full pod transcript
- `verdicts.json` — the reducer's adjudication
- `per-cell.md` — the table above, as generated
- `SHA256SUMS`
