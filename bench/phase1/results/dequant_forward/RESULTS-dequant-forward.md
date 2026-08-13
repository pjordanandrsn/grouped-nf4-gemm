# RESULTS — the training axis vs DEQUANT-ON-FORWARD

**Grades `kernel/prereg_dequant_forward.json`**, OTS-stamped pre-data at
`9b48483` (stamp `6d2d02b`; the pre-amendment attestation is kept as
`.pre-amendment1.ots`), amendment 1 stamped pre-data for run 2 at `918456d`
(stamp `fb5f142`). Adjudicated mechanically by
`bench/phase1/reduce_dequant_forward.py`; the verdicts below are that reducer's
output, not a reading of the tables.

## VERDICT: NOT CONFIRMED

Two runs on two devices. **Neither run had both devices instrument-clean at the
same time**, so the registered two-device conjunction never closed.

| | H100 80GB (sm_90) | RTX 4090 (sm_89) |
|---|---|---|
| **run 1** | 23 of 24 live — **S1 FAIL**, M1/F1/Q1/Q2/Q3 pass | **VOID** (9 of 24 outside band, 38%) |
| **run 2** (amended) | **VOID** (11 of 24 outside band, 46%) | 24 of 24 live — **S1 pass**, band MISS; device **CONFIRMED** |

Nothing was re-tuned, re-banded, or re-run for a better number. Both runs are
published with their receipts.

---

## What was measured, and against what

The training axis had only ever been measured against Unsloth's grouped GEMM
(1.11× median, 0.48–1.65, 9/16 cells, one device, exploratory). The baseline
practitioners with this problem actually run is different: **per routed expert,
call `bitsandbytes.functional.dequantize_4bit` on the packed weight inside the
forward, then `F.linear` on the result.** It is published at 743B by GenON —
[`genonai/nf4moe`](https://github.com/genonai/nf4moe), `QuantizedNaiveMoe`, and
the writeup at [`mncai/nf4moe`](https://huggingface.co/mncai/nf4moe) — and
re-derived by every hand-rolled fused-MoE QLoRA path. This leg measures it.

The arm was written from that source, not from a description of it: same
`quant_type="nf4"`, same blocksize 64, dequant per **hit** expert inside the
forward, weight materialized and dropped in the loop body. The module's routing
plumbing — one-hot hit mask, the single `.tolist()` host sync, the per-expert
`torch.where` gather, `index_add_` accumulation — is **not** charged to the
baseline, because every arm in this harness is measured at the KERNEL_CONTRACT
op boundary with rows pre-assembled, and charging one arm for assembly no other
arm pays is the asymmetry the head-to-head's amendment 2 existed to remove.
Hoisting it makes the baseline **faster** than what GenON published, which is
the conservative direction; its cost is measured separately as `D_routed` and
reported rather than assumed.

Both arms carry the identical LoRA (`lora_delta_grouped`, r=16, `lora_B`
zero-init, applied pre-activation). Neither uses gradient checkpointing — a cell
is one projection's forward and backward, so "configured identically" means off
in both. Neither applies routing weights. Weights are synthetic. The optional
Unsloth arm was deliberately not run: report-only under the prereg, and
`import unsloth` patches torch globally, which is a measurement risk to the two
arms that carry the verdict.

---

## The best-graded leg on each device

Reading rule: **`d/g` > 1 means the fused kernel is faster.** A cell whose
self-pair or drift left its band is **VOID** — the number is withheld, not
reported small.

| | H100, run 1 (23/24 live) | RTX 4090, run 2 (24/24 live) |
|---|---:|---:|
| `decode_m8` median | **1.070×** (5 of 7 at bar) | **1.112×** (7 of 8 at bar) |
| `tokbudget_2048` median | **1.709×** (7 of 8) | **1.827×** (8 of 8) |
| `tokbudget_11800` median | **1.468×** (6 of 8) | **2.441×** (8 of 8) |
| range at 11 800 tokens | 0.797 – 1.825 | 1.754 – 4.785 |
| transient memory D/G | 4.34 → 2.05 → 1.18 | 4.20 → 2.05 → 1.18 |
| J/step D/G | 1.14 → 1.72 → 1.48 | 1.21 → 1.82 → 2.45 |
| `b_rel` fused/baseline | 0.761 – 0.765 | 0.564 – 0.765 |
| shared LoRA floor (of G) | 0.54 → 0.29 → 0.22 | 0.62 → 0.33 → 0.32 |
| GenON plumbing on top | 1.56 → 1.53 → 1.19 | 1.70 → 1.25 → 1.07 |

The **large-budget numbers are the durable ones**: at `tokbudget_11800` every
cell was live in both runs on both devices, and the two runs agree to **1.005–
1.039 on the H100 and 0.977–1.023 on the 4090**. The `decode_m8` row is where
the instrument struggles and where the runs disagree by up to 33%.

---

## Findings

**1. The fused advantage is smallest at small batch — on both devices, in both
runs.** `decode_m8` medians: 1.070 / 1.132 (H100 runs 1, 2) and 1.112 (4090
run 2). The registered S1 band was **1.3–3.0×** and it **misses on every leg**.
S1's bar (≥1.0 on ≥6 of 8) fails on the H100 in run 1 (5 of 7 live) and passes
on the 4090 in run 2 (7 of 8).

Most of what is being measured there is not either kernel. **The shared LoRA
floor — the identical `lora_delta_grouped` call both arms make — is 54–66% of
the fused arm's time at `decode_m8`**, against cells that are themselves only
1.2–1.6 ms. Across all live cells it spans 0.18–0.68 of the fused arm, tracking
cell size. It is added to both arms and therefore compresses every ratio toward
1.0, and it is largest exactly where the ratios are smallest. It is measured per
cell and reported. It is **not subtracted from anything**: arithmetic on ratios
after the fact is how instruments get talked into saying what you wanted. A
floor-free comparison is a different experiment and needs its own registration.

**2. The M-axis answer is architecture-dependent, and my registered prediction
is wrong in both directions.** I registered monotone **decay** with token
budget, reasoning that GenON's dequant tax is fixed per step while the GEMM is
not. Measured:

- **H100 (HBM3):** 1.070 → 1.709 → 1.468. Rises, then falls.
- **RTX 4090 (GDDR6X):** 1.112 → 1.827 → 2.441. Rises monotonically.

Both are misses. The mechanism the prediction named is real and measurable, but
it is one of three effects and not the dominant one below 2048 tokens. Splitting
each arm's time into a T-independent part and a per-token part across the 2048
and 11 800 cells (**post-hoc and descriptive — not a registered criterion, and
it rescues nothing: S1 still fails and S2 is still a miss**), on the H100:

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
in every cell** — 33–54% of its total at 2048 tokens, under 10% at 11 800. That
is GenON's ~2.5 s/step dequant tax, in miniature, on a single projection. Being
the larger fixed cost, it amortizes faster, which is why the H100 curve falls
after 2048. On the 4090 it does not fall, because the extra bytes the dequant
round-trip moves cost more on GDDR6X than on HBM3 and that effect dominates.

**3. At GenON's own token budget, the sign is shape- and device-dependent.** At
11 800 tokens the fused arm wins **8 of 8 on the 4090** (1.754–4.785×) and
**6 of 8 on the H100** (0.797–1.825×). Both H100 losses are OLMoE — the
smallest-expert model in the census and this repo's long-standing known-loser
class — at 0.797 and 0.861. Published per-cell, because the split is the useful
statistic.

**4. The memory trade is real and it also decays with batch.** Transient bytes
for one forward+backward, D over G: **2.95–7.73× at `decode_m8`**, falling to
**1.07–1.74× at 11 800 tokens**, essentially identically on both devices
(medians 4.34/4.20 → 2.05/2.05 → 1.18/1.18). `F.linear` saves its weight for
backward, so the dequant-on-forward arm holds every hit expert's materialized
bf16 weight across the forward-to-backward window; gnf4's autograd Function
re-decodes one expert at a time and stores none. At large T the activations
dominate and the held weights stop mattering. M1's bar passes (8 of 8 every
leg); its predicted band (1.5–20×) **misses** at 1.18×.

**5. Fidelity: the fused kernel's error is architecture-invariant, and the
baseline's is not.** Across all 96 device-cells (24 × 4 legs), `b_rel` for the
fused arm is **1.6952e-3 – 1.7048e-3** — the same to four significant figures on
sm_90 and sm_89. The baseline sits at 2.2192e-3 – 2.2395e-3 in **82 of 96**, but
on the 4090 **seven large-M shapes jump to 2.5232e-3 – 3.0170e-3** — 14 of the
96 device-cells, because each recurs in both 4090 runs, and it does so
**bit-identically across two runs on two different 4090 instances** (e.g.
gemma-4 gate_up at 11 800 reads 3.0170e-3 in both). cuBLAS selects a different
`F.linear` kernel for those shapes on Ada — one that is both slower and less
accurate. F1's gate passes everywhere (fused never worse) and its predicted band
0.5–0.9 holds, but in those seven cells the two arms are not computing to the
same precision and the speed ratio there is partly a trade the baseline made,
not purely a kernel comparison.

**6. GenON's published routing plumbing costs 1.04–2.33× on top of the same
compute** across live cells (per-regime medians on the H100: 1.56 at
`decode_m8`, 1.19 at 11 800). That is the
one-hot mask, the `.tolist()` sync, the gather and the `index_add_` — the part
this leg deliberately did *not* charge to the baseline. The headline comparison
therefore runs against a baseline meaningfully faster than the published module.

**7. Energy tracks speed and adds no independent axis here.** J/step D over G
follows `d/g` closely in every regime on both devices (H100 1.14/1.72/1.48;
4090 1.21/1.82/2.45). Draw ranged 67–506 W across the four legs (the 4090's
smallest cells idle far lower than the H100's). There is now a training
energy number where there was none; it does not tell a different story from the
timing.

---

## The instrument, and what it cost

**The self-pair earned its place twice, and both times it was right.**

Run 1's consumer leg went VOID on 9 of 24 cells, 8 of them the whole
`decode_m8` row. The cause was diagnosable from the receipts rather than
guessed: inside every voided cell the four G timings step **down** across the
cell (gemma-4 down: 3.138, 3.135, 2.528, 1.579 ms) while every position-2 and
position-3 cell on the same box is flat to ~0.3% (gpt-oss gate_up at 11 800:
108.890, 108.783, 109.139, 108.855). The voided cells are exactly the ones that
run first after a per-spec `QuantStack` build — a long CPU-bound stretch with
the GPU idle. **They were measuring the card clocking back up.** `_timed`'s ten
warm-up iterations are ~14 ms on a 1.4 ms cell, nowhere near a boost. The
H100's locked datacenter clocks masked it in seven of its eight equivalent
cells; the consumer card did not.

Amendment 1 (stamped pre-data) added `_warm()`: 1.5 s of GPU-busy work before
the pilot and first timed block of every cell, wall-clock rather than
iteration-count, applied identically to every arm before any arm is timed. It
worked — the 4090 went from 9 void to **0 void**. And it showed what the gate
had saved: the cell the broken instrument read as **1.515** the repaired one
reads as **1.105**, a 27% error that would have been published.

Run 2's H100 then voided 11 of 24 on a *different* failure — drift **upward**
(1.05–1.19) across `decode_m8` and `tokbudget_2048` cells, i.e. the box slowing
across the cell rather than speeding up. That is instance noise on shared
rented hardware, not the repaired defect, and it is reported rather than
re-rolled. Its live cells agree with run 1's to 1.005–1.039 at 11 800 tokens.

---

## Per-cell

Every cell, including the losses and the voids: [`per-cell-run1.md`](per-cell-run1.md)
and [`per-cell-run2.md`](per-cell-run2.md). Reducer verdicts:
[`verdicts-run1.json`](verdicts-run1.json), [`verdicts-run2.json`](verdicts-run2.json).

Property suite **49/49** on every leg, plus this leg's 20 CPU tests. Wiring
smoke passed before any timing on every leg. Q2 (wiring, positive-controlled)
had **zero failures across all four legs**: the dequant counter saw one call per
hit expert per forward in every cell, the routed probe reproduced the sliced arm
on every row, and the `lora_A` positive control fired in all three arms.

---

## What can now be claimed that could not before

**Before:** the training axis had one comparator — Unsloth's grouped GEMM at
1.11× median, one device, exploratory, with per-cell self-pairs 0.618–1.088 that
voided the individual cells.

**Now:** against the dequant-on-forward pattern as published, with an adjacent
self-pair that is reported per cell and that voided what it should have, on two
devices:

- at **GenON's own token budget** the fused training path is **2.44× median on
  a consumer 4090** (8 of 8 cells, 1.75–4.79×) and **1.47× median on an H100**
  (6 of 8 cells, 0.80–1.83×, both losses OLMoE);
- at **batched decode** it is **1.07–1.11×**, and roughly half of what is being
  timed there is a host-side floor both arms share;
- **fidelity is 0.76×** the baseline's error, invariant across sm_86, sm_89 and
  sm_90 to four significant figures;
- **transient memory** is 1.07–7.73× lower depending on batch;
- and the pattern's **per-step fixed cost is 1.6–2.7× the fused path's**, which
  is the quantity that decides how the comparison moves with batch size.

That is a statement about the thing practitioners are running, not a translation
from a comparison they are not running.

**What did not survive:** the registered speed bar at small batch on the H100
(S1), all three predicted magnitude bands (S1, M1, and S2's direction), and the
two-device conjunction. The claim is narrowed accordingly.

---

## What this does and does not license

- **Licensed:** a comparison against the dequant-on-forward pattern as
  published, at the stated token budgets, on the stated two devices, on
  synthetic weights, at the projection boundary, from the legs that graded
  instrument-clean.
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
- **NOT licensed: quoting a VOID cell.** Nine cells in run 1 and eleven in run 2
  have no measurement.
- **Device count and reps are as stated:** two devices, **single-rep per
  device per run**, two runs. No cell is replicated within a run; the per-cell
  self-pair is the instrument check, and cross-run agreement is the replication
  that is claimed — and it is claimed only at the token budgets, where it holds
  to ≤4%.

## Receipts

```
run1/H100/  run1/ADA/  run2/H100/  run2/ADA/   per device-run:
  dequant_forward_<TAG>.json   24 cells: gates, fidelity, memory, energy, all timings
  dequant_forward_smoke.json   the pre-timing wiring smoke
  suite.txt                    property suite 49/49 + this leg's 20 CPU tests
  run.log                      full pod transcript
  DONE                         terminal state
verdicts-run1.json  verdicts-run2.json   the reducer's adjudication
per-cell-run1.md    per-cell-run2.md     the per-cell tables, as generated
SHA256SUMS
```

Four pods ran matrices; four more were created and destroyed during
provisioning (three RTX 4090 and one RTX 5090 came up `RUNNING` with no public
IP and were torn down by the wedge check or the provisional backstop). Every
teardown was verified by re-query, not trusted from the `DELETE`.

Close-out, measured rather than estimated: **zero live pods**,
`currentSpendPerHr` back at the **$0.005** idle-network-volume floor, and total
metered spend for the whole leg **$2.02**, read as an account-balance delta
(187.5650 → 185.5488) rather than summed from per-pod rates. Standing cap is
$35/job.
