# Head-to-head vs Unsloth's own MoE kernel — one box, one process

Protocol: `kernel/prereg_unsloth_head_to_head.json` (+ amendments 1 and 2), each
OTS-stamped before the data it governs existed. Reducer:
`bench/phase1/reduce_unsloth_h2h.py`. Receipts:
`bench/phase1/results/unsloth_h2h/`.

> **Status: `measured`. The timestamps are not yet attested.** The `.ots` files
> are committed, but their Bitcoin attestations are still pending confirmation,
> so `ots verify` will not complete yet — re-run `ots upgrade` and it will. Until
> then the pre-data ordering of protocol against results rests on this repo's
> commit history, not on an external clock. The verdict below is
> `H2H_CONFIRMED` **against the registered protocol**, which is a different and
> weaker claim than "independently timestamped". They are not conflated here.

---

## Why this run exists

> "Nobody has run your kernel head-to-head against Unsloth's on the same box.
> Until someone does, 'competitive' is inference from two sets of self-reported
> numbers."

That was correct, and more precisely correct than it knew.

## The correction this repo owes

Every `unsloth` receipt this repo has ever banked ran
**`grouped_gemm.ops.gmm`** — tgale96's standalone package — and **never
Unsloth's own kernel**. The README named the impl it ran and hedged the claim to
the execution *class* ("the grouped-bf16-GEMM class that unsloth's MoE backend
rides"), so nothing published was false. But it is not a head-to-head, and
"competitive" therefore rested on two independently-produced sets of numbers.

Unsloth's actual MoE kernel is
`unsloth/kernels/moe/grouped_gemm/interface.py::grouped_gemm` — their own Triton
grouped GEMM with TMA support, an autotuner, and a full fwd/dX/dW autograd
Function.

**Two independent reasons the old arm could never have reached it**, both
verified on silicon by the G1 gate rather than argued from source:

1. `bk_unsloth` tries tgale96's `grouped_gemm.ops.gmm` **first and returns on
   success**. Wherever that package is installed — as it was in the environment
   that banked the v6 comparator receipts — the Unsloth probes are never
   reached.
2. Where they *are* reached, `getattr(unsloth.kernels.moe, "grouped_gemm")`
   yields the **submodule**, not a callable, and the call raises
   `TypeError: 'module' object is not callable`. Observed verbatim in a
   container without tgale96 installed.

> **A mechanism I got wrong, recorded because the repo's own discipline requires
> it.** The first draft of this finding blamed the empty
> `unsloth/kernels/moe/__init__.py` for making that `getattr` return `None`. The
> file genuinely is 0 bytes — but the inference was wrong: python binds a
> submodule onto its parent package on import regardless of `__init__` contents.
> The conclusion survived; the stated mechanism did not, and was corrected
> before the prereg was stamped. Same shape as #43/#45: the result was real, the
> causal story was not.

## What was compared

| arm | what it is |
|---|---|
| `fused_nf4` | gnf4's shipped kernel, **shipped default config**, no per-shape tuning |
| `unsloth_native` | Unsloth's kernel, **4-bit storage** — dequant inside the timed region, because their GEMM consumes bf16 and a 4-bit checkpoint must be materialized to feed it |
| `unsloth_native_bf16` | Unsloth's kernel, **bf16-resident** — nothing dequantized, stack cached. Their design point. gnf4 does not compete here |
| `dequant_grouped` | the e4b product path, retained as the anchor tying this run to prior result-sets |

### Handicaps registered in Unsloth's favour

- **`autotune=True`**: *their* autotuner picks *their* best config per shape,
  against gnf4's shipped default.
- **An H100 was rented specifically so their TMA path is live.** TMA gates on
  compute capability ≥ 9. Benchmarking only on Ampere/Ada would have measured
  their kernel with its fast path compiled out, and the result would have
  deserved dismissal.
- Dequant is charged to them **only** in the 4-bit cell, where it is genuinely
  unavoidable; never in the bf16-resident cell.

---

## Results

Two rented devices per run, n=3 fresh-process reps each, every arm timed against
a base re-timed immediately before it. **Four runs, 11 pods, every one torn down
and verified 404**, total spend **~$10.10** against the $35 standing cap. The
confirmed numbers come from run 2 (H100) and run 4 (RTX 4090); runs 1 and 3 are
superseded and their defects are recorded under Deviations.

Ratios are `unsloth_time / fused_time`: **>1 means the fused kernel is faster.**

### Headline — confirmed run (amendment 1, true pairing)

| device | TMA | decode median | prefill median | energy J/token |
|---|---|---:|---:|---:|
| **H100 80GB HBM3** (sm_90) | **live** | **1.70×** | 1.67× | 2.51× better (23/24) |
| **RTX 4090** (sm_89) | off | **2.79×** | 2.79× | **3.32× better (24/24)** |

**The H100 is the load-bearing device.** Once Unsloth's TMA path is live the
decode margin falls from 2.79× to 1.70× — 40% of the advantage. A comparison run
only on consumer silicon overstates the result by that much and deserves the
rebuttal it would get.

**The losing cells.** On the H100, 22 of 24 cells favour the fused kernel; the
worst reads **0.89×**. The reproducible loser is OLMoE `gate_up` at prefill —
the smallest-expert census shape and a known loser class in this repo's
published history, which reproduces here against the real comparator rather
than against a proxy. On the 4090 the same shape wins, so the loss is
SM-count-dependent, consistent with the compute-poverty progression already
published in the cross-arch sweep.

### P4 — how far the old comparator sat from the real thing

`grouped_gemm.ops.gmm` (tgale96, what every prior receipt actually ran) over
unsloth's real kernel, same run, same cells:

| device | median | range |
|---|---:|---|
| H100 | **1.33×** | 0.97 – 3.40 |
| RTX A5000 | **1.84×** | 1.03 – 2.66 |

The proxy is materially **slower** than the kernel it stood in for, so the
published 4.67× / 3.02× figures were measured against a weaker opponent than
"unsloth's MoE backend" implies. This is the quantitative form of the correction
at the top of this document, and it moves in the direction that is unflattering
to us.

Do **not** divide 4.67 by these numbers to "correct" the old figure: they are
within-run medians of different runs on different silicon, and this repo has
broken that rule once already (#22's 22.21 GB/s) and produced a faster-than-light
result. The old figure is superseded, not rescaled.

### Unsloth's own regime — their ceiling

`unsloth_native_bf16 / fused_nf4`, weights already bf16, nothing dequantized.
**<1 means Unsloth is faster.** gnf4 does not compete here and this table is why
the 4-bit numbers above must not be read as a general claim:

| regime | H100 | RTX 4090 |
|---|---|---|
| `prefill_s2048` | **0.19 – 0.39** | 0.34 – 0.85 |
| `decode_m8` | 0.43 – 0.86 | 0.73 – 1.04 |
| `decode_bs1` | 0.66 – 1.50 | 1.11 – 1.93 |

At prefill on the H100, Unsloth's kernel is **2.6–5.3× faster than the fused
kernel**. Their kernel is excellent at the job it was built for. The fused
kernel's advantage is specifically the **4-bit-storage** regime — where a bf16
GEMM must first materialize the weights — and it is not an advantage in
general.

Read down **this** table: against the bf16-resident arm the fused kernel's
position improves monotonically as M shrinks (0.19 → 0.43 → 0.66 on the H100),
and only reaches parity-or-better at `decode_bs1`. Against a bf16 kernel with
nothing to dequantize, the fused kernel is simply losing a compute race, and it
loses it less the more bandwidth-bound the cell gets.

**The 4-bit column does NOT follow that shape, and the appendix shows it.**
Median `u/f` by regime runs 2.32 → 1.48 → 1.67 (H100) and 3.68 → 2.22 → 2.79
(4090): the minimum is at `decode_m8`, not at prefill. So the advantage is not a
simple decay in M. It tracks how bandwidth-bound the cell is — `decode_m8` has
enough rows to start amortizing the weight read but not enough to reach the
prefill tile's efficiency, so it is where the dequant round trip the fused
kernel skips is worth least. Both devices agree on the shape, which is what
makes it worth stating at all.

### Registered verdict: **H2H_CONFIRMED on both devices**

Under amendment 1 (true per-comparator pairing), runs 2 and 4:

| criterion | H100 (TMA live) | RTX 4090 (TMA off) |
|---|---|---|
| **P1** decode ≥1.0 on ≥7/8 cells | **PASS** (median 1.704) | **PASS** (median 2.794) |
| P1 predicted band 2.0–6.0× | **MISS** — 1.70 | hit — 2.79 |
| **P2** prefill ≥1.0 on ≥5/8 cells | **PASS** (1.675, band hit) | **PASS** (2.794, band hit) |
| **Q1** fidelity | PASS | PASS |
| **Q2** adjacent self-pair ⊂ [0.97, 1.03] | **PASS** [0.984, 1.007] | **PASS** [0.983, 1.001] |
| **Q3** engagement positive control | **PASS** — TMA `True` | **PASS** — TMA `False` |
| **E1** energy (report only) | 2.51×, 23/24 cells | 3.32×, **24/24 cells** |
| **verdict** | **H2H_CONFIRMED** | **H2H_CONFIRMED** |

Two architecturally distinct cards, one with Unsloth's fast path live. **The
two-card rule is met.**

**The P1 band miss on the H100 stands and is not adjusted.** 1.70 against a
registered 2.0–6.0. Amendment 1 predicted, pre-data, that it would miss again —
and it did. The bar (≥1.0 on ≥7/8) passes; the prediction does not.

### What it took to get here, because the first attempt said NOT CONFIRMED

Run 1 failed Q2 at 1.039 / 1.032. The first diagnosis — that the self-pair was
merely *placed* wrong — was **incomplete**. The base protocol registered
*"fused_nf4 is re-timed immediately before each comparator and the ratio taken
per pair"*, but the harness timed each backend **once** per cell and divided
every comparator against that one shared timing. **That is not pairing at all**,
and the cell-spanning self-pair was correctly refusing to certify it.

Amendment 1 implemented true pairing (`--paired-base`), which moved the H100's
self-pair from 1.039 to [0.984, 1.007] and the 4090's to [0.983, 1.001]. The
band was never widened; the measurement was fixed.

The effect sizes barely moved (H100 decode 1.86 → 1.70), which is the useful
part: the instrument was broken in a way that was *not* manufacturing the
result.

---

## What the fidelity axis shows

Measured on the A2000 (a correctness-only testbed by operator instruction;
numeric agreement is not contention-sensitive, so these are citable while its
wall times are not). 8 cells, OLMoE + Qwen3-30B × {gate_up, down} ×
{decode_bs1, prefill_s2048}:

| arm | `b_rel` vs fp64 |
|---|---|
| `fused_nf4` | 0.00166 – 0.00170 |
| `unsloth_native` | 0.00220 – 0.00225 |

The fused kernel carries **~0.75× the error** of the Unsloth path on every cell,
consistent with this repo's standing fp32-accumulation claim. The two Unsloth
arms are **bit-identical to each other** (`max|Δ| = 0.0` on every cell), so any
timing gap between them is attributable to the dequant boundary and nothing
else.

Property suite: **49 passed, 0 failed**.

---

## Deviations from the registered protocol, stated plainly

1. **`suite_expect: "44/44"` was stale at stamp time.** I copied it from the
   v6-era prereg without re-counting; the suite has since grown to **49** tests.
   The criterion as literally written is unmeetable. The substance — zero
   failures — is met. Recorded as a deviation, not reinterpreted.

2. **The property suite did not run on the rented devices.** `pytest` is absent
   for the torch interpreter in `runpod/pytorch:1.0.7`. It was discovered after
   the H100 matrix had started, and installing it mid-run would have perturbed a
   paid in-flight measurement to patch a gate — the wrong trade. The suite was
   run free on the A2000 instead. The per-cell `b_rel_vs_fp64` half of Q1 *was*
   measured on both rented devices for every arm and cell, and is the stronger
   per-cell evidence. The runner now installs pytest before any timing.

3. **Run 1 was never paired, and I described the defect too charitably at
   first.** The base protocol registered per-comparator re-timing of the base;
   the harness timed each backend once per cell and divided everything against
   one shared timing. My first diagnosis blamed only the self-pair's *placement*.
   The placement was a symptom. Amendment 1 implemented true pairing and both
   devices then cleared Q2 on an unchanged band. **Run 1's ratios are superseded
   and should not be quoted.**

4. **Runs 2–4 are NOT BLIND.** Run 1's effect sizes were known when the
   amendments were written. Both amendments say so in their own text. This is
   reproducibility under a corrected instrument, not discovery, and it carries
   less evidentiary weight than the blind run 1 would have if run 1's instrument
   had been sound.

5. **The second device changed identity across runs.** Run 1 = RTX 4090
   (sm_89), run 2 = RTX A5000 (sm_86, both 4090 rungs unavailable), run 4 = RTX
   4090 (sm_89). The confirming pair is H100 + RTX 4090. The A5000 row is
   reported but is not the device the two-card rule is satisfied on.

---

## The training axis — EXPLORATORY, and it reversed

Registered in the base prereg as unable to change any verdict, and it does not.
The forward verdict above is a **forward** result.

Run 3 measured this and reported unsloth **3.3× faster**. That number was wrong
in two compounding ways, both corrected together in amendment 2 so neither could
be cherry-picked:

- gnf4 was timed at `dgrad_kernel=False`, the **default** — a deliberately EXACT
  reference path (per-expert loop through `dequant_ref`), not a performance
  path. That pitted unsloth's *tuned* backward against gnf4's *reference* one.
- gnf4's arm carried LoRA and unsloth's carried none.

Corrected, H100, 16/16 cells, LoRA on both arms:

| | median | range |
|---|---:|---|
| **T1** single-launch dgrad vs the reference default | **1.84×** | 1.12 – 5.41 |
| **T2** `u_4bit / g` — 4-bit training regime | **1.11×** | 0.48 – 1.65 (9/16 ≥ 1.0) |
| **T3** `u_bf16 / g` — their ceiling | 0.72× | expected loss |

All three land as registered. **`dgrad_4bit_grouped` is worth 1.84× and shipped
OFF BY DEFAULT** — the default alone flipped this comparison from a loss (0.30×)
to a win (1.11×). That default is now `True`.

**This leg is still exploratory and licenses no training claim.** Per-cell
self-pair ran 0.618–1.088, so individual cells are VOID under amendment 2's own
rule even though the median (0.996) is clean. A training claim needs its own
confirmatory protocol and its own blind run.

**Where not to fight:** T3. In training the frozen weights are consumed by both
the forward and the backward of one step, so a trainer with spare VRAM
dequantizes once and amortizes across both. That trade favours materializing
whenever VRAM allows. gnf4's constituency is where it does not — the honest
claim is *competitive at equal VRAM, wins when VRAM binds*, plus energy.

---

## Limits

- **The confirmed verdict is forward-pass only.** Unsloth's kernel is a
  *training* kernel with dX/dW; under a frozen 4-bit base both arms compute dX
  and LoRA grads, not dW, so this exercises less of their kernel than it is
  built for. The training leg above is exploratory and confirms nothing.
- Synthetic weights and uniform/census routing, as in every prior phase-1
  result-set.
- Two devices. Nothing here licenses a claim about a third architecture.

---

## Appendix — full per-cell matrix

Generated by `bench/phase1/render_h2h_matrix.py` from the committed receipts. Do not hand-edit; re-run it.

- **H100** — NVIDIA H100 80GB HBM3 (cc 9.0), TMA `True`, n=3 reps, verdict **H2H_CONFIRMED**
- **4090** — NVIDIA GeForce RTX 4090 (cc 8.9), TMA `False`, n=3 reps, verdict **H2H_CONFIRMED**

`u/f` and `J/tok`: **>1 = gnf4 faster / more efficient**. `bf16/f`: **<1 = Unsloth faster** (their bf16-resident design point, where gnf4 does not compete). `proxy`: tgale96's package over Unsloth's real kernel.


### `decode_bs1`

| shape | proj | H100 u/f | H100 bf16/f | H100 J/tok | 4090 u/f | 4090 bf16/f | 4090 J/tok | proxy (H100) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-30B-A3B` | down | 2.847 | 1.207 | 3.71 | 3.861 | 1.672 | 6.56 | 0.986 |
| `Qwen3-30B-A3B` | gate_up | 2.346 | 0.950 | 5.48 | 3.507 | 1.320 | 8.02 | 1.013 |
| `OLMoE-1B-7B-0924` | down | 3.531 | 1.502 | 6.92 | 4.447 | 1.929 | 11.50 | 0.986 |
| `OLMoE-1B-7B-0924` | gate_up | 2.964 | 1.148 | 7.00 | 4.694 | 1.659 | 17.05 | 1.154 |
| `gemma-4-26B-A4B` | down | 2.298 | 0.978 | 2.94 | 3.347 | 1.465 | 5.28 | 0.975 |
| `gemma-4-26B-A4B` | gate_up | 1.706 | 0.681 | 3.83 | 3.075 | 1.112 | 7.05 | 1.219 |
| `gpt-oss-120b` | down | 1.395 | 0.667 | 2.80 | 2.894 | 1.135 | 5.66 | 1.429 |
| `gpt-oss-120b` | gate_up | 1.626 | 0.661 | 3.95 | 4.566 | 1.597 | 9.36 | 2.067 |

### `decode_m8`

| shape | proj | H100 u/f | H100 bf16/f | H100 J/tok | 4090 u/f | 4090 bf16/f | 4090 J/tok | proxy (H100) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-30B-A3B` | down | 1.701 | 0.736 | 2.12 | 2.303 | 1.005 | 2.64 | 1.006 |
| `Qwen3-30B-A3B` | gate_up | 1.478 | 0.608 | 2.48 | 2.137 | 0.841 | 3.11 | 1.028 |
| `OLMoE-1B-7B-0924` | down | 2.026 | 0.859 | 2.82 | 2.416 | 1.043 | 2.94 | 0.998 |
| `OLMoE-1B-7B-0924` | gate_up | 1.854 | 0.723 | 2.72 | 2.694 | 0.944 | 3.89 | 1.202 |
| `gemma-4-26B-A4B` | down | 1.482 | 0.637 | 1.84 | 2.119 | 0.934 | 2.53 | 0.989 |
| `gemma-4-26B-A4B` | gate_up | 1.188 | 0.464 | 2.19 | 2.028 | 0.733 | 3.47 | 1.222 |
| `gpt-oss-120b` | down | 0.997 | 0.466 | 1.83 | 1.844 | 0.747 | 3.11 | 1.429 |
| `gpt-oss-120b` | gate_up | 1.043 | 0.430 | 2.46 | 2.583 | 0.928 | 4.75 | 2.068 |

### `prefill_s2048`

| shape | proj | H100 u/f | H100 bf16/f | H100 J/tok | 4090 u/f | 4090 bf16/f | 4090 J/tok | proxy (H100) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-30B-A3B` | down | 2.768 | 0.363 | 2.54 | 3.437 | 0.527 | 3.05 | 1.681 |
| `Qwen3-30B-A3B` | gate_up | 1.896 | 0.282 | 1.92 | 2.246 | 0.442 | 2.08 | 1.870 |
| `OLMoE-1B-7B-0924` | down | 1.335 | 0.242 | 1.24 | 1.635 | 0.382 | 1.56 | 1.524 |
| `OLMoE-1B-7B-0924` | gate_up | 0.890 | 0.191 | 0.97 | 1.198 | 0.335 | 1.15 | 2.007 |
| `gemma-4-26B-A4B` | down | 3.067 | 0.392 | 2.64 | 3.397 | 0.582 | 3.23 | 1.634 |
| `gemma-4-26B-A4B` | gate_up | 1.573 | 0.257 | 1.75 | 2.068 | 0.450 | 1.97 | 2.146 |
| `gpt-oss-120b` | down | 1.731 | 0.356 | 2.21 | 3.342 | 0.832 | 3.30 | 3.061 |
| `gpt-oss-120b` | gate_up | 1.619 | 0.356 | 2.21 | 3.484 | 0.846 | 3.34 | 3.403 |

### Median `u/f` by regime

| regime | H100 | 4090 |
|---|---:|---:|
| `decode_bs1` | 2.32 | 3.68 |
| `decode_m8` | 1.48 | 2.22 |
| `prefill_s2048` | 1.67 | 2.79 |
