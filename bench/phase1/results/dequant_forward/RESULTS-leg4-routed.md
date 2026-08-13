# RESULTS — leg 4: routing-faithful fixture, kernel and step ratios split

**Grades `kernel/prereg_dequant_forward_routed.json`** (OTS-stamped pre-data).
Descriptive by design: this leg sets **no pass/fail bar**, because legs 1–3 set
bars on a quantity now known to be mislabelled.

## The result

`d/g` > 1 means the fused path is faster. `wall` is what a training loop pays,
`gpu` is summed CUDA kernel self-time, `gap` = wall − gpu (launch gaps and host
stalls). Summed kernel time does **not** count gaps between launches, which is
exactly why the split is necessary.

| device | T | wall | **gpu** | gap | busy G / D |
|---|---|---:|---:|---:|---|
| H100 | 32 | 11.37 | **1.11** | 26.3 | 69% / **6%** |
| H100 | 2048 | 3.60 | **1.12** | 19.2 | 87% / 30% |
| RTX 4090 | 32 | 10.05 | **2.16** | 22.6 | 60% / 13% |
| **RTX 4090** | **2048** | **2.66** | **1.69** | 6.08 | **93% / 78%** |

**The advantage has two components, and only one of them is device-independent.**

1. **Eliminated launches.** `d/g_gap` exceeds `d/g_gpu` on 8/8 H100 cells and
   6/8 on the 4090. At training shape the baseline makes ~59 Python-mediated
   launches where the fused path makes one, and on the H100 it is **6%
   GPU-busy** while doing it.
2. **Reduced memory traffic — only where bandwidth binds.** `d/g_gpu` is
   **1.114** median on the H100 (0.815–1.522, i.e. parity) and **2.062** on the
   4090 (1.215–3.553). GDDR6X pays for the dequant round-trip's extra bytes;
   HBM3 largely absorbs them. This is the same saturation law leg 1 found on the
   M axis, arriving from a different direction.

**The cleanest single result in the program** is the 4090 at 2048 tokens: both
arms GPU-bound at 93% and 78%, faithful routing, on the consumer class this
kernel is built for — **1.69× in kernel time and 2.66× in wall time**, with no
host-overhead caveat attached.

## What the old fixture was costing

At T=32 the fiction (`top_k` groups) read **2.26–2.56** where faithful routing
reads **10.05–11.37**. Legs 1–3 were **understating** gnf4 by roughly 4–5× at
training shape, because they gave the baseline 8 dequant calls where real
routing gives it ~59.

The correction's direction is model-dependent, as the mechanism requires:

- **OLMoE at 2048**: 64 → 64 groups (occupancy already 1.000), so only skew
  changes and the ratio barely moves (2.33 → 2.08).
- **Qwen3-30B at 2048**: 128 → **99** groups. Qwen has ~27 experts that are
  essentially never routed to, so the old fixture was *over*-charging the
  baseline there and the faithful ratio comes **down** (4.33 → 3.44).

Registered predictions: **P1** (fixture moves wall ratio >10%) 7/8 and 6/8.
**P2** (kernel ratio shifts less than wall) 5/8 and 7/8 — barely met on the
H100. **P3** (advantage is gaps, not kernel work) 8/8 and 6/8.

## A claim this corrects — mine, made one device early

On seeing the H100 alone I stated the finding as "kernel time at parity, the
advantage is entirely launch gaps." **The 4090 refutes the general form**: its
`d/g_gpu` is 2.06, not 1.11. Parity is an HBM3 property, not a property of the
two kernels. This is the second time this session a one-device reading of mine
was corrected by the second device, and both times the two-card rule caught it.

## What this licenses

- **Licensed**: statements about OLMoE and Qwen3-30B under measured routing, at
  the stated token counts, on two devices, split into step / kernel / gap terms,
  on synthetic weights.
- **NOT licensed**: anything about Gemma-4 or GPT-OSS-120B under faithful
  routing — 8 cells are NOT-RUN because they have no measured routing and the
  fixture refuses to borrow another model's.
- **NOT licensed**: re-labelling legs 1–3's numbers with these ratios. Different
  fixtures; only the within-process matched pairs here support a comparison.
- **NOT licensed**: calling `d/g_gpu` "the kernel speedup" without also saying
  that summed kernel time excludes launch gaps — on the H100 that omission
  would turn a 11.4× step win into a claimed 1.1× and be just as wrong in the
  other direction.

---

# ⚠️ THE SPEED HEADLINE ABOVE DOES NOT SURVIVE AN ADVERSARIAL TEST

Registered pre-data in `kernel/prereg_dequant_forward_graphed.json`, whose stop
rule requires this be written here verbatim including if it removes the
headline. It does.

**The race actually available to a user.** The baseline CUDA-graphs cleanly
(4/4 attempts, both devices, process-isolated); the fused path fails capture
8/8. So a user choosing between these two can erase the baseline's launch
overhead today and cannot do the same for gnf4. Racing `D_base` graphed against
`G_base` as it ships:

| | H100 | RTX 4090 |
|---|---|---|
| T=32, ungraphed → **graphed** | 6.76 → **0.858** | 11.71 → **0.949** |
| baseline gained from graphing | **8.61×** | **12.49×** |
| T=2048, ungraphed → **graphed** | 1.63 → 1.059 | 2.94 → **1.489** |
| cells where the graphed baseline **beats** shipped fused | **5 of 8** | 2 of 8 |

**At training shape the fused path loses to a graphed baseline on both
devices.** The 10–11× in the table above was measuring launch overhead the
baseline is free to remove and gnf4 is not. Two handicaps were registered in
the baseline's favour and stand: a replayed graph reuses its input buffers
where a real loop would copy fresh activations in, and the fused arm got no
equivalent help because none exists for it today.

## What actually survives

1. **Token-budget scale on a bandwidth-limited card.** On the 4090 at 2048
   tokens the fused path beats even the graphed baseline at **1.489× median**
   (per-cell 1.09–3.15). This is leg 4's memory-traffic component, which
   graphing cannot touch. On the H100 the same cell reads 1.059 — parity —
   because HBM3 absorbs the extra traffic.
2. **Memory, untouched and large.** Transient memory D/G under faithful routing
   is **18.7–48.6× at training shape** (1.4–2.2× at 2048). A CUDA graph does
   not shrink the baseline's saved-weight footprint: `F.linear` still holds
   every hit expert's materialised bf16 weight across the fwd→bwd window, and
   real routing gives it ~59 of them. P3 confirmed, and the old fiction had
   understated this axis by **5.2–6.5×**.

## The claim this leaves

**Competitive at equal VRAM; wins when VRAM binds** — which is the position
this repo already held, now measured against the strongest comparator
available rather than the convenient one. Specifically:

- a small-batch *speed* claim against dequant-on-forward is **not supported**
  once the baseline is graphed, and no results doc here may make one;
- a *token-budget* speed claim on bandwidth-limited consumer hardware **is**
  supported at ~1.5×;
- the *memory* claim is supported at 19–49× and is the axis that decides
  whether a given model trains at all on a given card.

## What would change this

Making the fused path capturable — the two named hazards are the per-element
`int()` over `expert_ids` in `FusedGroupedNf4.forward` and the pageable
host-to-device copy `gemm_4bit_grouped` performs when handed a list. That is a
change to shipped code and a **separate experiment with its own registration**;
the stop rule here explicitly bars adding it after the fact to even this race.

## RESOLVED — the graphed race was an artifact of the static fixture

This replaces the caveat that stood here, per the stop rule in
`kernel/prereg_dequant_forward_dynamic.json` (OTS-stamped pre-data), which
required the measured answer be written in **either** direction. Both rented
devices, 8 gradeable cells each; the other 8 are `NOT-RUN` for want of measured
routing, as everywhere else in this program.

**CUDA graphs need static shapes and MoE routing does not provide them.** Once
the baseline is charged for obtaining them, the advantage it gained from
graphing is more than spent. `d/g` > 1 means the fused path is faster.

| cell | pad tax | H100 ungraphed | H100 graphed+padded | **H100 best** | 4090 ungraphed | 4090 graphed+padded | **4090 best** |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLMoE gate_up T=32 | 6.75× | 5.843 | 0.968 | **0.968** | 7.664 | 1.493 | **1.493** |
| OLMoE down T=32 | 6.75× | 8.275 | 1.128 | **1.128** | 7.344 | 1.617 | **1.617** |
| Qwen gate_up T=32 | 11.00× | 6.698 | 2.522 | **2.522** | 6.111 | 4.202 | **4.202** |
| Qwen down T=32 | 11.00× | 10.627 | 2.931 | **2.931** | 6.880 | 2.871 | **2.871** |
| OLMoE gate_up T=2048 | 4.95× | 1.177 | 3.379 | **1.177** | 1.966 | 9.551 | **1.966** |
| OLMoE down T=2048 | 4.95× | 1.993 | 3.425 | **1.993** | 2.064 | 8.471 | **2.064** |
| Qwen gate_up T=2048 | 8.12× | 2.188 | 11.864 | **2.188** | 3.370 | 33.652 | **3.370** |
| Qwen down T=2048 | 8.12× | 3.610 | 8.763 | **3.610** | 3.416 | 21.241 | **3.416** |

**Read the `best` columns, not the race.** The baseline is not obliged to pad —
it picks whichever configuration is faster for it, and at T=2048 that is plainly
*not* graphing. Quoting the graphed-and-padded race alone (median 3.16 on the
H100, 6.34 on the 4090; 6.09 and 15.40 if restricted to T=2048) would flatter
gnf4 by assuming an opponent's mistake.
Against the baseline's **best available like-for-like configuration** the margin
is **2.09× median on the H100 and 2.47× on the 4090** — smaller than the race,
larger than the ungraphed comparison, and the number that should be quoted.

**The cell that loses: H100 OLMoE `gate_up` T=32, at 0.968.** A graphed, padded
baseline beats the fused path there by 3%. It is the only cell of 16 below 1,
and it is on the device and regime where earlier legs already found the
comparison is ~90% host time. Its sibling `down` at the same shape reads 1.128.

Padding is not uniformly a penalty, which is worth stating because it argues
against us: on 2 of 16 cells `d_padded_plain / d_faithful` came in **below 1**
(H100 OLMoE gate_up T=32 at 0.73, down at 0.90) — 64 uniform groups can beat 58
ragged ones when the GPU is idle enough that the extra rows are free. The tax is
a row ratio, and it converts to time only when the device is saturated. That is
the whole mechanism: at T=32 padding is nearly free and graphing pays, at T=2048
padding is fully priced and graphing does not.

**Prereg grades.** P1 **confirmed** (tax > 3× at T=32: measured 6.75× and
11.00×). P2 **REFUTED** — predicted < 1.5× at T=2048, measured 4.95× and 8.12×.
The prediction reasoned from occupancy, which does saturate; the tax is set by
**skew**, which does not. Capacity must cover the *hottest* expert, and the
measured routing runs 31–795 against a uniform 256 (OLMoE cv 0.506; Qwen cv
1.607 with ~20% of experts empty). Skew persists at every token count, so the
tax never collapses. That refutation is worth more than P1 — it corrects the
quantity, not just the number. P3 registered **no direction** and reads above 1
in **15 of 16 cells**.

**Re-capturing per step is closed, not merely expensive.** Capture cost is
median 324 ms (H100) and 473 ms (4090), max 2583 ms, against a step of ~1–3 ms
— two to three orders of magnitude underwater. Bucketing is padding with extra
steps and inherits the tax above.

**What this does not license.** It restores the *comparison*, not a claim that
gnf4 is 2× faster in general: the T=32 cells remain host-dominated, the two
`NOT-RUN` models are still unmeasured, and routing fidelity is still one
2048-token capture per model. The Unsloth 1.11× comparison is untouched by any
of this.

### Amendment 1 — the capacity_factor arm (report-only, NOT a speed result)

The no-drop capacity above is the honest like-for-like bound, but it is not what
trainers run: real MoE training sets `capacity_factor` 1.0–2.0 and **discards**
the overflow. Leaving that unpriced would imply 5–11× is the baseline's only
option, which is false. At `cf` the row tax is ~`cf` by construction, so the
currency is **discarded routed work**, computed directly from the measured
per-layer histograms (median across all layers, sampling model removed):

| cf | row tax | OLMoE drops | Qwen drops |
|---:|---:|---:|---:|
| 1.00 | 1.00× | 22.6% | 54.5% |
| **1.25** | **1.25×** | **12.5%** (6.3–22.4) | **47.0%** (24.9–60.2) |
| 1.50 | 1.50× | 7.2% | 40.7% |
| 2.00 | 2.00× | 3.1% | 28.0% |
| no-drop | 4.95× / 8.12× | 0% | 0% |

A sampled cross-check agrees (14.1% / 45.9% at cf=1.25), so the skew is in the
measurement, not the simulation. **There is no good capacity setting for a
skewed router:** pad and pay 5–8× the rows, or clip and lose 12–47% of the
computation. Qwen is punished on both. This is MegaBlocks' dropless-MoE
motivation arrived at from the kernel side, and it is the sturdiest thing in
this section because it is a property of the routing, not of a device.

Two binding constraints, both registered in amendment 1 (which is **not blind**
— it was written after the no-drop partials, and the drop curve is a
deterministic function of histograms already in this repo, so it is not dressed
up as a prediction). **No timing from this arm may be quoted without its drop
rate in the same sentence**, and the arm may not be reported as a speed result:
the baseline is simultaneously given a graph the fused path cannot have *and*
excused work the fused path still performs in full. And the drop rates are a
**ceiling** — each histogram is one 2048-token inference capture, with no batch
mixing and no load-balancing auxiliary loss, both of which would flatten the
router. That bias makes the baseline look worse than it is.

The quality cost of discarding 12–47% of routed work is **not measured here and
cannot be** — that needs a training run and an eval, not a timer.

