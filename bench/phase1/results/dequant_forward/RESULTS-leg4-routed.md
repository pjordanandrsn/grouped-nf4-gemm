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
