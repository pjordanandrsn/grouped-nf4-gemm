# K26 — results: **DECODE**. The NF4 codebook lookup is about 80 % of K25's time, and an exact select-tree decode takes K25 to 0.38× at bit-identical outputs

Registration: `kernel/PREREG-k26-nf4-decode-ablation.md` (#430, `100b0a0`). Issue: experts4bit-qlora#564. Runner:
experts4bit-qlora `bench/k26/` (#830).

**Verdict by `k26_bench.verdict`: DECODE.** Its registered pointer: **K25 takes the select-tree decode.**

```
K26_VERDICT DECODE: affine/pair granite 0.223, olmoe 0.207: <= 0.6 in both -- the codebook decode is the bottleneck; next: K25 takes the select-tree decode (bit-equal; tree/pair 0.383 <= 0.8)
```

## The arms (ms per decode step, both GEMMs over every layer; median of 8 steps × 20 replays)

| arm | Granite (32 layers) | of floor | OLMoE (16 layers) | of floor |
|---|---:|---:|---:|---:|
| `pair`, K25 as merged | 5.869 | 18 % | 9.898 | 21 % |
| `copy`, the control | 5.877 | 18 % | 9.894 | 21 % |
| `affine` (`nibble − 8`, no lookup) | 1.310 | 83 % | 2.050 | 103 % |
| `noscale` | 1.169 | 93 % | 1.859 | 113 % |
| `bytes` | 1.131 | 96 % | 1.838 | 115 % |
| **`tree`** (exact select tree) | **2.189** | **49 %** | **3.794** | **56 %** |
| `served` (`_gemm_nf4_grouped`) | 5.650 | 19 % | 9.540 | 22 % |
| `scopy`, the served control | 5.588 | 19 % | 9.519 | 22 % |
| `stree` (served path + tree) | 6.063 | 18 % | 9.894 | 21 % |
| `load` (K25, KC 128), descriptive | 12.593 | 9 % | 22.556 | 9 % |
| the floor (active-expert bytes / copy bandwidth, 1525 GB/s) | 1.083 | | 2.109 | |

**Numerics, before timing.** These held on both families:
- K25 against the fp32 oracle (relative 0.0033 and 0.0035);
- `copy` bit-equal to K25;
- `tree` bit-equal to K25 on both projections;
- `scopy` bit-equal to the served kernel.

**`stree` is not bit-equal** to the served kernel, as on the A2000.

- **The lookup is the bottleneck.** Replacing it with `nibble − 8` (same bytes, same absmax) cuts K25's time by 78 %
  (Granite) and 79 % (OLMoE), and brings the kernel to 83–103 % of the copy floor. Over 100 % means the copy
  bandwidth is not a hard ceiling for these access patterns (L2 hits); the floor is an estimate.
- **The exact tree** gives the same fp32 weights and bit-identical outputs, at 0.373 (Granite) and 0.383 (OLMoE) of
  K25's time: 2.7× and 2.6× faster. That is also 0.39× and 0.40× the served NF4 GEMM.
- **The served kernel does not take the tree.** Its TF32 path with the tree is not bit-identical (the layout changes
  how the dot lowers, as `tl.gather` did in K25) and is not faster (1.073 / 1.037).

## Against the predictions

| prediction | outcome |
|---|---|
| DECODE, `affine / pair` 0.10–0.40 | **held**: 0.223 / 0.207 |
| `tree` bit-equal, `tree / pair` 0.40–0.75 | bit-equal held; **0.373 / 0.383, just below the band** |
| `stree` not bit-equal; the served kernel takes nothing | held |
| `served / pair` 0.95–1.10 | held: 0.963 / 0.964 |
| the copy controls hold | held: 1.001 / 1.000 and 0.989 / 0.998, all bit-equal |

## The run

- **`k26-5090-1`:** one RTX 5090 (driver 580.95.05, power.limit 600 W) on an AMD host. grouped-nf4-gemm at
  `100b0a0`, torch 2.8.0, triton 3.4.0.
- **Premise:** K25's contract compiled on the card (sm_120), 28/28.
- **Timeline (UTC):** destroyed 18:54:59, absent. **Cost:** $0.0322 of the $0.75 ceiling.

## What follows (the registered pointer)

- **K25 takes the select-tree decode.** It is bit-identical, so experts4bit-qlora lane P92's reading of K25 applies
  to it unchanged: Granite's K8 inside the gate, OLMoE's c4val1 K8 −0.107 ppl. The in-model speed is the next lane's
  to read.
- **OLMoE's quality** depends on K25's arithmetic (bf16 weights), which the tree does not change. The open lever
  there is a K25 that keeps the served kernel's weight precision (fp32 weights, TF32 MMA) with the tree decode.

## Receipts

[`receipts-k26/5090/`](receipts-k26/5090/) holds `k26.json`, summary, forensics, versions, the bench and contract logs,
the teardown proof and `SHA256SUMS`.

## Erratum (2026-10-05): two prediction bands were informed by A2000 timings

The PREREG's "What was seen before this page" records the bench's `--quick` correctness pass on the house A2000 and the
timings it showed (`affine / pair`, `tree / pair`, `stree / served`), and says the predictions are informed by that
pass. So the DECODE band (`affine / pair` 0.10–0.40) and the `tree / pair` band (0.40–0.75) were seeded by A2000
timings. Under the testbed policy (the A2000 is a correctness-only testbed; every timing, ratio or band basis comes from rented compute on the target card; grouped-nf4-gemm#475, `docs/audits/a2000-timing-2026-10-05.md` §2), an A2000 timing may not seed a prediction, whatever its label ("not a reading, but seen").

What it changes:
- **The registration stands as stamped.**
- **The verdict and the registered pointer do not rest on the A2000.** They come from the rule's thresholds
  (`affine / pair` ≤ 0.60 for DECODE, `tree / pair` ≤ 0.80 for the pointer), which the PREREG states in its rule
  section without reference to the A2000 pass, read on the 5090: 0.223 / 0.207 and 0.373 / 0.383.
- **The prediction table above reads two A2000-seeded bands.** DECODE held; `tree / pair` landed just below its band.
  That miss is the cost the policy names: a band seeded on sm_86 timings, read on sm_120.
- **What the A2000 pass legitimately established is numerics:** K25 against the oracle, the copy control and `tree`
  bit-equal, `stree` not bit-equal to the served kernel. The 5090 reproduced each.
- **`served / pair` 0.95–1.10** came from experts4bit-qlora lane P92's in-model census, not from the A2000 pass, and is
  unaffected.
- **For later lanes:** K27's registration repeated the pattern (its erratum is in
  `RESULTS-k27-nf4-tree-precision.md`). Bands come from a 5090 or H100 reading, or from a rented microbench of the
  component on the target card.
