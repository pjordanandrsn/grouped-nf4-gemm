# RESULTS — K15: Marlin beats our dequant path 1.6–2.0× at M=16, the advantage
# is the KERNEL not the format, and the prize is a third of what I estimated.

Measured 2026-09-11 under `PREREG-k15-marlin-comparator.md`. Receipts in
`receipts-k15/` (run `k15-marlin-4`). RTX 5090, 170 SMs, driver 595.84. Ours in
the image python (torch 2.8.0+cu128), Marlin in its own venv (vllm 0.28.0, torch
2.13.0+cu130). Launch floor **3.10 µs**, measured streaming ceiling
**1529 GB/s**.

```
K15 VERDICT: INCONCLUSIVE by the registered rule, and informative anyway
  saving 0.583-0.812 ms/step lands in the 0.5-1.0 band the prereg fixed as
  inconclusive; #564's 1.67 ms estimate is refuted as too optimistic
```

## The table, same box, M=16

| shape | ours `bf16` | Marlin g128 | Marlin g32 | our best int4 arm |
|---|---|---|---|---|
| `q_proj` | 10.3 µs | **6.37** (1.62×) | 8.18 | 12.5 |
| `o_proj` | 16.5 | **8.28** (1.99×) | 8.24 | 18.9 |
| `k_proj` | 4.5 | — | — | 10.4 |
| `v_proj` | 5.1 | — | — | 10.9 |

Relative error 0.0000–0.0005 against the dequantised reference Marlin's own
quantiser returns, so the calls are right.

`k_proj` and `v_proj` lost every cell to my workspace sizing: `MarlinWorkspace`
allocates `N//64 × 16` = 128 for N=512 and the kernel refuses below the SM count,
170. Fixed in #363 — **but the lane ran out of budget before a box could use the
fix** (see Cost). Those two shapes are unmeasured and are not interpolated here.

## The registered predictions

1. **Marlin g128 beats our bf16 path at M=16 on `q_proj`.** 6.37 against 10.35.
   **CONFIRMED**, and on `o_proj` too.
2. **Marlin lands within 2× of its own byte roofline.** `q_proj` 2.05×,
   `o_proj` 2.67× of `max(floor, bytes/BW)`. **REFUTED.** Marlin runs at
   **34–50 %** of the streaming ceiling — it is not memory-bound at these shapes
   either, and nobody is getting the 0.306 ms that #564 assumed.
3. **At matched bytes (g32), Marlin still beats our best int4 arm.** 8.18 against
   12.5 on `q_proj`; 8.24 against 18.9 on `o_proj`. **CONFIRMED.**

Prediction 3 was the one I cared about and it answers the engineering question:
on `o_proj` the group size makes no measurable difference at all (8.28 vs 8.24),
so **the advantage is the kernel, not the weight format.** We do not need to
repack to g128. We need a better GEMM.

## The prize, corrected

Per layer our four projections cost **36.4 µs**. Substituting Marlin where it was
measured:

| | µs/layer saved | ms/step |
|---|---|---|
| lower bound — `k_proj`/`v_proj` unchanged | 12.15 | **0.583** |
| upper bound — `k`/`v` scale like `o_proj` (1.99×) | 16.93 | **0.812** |

**experts4bit-qlora#564 estimated 1.67 ms and that is wrong.** It divided the
comparator's weight bytes by the streaming ceiling and assumed Marlin reaches it.
Marlin reaches a third to a half of it, so the real lever is **0.58–0.81 ms/step**
— still the largest single addressable item in the step, and roughly a third of
what I claimed. The decision rule fixed 0.5–1.0 ms as *inconclusive*, and that is
where this lands; the rule exists so the number is not rounded toward the answer
the lane went looking for.

## What follows, and what does not

- **A Marlin-class kernel on our own int4-b32 format is worth having**: g32
  Marlin is 1.53× our best int4 arm on `q_proj` and 2.29× on `o_proj`, and
  1.26–2.00× our bf16 dequant path. That is a kernel lane with a measured target,
  which is what this lane existed to produce.
- **Adoption is not established.** This compares engines as shipped across
  different torch and CUDA builds in different processes, as the prereg said it
  would. Using Marlin itself would mean a GPTQ repack, a dependency, and a
  quality gate — none of it touched here.
- **Nothing here is about the expert tier**, which is 52 % of the step and which
  #564 shows at 80–100 % of its memory roofline on an unmeasured distinct-expert
  count.

## Cost, and a ceiling breach

Six ledger rows, **$1.1253**, against a registered **$1.00** lane ceiling — over
by $0.13, and the lane is closed rather than extended.

| run | $ | |
|---|---|---|
| `k15-marlin` | 0.1088 | host never opened its ssh port |
| `k15-marlin-2` | 0.1646 | my `--system-site-packages=false`, not a flag |
| `k15-marlin-3` | 0.1584 | CUDA 12.9 box; vLLM 0.28.0 needs 13.0 |
| `k15-marlin-4` | 0.0838 | **the measurement** |
| `k15-marlin-5` | 0.1046 | host refused both account keys |
| `k15-marlin-6` | 0.5051 | host-limited: 38 min of pip for a 4 min install |

Four of the six were host draws that failed. Two were mine: a flag that does not
exist, and a driver requirement I had already seen the A2000 raise and did not
carry into the runner. The driver is now gated before any install, so that class
costs seconds.
