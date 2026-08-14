# Phase 1 — baseline harness

Measures the baselines the fused kernel is registered against
(`gemm_predictions.json`): census shapes × regimes × backends, with CUDA-event
medians, J/token power receipts, and the per-cell fp64-reference fidelity that
`TOLERANCE_CONTRACT.md` makes the comparator (the dequant path's `b_rel` is the
2× bound's denominator).

```
python bench/phase1/harness.py --smoke              # tiny shapes, sanity (GPU)
python bench/phase1/harness.py --models OLMoE       # one model's cells
python bench/phase1/harness.py                      # full census sweep
```

Backends in v0: `dequant_grouped` (the e4b product path: per-active-expert
`dequantize_4bit` → bf16 mm), `gemv_4bit` (bnb's in-kernel-dequant reference,
bs1 only). `unsloth` and `marlin` are registered but import-guarded — they
record as `skipped` with the reason until their wiring lands (their absence is
visible in the receipts, not silent).

Receipts: `phase1_<gpu>.json` — one cell per (model, proj, regime, backend)
with `ms_median`, `tok_per_s`, `j_per_token` (+ sampling method/rate),
`b_rel_vs_fp64`, and the env pin (GPU, capability, driver, torch, bnb). The
Phase-2 kernel drops into the same registry so its cells land beside the
baselines it must beat: ≥1.3× tok/s at decode bs1 on sm_86, ≥1.0× train fwd,
J/token strictly below, fidelity-ordered per the contract.

Regime notes: `decode_bs1` = top_k experts × one token (the skinny extreme the
roofline says is memory-bound, ceiling ~8.1× vs the two-pass dequant path);
`prefill_s2048` = uniform-routing analytic M per expert (census note; replaced
by measured routing histograms later in Phase 1).

---

## Two standing defaults, added 2026-08-14

### 1 · Every leg reports its MEASUREMENT CLASS, beside the self-pair

`harness.gpu_busy_fraction()` measures summed CUDA kernel self-time per step
over wall per step, across steps run back to back with no syncs — i.e. how a
training loop actually runs. It now runs in **every** leg driver, not as an
after-the-fact probe, and lands `gpu_busy` per arm per cell in the receipt.

**Registered interpretation** (`kernel/prereg_gpu_busy_labelling.json`, fixed —
not chosen per cell): a cell where **either** arm is below **50%** GPU-busy is a
`step_ratio`, **not a kernel measurement**, and must be labelled that way in any
results table. It is still a real number — it is the wall-clock cost a training
loop pays — but it is not a claim about kernels. A cell run with `--no-busy` is
`unknown`, which is **not** `kernel` and must never be printed as one.

This exists because it was learned the expensive way: the probe was written
after legs 2 and 3 were graded and demoted their *primary* criterion from a
kernel result to a step ratio. The self-pair asks whether the box drifted; this
asks whether the quantity is about what it claims to be. Neither substitutes for
the other and both are cheap.

```
python bench/phase1/train_dequant_forward.py --busy-steps 50   # default
python bench/phase1/train_dequant_forward.py --no-busy         # opt out, labelled unknown
```

### 2 · Real prose is the fixture default; random ids are opt-in

`e2e_train_arms.py --data` now defaults to **`text`** (wikitext-2). Random token
ids **understate the fused advantage by 1.6–1.7× on every matched pair**,
because MoE routing is content-dependent: random ids hit *fewer* experts
(occupancy 0.984 → 0.875) and far more unevenly (cv 0.687 → 1.463), so the
baseline's per-expert Python loop runs fewer iterations — which is precisely the
cost the fused path removes. They remain available (`--data random`, or `both`
for the matched pair) for work that genuinely needs content-independence.

Every leg now emits `routing` — occupancy, cv, hit experts, and the **name of
the fixture they came from** — into every receipt, so a constructed fixture can
never be read as measured routing. Standing rail: **any results table citing a
random-id cell states the 1.6–1.7× understatement factor beside it.**
