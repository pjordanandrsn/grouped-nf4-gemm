# grouped-nf4-gemm — single-launch W4A16 GEMM over fused NF4 MoE expert stacks

A Triton kernel that runs the grouped expert GEMM **directly on NF4-packed
weights** — one launch for all active experts, LUT decode to fp32 in
registers, blockwise fp32 absmax, fp32 accumulation, bf16 epilogue. No
per-expert dequantize-then-`bmm` round trip, no bf16 weight materialization.
It consumes the canonical packed layout and conventions of the bitsandbytes
`gemm_4bit` family ([#1949](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1949)):
`[E, N, K/2]` uint8 + fp32 blockwise absmax, with `(sizes, expert_ids)`
supplied after the usual token→expert sort.

**Why:** for frozen 4-bit MoE experts, the standard path pays to decode the
weights into bf16 and then reads them again — at batch-1 decode that round
trip (plus ~3 kernel launches per active expert) dominates. Fusing the decode
into the GEMM deletes it. The measured side effect worth stating plainly:
**fp32 accumulation makes the fused path *more accurate* than the
materialize-to-bf16 baseline, in every cell ever measured here.**

## The claim (blind-confirmed, receipts in-repo)

Everything below is from **pre-registered, OpenTimestamps-stamped blind
confirmatory runs** (protocol + pass/fail criteria stamped before data; two
devices; n=3 fresh-process reps; worst/median-rep reduction; failures
reported at full volume). On sm_86 at batch-1 decode, versus the
dequantize-then-matmul baseline on the same stacks:

- **Fidelity:** property suite green on every device, every run (35 → 44
  tests as the kernel grew); fused output error **below the baseline's in
  every cell ever measured** (fp32 accumulate).
- **Energy:** fused J/token **below the baseline in 104 of 112
  confirmatory-grade cells across v1–v3**. Six of the eight misses are the
  `top_k=1`/tiny class (named below); the other two are parity-margin
  readings (1.005, 1.010) on a single instance. On bandwidth-bound cells the
  energy win has never failed to replicate.
- **Speed:** census MoE shapes (OLMoE, Qwen3-30B, Gemma-4, GPT-OSS-120B,
  gate_up + down) run **1.16–2.73× at median** (one census cell —
  gpt-oss `down`, 2880×2880 — is instance-sensitive: 0.7–2.0× across five
  instances). Fresh off-census shapes with `top_k ≥ 6` (DeepSeek-V3,
  granite-3.1, Qwen3-Next) run **1.0–1.8× at median**; `k=2`-large shapes
  (Grok-1, Mixtral-8x22B) 1.0–1.24×, never slower.
- **Versus the other execution classes** (same-run census on the v6
  kernel, [receipts](bench/phase1/results/comparators_v6/RESULTS-comparators-v6.md)):
  the grouped-bf16-GEMM class that unsloth's MoE backend rides
  (`grouped_gemm.ops.gmm`, dequant inside the timed path as 4-bit storage
  requires) loses to the fused kernel on **every census cell — decode
  median 4.67×, prefill median 3.02×**. Axolotl/PEFT stacks have no kernel
  of their own (their QLoRA forward is bitsandbytes `Linear4bit` — see the
  flagship bnb baseline). GPTQ-Marlin is fidelity-excellent but per-expert
  (launch-storm at MoE decode) and format-incompatible with NF4 checkpoints.
- **Known losers:** `top_k=1` cells are **instance-unstable in both
  directions** (Scout `down` measured 0.47–1.12 across six contexts on
  identical code — split-K helps paired but can't stabilize the class), and
  **tiny shapes (≲5 M weight elements) lose outright** (0.24–0.35× speed,
  4–7× energy). v4 adds a dispatch floor that routes tiny cells back to the
  dequant path.
- **Prefill** (compute-bound M): the v6 register-LUT mainloop rewrite
  (blind-CONFIRMED) runs **1.39–1.54× the prior mainloop on every census
  prefill cell**; against the dequant path the census reads 1.14–2.78× with
  all three large gate_ups above 1.15 — gate_up is no longer a loser class.
  One caveat carried at full volume: the dequant *baseline* itself swings
  ~25% between cloud instances (the fused kernel holds within 0.2 ms), and
  OLMoE gate_up (the smallest-expert shape) remains below parity at ~0.6×.

Six blind confirmatories have run; the first five **did not fully pass as
registered**, each results doc says exactly what failed and why, and the
sixth passed clean:
[v1](kernel/RESULTS-gate2-confirmatory.md) (caught the original per-shape
config table overfitting its census), [v2](kernel/RESULTS-v2-confirmatory.md)
(validated the replacement single-constant config on 64-SM parts and the
off-census `k≥6` wins), [v3](kernel/RESULTS-v3-confirmatory.md) (found the
v2-era SM-conditional premise was measurement noise, quantified the
`top_k=1` and tiny-shape loss classes, and established the methodology rule
that latency-bound cells only support paired claims),
[v4](kernel/RESULTS-v4-confirmatory.md) (dispatch floor + split-K work floor
+ prefill config; caught its own dispatch-point regression),
[v5](kernel/RESULTS-v5-confirmatory.md) (the load-time dispatch fix, clean on
the A5000 11/11 with energy 8/8 on both devices; one contended-A2000 noise
cell kept it from a full pass — the dispatch line is closed),
[v6](kernel/RESULTS-v6-confirmatory.md) (**CONFIRMED**, all five criteria:
the register-LUT M-tile mainloop, adjudicated on the instance-robust paired
rewrite ratio after the dress rehearsal exposed the dequant baseline's
host lottery). The preregs,
amendments, evidence JSONs, sweeps, and mechanical reducers are all
committed; `.ots` files anchor the protocols to Bitcoin before the data
existed.

## Flagship: a 235B MoE decoding at the PCIe physical limit on ≤16 GB of VRAM

`bench/phase3/` runs Qwen3-235B-A22B with **all expert weights NF4-packed in
host pinned RAM (~128 GB)** and streamed per-token over PCIe, with this
kernel as the sole MoE compute. Same discipline (prereg + OTS, receipts
in-repo):

- **[Phase A](bench/phase3/flagship/RESULTS-flagship-offload.md)** (synthetic
  weights, real GQA attention + router): **5.57 tok/s = 102–103% of the
  measured 44.3 GB/s link's waterfall ceiling** — the stream fully hides
  compute — on a **13.6 GB** working set. The dequantize-then-matmul path on
  the identical pipeline: 1.81 tok/s (34% of ceiling). ALL PASS.
- **[The gap is architectural](bench/phase3/flagship/RESULTS-flagship-bnb-baseline.md)** —
  we registered the prediction that bnb's own CUDA dequant kernel would
  also hide under the copy shadow (which would have narrowed our claim),
  and it was **refuted**: the standard path reaches **40% of waterfall**
  (per-expert dequant+GEMM compute outlasts the shadow), versus 93–94%
  fused on the same pod. Against the strongest standard comparator the
  fused path is **2.33× tokens/s and 2.21× J/token**.
- **[Phase B](bench/phase3/flagship/RESULTS-flagship-phaseB.md)** (the real
  438 GB checkpoint, stream-quantized to NF4 in place): **coherent greedy
  text at 4.3–4.4 tok/s on 15.2 GB VRAM**, replicated across five pods.
- **Expert prefetch is measured CLOSED, negative** — four registered arcs
  ([B2](bench/phase3/flagship/RESULTS-flagship-phaseB2.md) speculation:
  token-to-token expert stickiness is only 0.44;
  [B3](bench/phase3/flagship/RESULTS-flagship-phaseB3.md) early routing: the
  pre-attention router predicts the post-attention top-8 at **0.93** but the
  CPU sync tax eats the win;
  [B4](bench/phase3/flagship/RESULTS-flagship-phaseB4.md) threaded issuance:
  GIL tax, 0.57×;
  [B5](bench/phase3/flagship/RESULTS-flagship-phaseB5.md) GPU-driven
  zero-copy gather: hit rate H makes speculation move (2−H)× the bytes, and
  the observed loss matches that law to ~1% — break-even needs H ≳ 0.95,
  above this model's 0.93 predictor ceiling).
- **Recommended configuration: `--prefetch-mode gpu`** — expert ids stay
  GPU-resident and a triton kernel ([`kernel/host_gather.py`](kernel/host_gather.py))
  gathers expert rows straight from pinned host RAM over UVA (zero-copy),
  with no per-layer memcpy launches and no GPU→CPU syncs. It is the fastest
  measured arm (4.39–4.41 tok/s, +1.5% over serialized memcpy, byte-identical
  greedy output 6/6) and validates SM-issued UVA reads at ≥ copy-engine
  throughput at 7.98 GB/token.

## Reproduce

See [REPRO.md](REPRO.md) — suite, benchmark, and verdict reduction are each
one command from a frozen tree. Requires an sm_86 GPU, `torch ≥ 2.8`,
`bitsandbytes`, and a C compiler on PATH (triton builds launcher stubs at
runtime).

```
python -m pytest kernel/test_nf4_grouped.py -q        # 44 tests, ~2.5 min
python bench/phase1/harness.py --models OLMoE --regimes decode_bs1 \
    --backends dequant_grouped fused_nf4 --out receipts.json
```

## Layout

- `kernel/nf4_grouped.py` — the kernel (decode gemv path + M-tile path),
  packing helpers, torch reference decode
- `kernel/test_nf4_grouped.py` — property suite (bnb decode exactness at
  bf16 output precision, fidelity ordering, adversarial absmax, boundaries)
- `kernel/prereg_*.json` + `.ots` — pre-registered protocols, stamped
- `kernel/RESULTS-*.md` — results, including the failures
- `bench/phase1/` — backend-registry harness (dequant/gemv/grouped-mm/
  unsloth/marlin/fused), confirmatory evidence, reducers
- `bench/phase2/` — decode config sweeps (both devices); `arch/` —
  cross-architecture census (sm_86/89/90)
- `bench/phase3/` — the 235B offload flagship: `offload_decode_235b.py`
  (Phase A, synthetic), `offload_generate_235b.py` (real checkpoint,
  generation, prefetch arms), `flagship/` — results + receipts
- `kernel/host_gather.py` — GPU-driven zero-copy gather from pinned host
  memory (UVA), the recommended offload copy path
- `docs/KERNEL_CONTRACT.md`, `docs/TOLERANCE_CONTRACT.md` — op contract and
  fidelity spec; `census/`, `roofline/` — shape census + ceilings

Regenerate the machine-generated artifacts:

```
python3 census/make_census.py     # census/shape_census.json
python3 roofline/roofline.py      # roofline/ceilings.json
```

## Status / roadmap

Landed through v6: universal decode constant (the dense-sweep result),
split-K for starved grids (with a per-split work floor), a load-time
min-bytes dispatch floor (tiny cells route to the dequant path via
`decode_dispatch()`), the register-LUT prefill mainloop (v6, confirmed),
and the flagship offload pipeline (Phase A/B + the closed prefetch program
+ the UVA gather path). In flight: sm_120 census; the bnb-CUDA-dequant
pipeline baseline (registered prediction: the standard path also hides
under the PCIe shadow at decode, narrowing our speed claim to
energy/VRAM/simplicity — run blind either way). Ecosystem landing is
calendar-gated on the bitsandbytes v0.50.0 release; see the coordination
note on #1949.

## License & attribution

MIT ([LICENSE](LICENSE)). Portions developed with Claude Code as an AI
assistant under the author's direction and review — see
[ATTRIBUTION.md](ATTRIBUTION.md). All claims are the author's responsibility.
