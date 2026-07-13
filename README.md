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
- **Known losers:** `top_k=1` cells are **instance-unstable in both
  directions** (Scout `down` measured 0.47–1.12 across six contexts on
  identical code — split-K helps paired but can't stabilize the class), and
  **tiny shapes (≲5 M weight elements) lose outright** (0.24–0.35× speed,
  4–7× energy). v4 adds a dispatch floor that routes tiny cells back to the
  dequant path.
- **Prefill** (compute-bound M): after a group-size-keyed config pass,
  down-projections run **~1.9×** the dequant path and Qwen gate_up hits
  parity on the A5000; remaining gate_up cells sit at 0.4–0.8× pending a
  mainloop rewrite. Decode is still the primary product surface.

Three blind confirmatories have run; **none fully passed as registered**,
and each results doc says exactly what failed and why:
[v1](kernel/RESULTS-gate2-confirmatory.md) (caught the original per-shape
config table overfitting its census), [v2](kernel/RESULTS-v2-confirmatory.md)
(validated the replacement single-constant config on 64-SM parts and the
off-census `k≥6` wins), [v3](kernel/RESULTS-v3-confirmatory.md) (found the
v2-era SM-conditional premise was measurement noise, quantified the
`top_k=1` and tiny-shape loss classes, and established the methodology rule
that latency-bound cells only support paired claims). A fourth (v4: dispatch
floor + prefill config) is registered and running as of 2026-07-13; its
verdict will be committed either way. The preregs, amendments, evidence
JSONs, sweeps, and mechanical reducers are all committed; `.ots` files
anchor the protocols to Bitcoin before the data existed.

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
- `bench/phase2/` — decode config sweeps (both devices)
- `docs/KERNEL_CONTRACT.md`, `docs/TOLERANCE_CONTRACT.md` — op contract and
  fidelity spec; `census/`, `roofline/` — shape census + ceilings

Regenerate the machine-generated artifacts:

```
python3 census/make_census.py     # census/shape_census.json
python3 roofline/roofline.py      # roofline/ceilings.json
```

## Status / roadmap

Landed through v4: universal decode constant (the dense-sweep result),
split-K for starved grids (with a per-split work floor), a min-bytes
dispatch floor (tiny cells route to the dequant path via
`decode_dispatch()`), and the group-size-keyed prefill config. In flight:
the v4 blind confirmatory. Next: prefill mainloop rewrite for gate_up
parity; sm_120. Ecosystem landing is calendar-gated on the bitsandbytes
v0.50.0 release; see the coordination note on #1949.

## License & attribution

MIT ([LICENSE](LICENSE)). Portions developed with Claude Code as an AI
assistant under the author's direction and review — see
[ATTRIBUTION.md](ATTRIBUTION.md). All claims are the author's responsibility.
