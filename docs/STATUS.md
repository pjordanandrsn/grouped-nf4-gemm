# Status — what this kernel does, what changed, what is open

**As of 2026-09-03, version 0.29.0.** One page. The README argues; this
page states. Every line here has an entry in
[`docs/claims.json`](claims.json) with its evidence path, and nothing is
here that does not.

Evidence tiers, unchanged from the rest of the repo: **confirmed** =
pre-registered, stamped, blind confirmatory run; **measured** = a run
with a committed receipt; **projected** = arithmetic, not a run. One
addition, because it was being blurred: **measured-private** = the run
happened and the number is real, but the receipt lives in a private
audit tree, so *you cannot check it from this repository*. Those are
marked. Treat them as you would any unverifiable number.

---

## What it does today

**The kernel.** One Triton launch runs the grouped expert GEMM directly
on 4-bit-packed weights — NF4 on the bitsandbytes layout, and native
MXFP4 (OCP e2m1 + e8m0) on a checkpoint's exact released bytes. fp32
accumulation makes it *more accurate* than dequantise-to-bf16-then-GEMM,
in every confirmatory cell ever measured.

| | measured | tier |
|---|---|---|
| Decode, census MoE shapes vs the dequant path (sm_86) | 1.16–2.73× median | confirmed |
| Energy, J/token below baseline | 104 of 112 cells | confirmed |
| Real OLMoE QLoRA finetune, fused vs per-expert loop (prose) | 4.50× (4090), 4.75× (H100) | confirmed |
| vs Unsloth's own kernel, 4-bit-storage regime, decode | 1.70× (H100), 2.79× (4090) | confirmed |
| vs `torch._grouped_mm` on bf16, Qwen3-30B cell (RTX 5090) | 2.1–6.0×, on half the bytes | measured |
| Training backward, one launch, E=256 step | 403.7 → 26.5 ms | measured |

**Three things that limit those numbers, stated here rather than in a
footnote:**

1. **Against a CUDA-graphed baseline the fused path loses at decode**
   (0.949× on a 4090, 0.858× on an H100). What survives graphing is the
   memory-traffic win at training shape on bandwidth-limited cards
   (1.489× on the 4090; parity on the H100). A "fused wins at decode"
   reading that ignores graphing is wrong.
2. **Unsloth wins its own regime.** Against their bf16-resident kernel
   they run 2.6–5.3× faster at prefill on an H100. The advantage above
   is the 4-bit-storage regime specifically.
3. **Known losers:** `top_k=1` cells are instance-unstable in both
   directions; shapes under about 5 M weight elements lose outright
   (0.24–0.35× speed, 4–7× energy) and are routed back to the dequant
   path by a dispatch floor.

**Serving (sm_120).** Both decode knobs ship ON, capability-conditional:
an unset env takes fp8 where fp8 can run and the certified f32 path
otherwise, and an explicit request is never silently downgraded. The
certified single-stream anchor for Qwen3-30B-A3B on the RTX 5090 class
is **7.37 ms/step ±4.2% (≈130–142 tok/s)** — the class carries 8.5%
inter-box dispersion while each box repeats itself to 0.16%, so quote
the range, not a point.

`fp8_paged_decode_attention` takes sliding windows, attention sinks, a
custom attention scale and per-layer stride overrides (0.24.0), which is
what lets one engine serve Granite, Gemma-4 and gpt-oss geometries. 35
of 35 fp8-mode GPU tests pass on a 5090; `window=0, sinks=None` is
byte-for-byte the old path.

**Reaching past VRAM.** Qwen3-235B-A22B decodes at 4.3–4.4 tok/s on
15.2 GB of VRAM from a 438 GB checkpoint held in pinned host RAM,
replicated across five pods. The law is additive and per-box:
`t_token ≈ c_box + bytes/link`, with `c_box` measured at 53.5–114.0 ms
across seven hosts. A fixed fraction-of-waterfall is **not** the law and
was retired in July.

The NVMe tier is a **batch** tier: at `S ≈ 3.45 GB/s` a fully cold 235B
is ~2.3 s/token and a K3-class model ~7.5 s/token. It buys reachability
and provenance, not latency.

**Provenance.** gpt-oss-120b serves on its exact released MXFP4 bytes at
ppl 26.72 against the shipped-precision reference 26.75 — the +9.4%
NF4-requantisation tax is deleted — and QLoRA-trains at 9.82 GB peak
with 144/144 hashes identical before, during and after. The reference
MXFP4 decode reproduces Kimi K3's own declared reference exactly
(33,030,144 elements, max delta 0).

---

## What changed — retired, superseded, corrected

Kept here because a claim that quietly disappears is worse than one that
was wrong.

- **`sm_120` is no longer "parked".** The README's roadmap still says
  three cloud provisioning failures parked Blackwell work. That was true
  in July; since 0.15.0 the RTX 5090 has been the *primary* serving
  target — the M=1 config retune, the sm_120 census, the decode anchor,
  the M3 defaults, the int4 lanes and the paged attention were all
  measured on rented 5090s. **Retired as stale.**
- **The "4.67× vs the grouped-bf16 execution class" number is
  superseded.** That backend never executed Unsloth's own kernel, and
  the proxy it did run is 1.33× slower than the real thing — so the
  comparison was against a weaker opponent than the label implied. The
  head-to-head (1.70× / 2.79×) replaces it. The old number is kept in
  the README with that caveat attached, not rescaled.
- **Split-K on the decode GEMV is refuted** (flat at `gate_up`, ~14%
  worse at `down`). The kernel ships dormant *as the evidence*.
- **A fixed fraction-of-waterfall is retired as a law** (two 0.77
  readings were a two-host coincidence).
- **The cold-engine "free floor" premise is refuted** on its target box:
  no AVX-512 means bitsandbytes' CPU dequant runs at 0.041 GB/s against
  a ~12 GB/s ceiling.
- **Expert prefetch is closed, negative**, over four registered arcs.
  Speculation moves (2−H)× the bytes and break-even needs H ≳ 0.95,
  above this model's 0.93 predictor ceiling.

---

## What is open

- **#319 — the f32 paged-decode compute modes miss their reference** on
  torch 2.8.0+cu128 / triton 3.4.0 (10 of 35 tests, up to 0.074 against
  a 0.02 tolerance), on unmodified `main`. The fp8 modes — the sm_120
  serving default — pass, so serving is unaffected; the Ampere/Hopper
  compute path is not.
- **#87** — `gemm_4bit_grouped` int32 offset overflow at large
  `max(expert_ids)`, distinct from the 2 GiB stride fix in 0.13.2/0.14.0.
- **#73, #60, #58** — arena/NVMe efficiency: host copy is ~71% of a K3
  layer; staging blocks ~30% of a training step; 8 requests issued per
  layer where 2 would do.
- **#71** — `PINNED_ROW_FACTOR` is ~2× conservative on cgroup v1; v2
  needs a box the rented pods cannot give.
- **`docs/context-budgets.md` is rung-one only** (A2000-measured
  KB/token); full-depth real-weight confirmation is pending and the K3
  row is a declared gap. Its own text forbids promoting pending rows to
  the README — that still holds.
- **`docs/cold-engine/STAGE3-SYNTHESIS.md` carries one correction
  outstanding**: gate 1's published read counts are uncorrected, and no
  read count in that document should be quoted until it is re-run.
- **Every non-CUDA row is a `port target`.** ROCm/XPU numbers do not
  exist; `PROJECTIONS-multiarch.md` is arithmetic, stamped before the
  silicon, and explicitly invites refutation.

---

## Reading the numbers without getting them wrong

- **Quote the hardware.** Ratios move with the card: the Unsloth margin
  drops 40% when their TMA path is live; the fused/graphed split
  reverses between a 4090 and an H100 because the baseline's working set
  fits H100 cache.
- **Quote the baseline.** Almost every ratio here is against
  *this project's own* per-expert loop or the dequant path — not against
  a third party's implementation, except the one head-to-head that says
  so.
- **A benchmark on random token ids understates this kernel by ~1.6×**,
  because prose routes to 98.4% of experts and random ids to 87.5%.
  Benchmark on real text.
- **Peak VRAM does not improve.** The fused arms peak *higher*; only the
  transient held across forward-to-backward is smaller.
- **`measured-private` means you cannot check it.** Three entries in
  `claims.json` are in that state (the int4-b32 GEMV cells, the
  calibrated-pack quality numbers, the decode-glue composition). They are
  real runs with real receipts, in a tree this repository does not carry.
