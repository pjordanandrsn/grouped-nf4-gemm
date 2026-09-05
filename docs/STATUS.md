# Status — what this kernel does, what changed, what is open

**As of 2026-09-05, `grouped-nf4-gemm` version 0.30.1.** One page. The README argues; this
page states. Every line here has an entry in
[`docs/claims.json`](claims.json) with its evidence path, and nothing is
here that does not.

Evidence tiers, unchanged from the rest of the repo: **confirmed** =
pre-registered, stamped, blind confirmatory run; **measured** = a run
with a committed receipt; **projected** = arithmetic, not a run. One
addition, because it was being blurred: **measured-private** = the run
happened and the number is real, but the receipt lives in a private
audit tree, so *you cannot check it from this repository*. Those are
marked. Treat them as you would any unverifiable number. The vocabulary is
`status_vocabulary` in [`claims.json`](claims.json) and `evidence_vocabulary`
in [`system-manifest.json`](system-manifest.json); every line below names
its claim ID.

---

## What it does today

**The kernel.** One Triton launch runs the grouped expert GEMM directly
on 4-bit-packed weights — NF4 on the bitsandbytes layout, and native
MXFP4 (OCP e2m1 + e8m0) on a checkpoint's exact released bytes. With fp32
accumulation it has never measured *less* accurate than the
dequantise-to-bf16-then-GEMM comparator in any registered confirmatory
cell (`gnf4.kernel.fused-more-accurate-than-dequant-bf16`).

| | measured | tier | claim ID |
|---|---|---|---|
| Decode, census MoE shapes vs the dequant path (sm_86) | 1.16–2.73× median | confirmed | `gnf4.kernel.decode-speed-census` |
| Energy, J/token below baseline | 104 of 112 cells | confirmed | `gnf4.kernel.energy-104-of-112` |
| Real OLMoE QLoRA finetune, fused vs per-expert loop (prose) | 4.50× (4090), 4.75× (H100) | confirmed | `gnf4.kernel.e2e-training-real-prose` |
| vs Unsloth's own kernel, 4-bit-storage regime, decode | 1.70× (H100), 2.79× (4090) | confirmed | `gnf4.kernel.h2h-unsloth` |
| vs `torch._grouped_mm` on bf16, Qwen3-30B cell (RTX 5090) | 2.1–6.0×, on half the bytes | measured | `gnf4.kernel.sm120-census-vs-grouped-mm` |
| Training backward, one launch, E=256 step | 403.7 → 26.5 ms | measured | `gnf4.kernel.dgrad` |

**Three things that limit those numbers, stated here rather than in a
footnote:**

1. **Against a CUDA-graphed baseline the fused path loses at decode**
   (0.949× on a 4090, 0.858× on an H100). What survives graphing is the
   memory-traffic win at training shape on bandwidth-limited cards
   (1.489× on the 4090; parity on the H100). A "fused wins at decode"
   reading that ignores graphing is wrong
   (`gnf4.kernel.graphed-baseline-decode-loses`).
2. **Unsloth wins its own regime.** Against their bf16-resident kernel
   they run 2.6–5.3× faster at prefill on an H100. The advantage above
   is the 4-bit-storage regime specifically (`gnf4.kernel.h2h-unsloth`).
   The model-level, training-axis end-to-end comparison is a separate
   claim in experts4bit-qlora's register
   (`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05`); it does not supersede
   this kernel-level one.
3. **Known losers:** `top_k=1` cells are instance-unstable in both
   directions; shapes under about 5 M weight elements lose outright
   (0.24–0.35× speed, 4–7× energy) and are routed back to the dequant
   path by a dispatch floor (`gnf4.kernel.decode-speed-census`).

**Serving (sm_120).** Both decode knobs ship ON, capability-conditional:
an unset env takes fp8 where fp8 can run and the f32 path otherwise, and
an explicit request is never silently downgraded. The paged attention is
therefore two support states, and `capabilities.json` carries it as two
entries: the **fp8 compute path** (`fp8-paged-attention-fp8-compute`;
sm_89+ precondition; measured on the RTX 5090 only;
`gnf4.serve.m3-defaults-on`) is supported; the **f32 compute path**
(`fp8-paged-attention-f32-compute`) — the sm_80–sm_88 default, the
fallback where an fp8 constraint fails, and every explicit f32 request on
any card — is open under #319 (`gnf4.open.f32-compute-modes-triton34`)
and carried as `unsupported` until it closes. The
certified single-stream anchor for Qwen3-30B-A3B on the RTX 5090 class
is **7.37 ms/step ±4.2% (≈130–142 tok/s)** — the class carries 8.5%
inter-box dispersion while each box repeats itself to 0.16%, so quote
the range, not a point (`gnf4.serve.decode-anchor-5090`).

`fp8_paged_decode_attention` takes sliding windows, attention sinks, a
custom attention scale and per-layer stride overrides (0.24.0), which is
what lets one engine serve Granite, Gemma-4 and gpt-oss geometries. 35
of 35 fp8-mode GPU tests pass on a 5090; `window=0, sinks=None` is
byte-for-byte the old path
(`gnf4.serve.fp8-paged-attn-windows-sinks-scale`).

**Reaching past VRAM.** Qwen3-235B-A22B decodes at 4.3–4.4 tok/s on
15.2 GB of VRAM from a 438 GB checkpoint held in pinned host RAM,
replicated across five pods. The law is additive and per-box:
`t_token ≈ c_box + bytes/link`, with `c_box` measured at 53.5–114.0 ms
across seven hosts. A fixed fraction-of-waterfall is **not** the law and
was retired in July (`gnf4.flagship.235b-phaseB`).

The NVMe tier is a **batch** tier: at `S ≈ 3.45 GB/s` a fully cold 235B
is ~2.3 s/token and a K3-class model ~7.5 s/token. It buys reachability
and provenance, not latency (`gnf4.nvme.tier-batch-only`).

**Provenance.** gpt-oss-120b serves on its exact released MXFP4 bytes at
ppl 26.72 against the shipped-precision reference 26.75 — the +9.4%
NF4-requantisation tax is deleted — and QLoRA-trains at 9.82 GB peak
with 144/144 hashes identical before, during and after
(`gnf4.mxfp4.serve-tax-deleted`, `gnf4.mxfp4.train-9.82gb`). The reference
MXFP4 decode reproduces Kimi K3's own declared reference exactly
(33,030,144 elements, max delta 0; `gnf4.k3.oracle-exact`).

---

## What changed — retired, superseded, corrected

Kept here because a claim that quietly disappears is worse than one that
was wrong.

- **`sm_120` is no longer "parked".** The README's roadmap still says
  three cloud provisioning failures parked Blackwell work. That was true
  in July; since 0.15.0 the RTX 5090 has been the *primary* serving
  target — the M=1 config retune, the sm_120 census, the decode anchor,
  the M3 defaults, the int4 lanes and the paged attention were all
  measured on rented 5090s. **Retired as stale** (`gnf4.retired.sm120-parked`).
- **The "4.67× vs the grouped-bf16 execution class" number is
  superseded.** That backend never executed Unsloth's own kernel, and
  the proxy it did run is 1.33× slower than the real thing — so the
  comparison was against a weaker opponent than the label implied. The
  head-to-head (1.70× / 2.79×) replaces it. The old number is kept in
  the README with that caveat attached, not rescaled
  (`gnf4.kernel.comparators-v6-execution-class`, superseded by
  `gnf4.kernel.h2h-unsloth`).
- **Split-K on the decode GEMV is refuted** (flat at `gate_up`, ~14%
  worse at `down`). The kernel ships dormant *as the evidence*
  (`gnf4.retired.splitk-gemv`).
- **A fixed fraction-of-waterfall is retired as a law** (two 0.77
  readings were a two-host coincidence).
- **The cold-engine "free floor" premise is refuted** on its target box:
  no AVX-512 means bitsandbytes' CPU dequant runs at 0.041 GB/s against
  a ~12 GB/s ceiling (`gnf4.cold-engine.phase0-premise-refuted`).
- **Expert prefetch is closed, negative**, over four registered arcs.
  Speculation moves (2−H)× the bytes and break-even needs H ≳ 0.95,
  above this model's 0.93 predictor ceiling
  (`gnf4.flagship.prefetch-closed-negative`).
- **#87 is closed by observation in every carrier** (PR #342; boundary
  test `kernel/test_expert_offset_boundary.py`; GPU run on an RTX 5090
  2026-09-05: 10 passed). The line this page carried until 0.30.1 —
  "`gemm_4bit_grouped` int32 offset overflow at large `max(expert_ids)`,
  distinct from the 2 GiB stride fix in 0.13.2/0.14.0" — was not
  supported by the issue text: the expert-id cast the issue asked for
  shipped in 0.13.2 (NF4) and 0.14.0 (MXFP4), and the carriers it flagged
  by inspection (split-K, dgrad) plus the ones it did not name (int4-b32,
  the fp8 paged readers) had never been exercised above the boundary.
  Now they are, each above-boundary case in its own process
  (`gnf4.kernel.expert-offset-boundary.5090.2026-09-05`, measured; the
  test file and the changelog entry are the public evidence, the GPU log
  is in the private receipt tree). #324's pre-launch shape refusal ships
  in the same release and moves no registered number.

---

## What is open

- **#319 — the f32 paged-decode compute modes miss their reference** on
  torch 2.8.0+cu128 / triton 3.4.0 (10 of 35 tests, up to 0.074 against
  a 0.02 tolerance), on unmodified `main`. The fp8 modes — the default
  on sm_89+ where the constraints pass, sm_120 serving included — pass,
  so that serving path is unaffected; the sm_80–sm_88 default path and
  every explicit f32 request (on any card, Hopper included) are not
  (`gnf4.open.f32-compute-modes-triton34`).
- **#73, #60, #58** — arena/NVMe efficiency: host copy is ~71% of a K3
  layer; staging blocks ~30% of a training step; 8 requests issued per
  layer where 2 would do.
- **#71** — `PINNED_ROW_FACTOR` is ~2× conservative on cgroup v1; v2
  needs a box the rented pods cannot give. (#73, #60, #58 and #71 are
  `gnf4.open.issues`.)
- **`docs/context-budgets.md` is rung-one only** (A2000-measured
  KB/token); full-depth real-weight confirmation is pending and the K3
  row is a declared gap. Its own text forbids promoting pending rows to
  the README — that still holds (`gnf4.open.context-budgets-rung-two`).
- **`docs/cold-engine/STAGE3-SYNTHESIS.md` carries one correction
  outstanding**: gate 1's published read counts are uncorrected, and no
  read count in that document should be quoted until it is re-run.
- **Every non-CUDA row is a `port target`.** ROCm/XPU numbers do not
  exist; `PROJECTIONS-multiarch.md` is arithmetic, stamped before the
  silicon, and explicitly invites refutation (`gnf4.projection.multiarch`,
  projected).

---

## Reading the numbers without getting them wrong

- **Quote the hardware.** Ratios move with the card: the Unsloth margin
  drops 40% when their TMA path is live; the fused/graphed split
  reverses between a 4090 and an H100 because the baseline's working set
  fits H100 cache.
- **Quote the baseline.** Almost every ratio here is against
  *this project's own* per-expert loop or the dequant path — not against
  a third party's implementation, except the one head-to-head that says
  so. The dequant path is bitsandbytes `dequantize_4bit` per active
  expert followed by a bf16 matmul, as each receipt ran it. Since
  bitsandbytes 0.50.0 (upstream #1949, merged 2026-05-21) its supported
  ordinary 2-D inference cells compute from the packed weights directly;
  no receipt here times that path, the grouped routed-MoE GEMM is a
  separate contract upstream does not have, and the conventional 4-bit
  backward still dequantises for dX. The ratios stay what their receipts
  measured (noted 2026-09-04).
- **A benchmark on random token ids understates this kernel by ~1.6×**,
  because prose routes to 98.4% of experts and random ids to 87.5%.
  Benchmark on real text.
- **Peak VRAM does not improve.** The fused arms peak *higher*; only the
  transient held across forward-to-backward is smaller.
- **Quote the register that owns the number.** A model-level figure — tok/s
  for a named model on a named card, a perplexity-gate verdict — is
  registered in [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora)'s
  `docs/claims.json`, not here; quote it
  with that register's claim ID and status, and a `measured-private` status
  there means not publicly reproducible, exactly as it does here. Kernel-level
  numbers are this register's.
- **`measured-private` means you cannot check it.** Three entries in
  `claims.json` are in that state: `gnf4.serve.int4-b32-gemv` (the GEMV
  cells), `gnf4.serve.gptq-pack-int4-b32` (the calibrated-pack quality
  numbers), `gnf4.serve.decode-glue-kernels` (the composition). They are
  real runs with real receipts, in a tree this repository does not carry.
