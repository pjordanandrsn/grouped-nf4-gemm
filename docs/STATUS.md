# Status — what this kernel does, what changed, what is open

**As of 2026-10-08, `grouped-nf4-gemm` version 0.43.0.** This page states the current position in each area, with
the claim id behind each number; [`docs/claims.json`](claims.json) holds every claim's full text and evidence. The
dated narrative behind these positions, with the readings they replaced and why, is
[`STATUS-RECORD.md`](STATUS-RECORD.md).

Evidence tiers: **confirmed** = pre-registered, stamped, blind confirmatory run; **measured** = a run with a committed
receipt; **measured-private** = a real run whose receipt is in a private audit tree, so you cannot check it from here;
**projected** = arithmetic, not a run. A **retired** claim was published and is now known wrong; a **superseded** one
is true as measured, but a later row is the one to quote. To check whether a benchmark claim has been superseded or
retired, read its `status` in [`claims.json`](claims.json).

---

## What it does today

**The kernel.** One Triton launch runs the grouped expert GEMM directly on 4-bit-packed weights: NF4 on the
bitsandbytes layout, and native MXFP4 (OCP e2m1 + e8m0) on a checkpoint's exact released bytes. With fp32 accumulation
it has never measured less accurate than dequantise-to-bf16-then-GEMM in any registered confirmatory cell
(`gnf4.kernel.fused-more-accurate-than-dequant-bf16`; [confirmatories](../kernel/RESULTS-v6-confirmatory.md)).

| | measured | tier | claim ID |
|---|---|---|---|
| Decode, census MoE shapes vs the dequant path (sm_86) | 1.16–2.73× median | confirmed | `gnf4.kernel.decode-speed-census` |
| Energy, J/token below baseline | 104 of 112 cells | confirmed | `gnf4.kernel.energy-104-of-112` |
| Real OLMoE QLoRA finetune, fused vs per-expert loop (prose) | 4.50× (4090), 4.75× (H100) | confirmed | `gnf4.kernel.e2e-training-real-prose` |
| vs Unsloth's own kernel, 4-bit-storage regime, decode | 1.70× (H100), 2.79× (4090) | confirmed | `gnf4.kernel.h2h-unsloth` |
| vs `torch._grouped_mm` on bf16, Qwen3-30B cell (RTX 5090) | 2.1–6.0×, on half the bytes | measured | `gnf4.kernel.sm120-census-vs-grouped-mm` |
| Training backward, one launch: gradient vs the exact per-expert loop (A2000, a correctness read) | ~0.0029 relative, inside the bf16 budget | measured | `gnf4.kernel.dgrad` |
| Fused training step with it vs the per-expert loop, Qwen3-30B-A3B, 48 layers (rented RTX A6000; experts4bit-qlora's receipt) | 2.52× (1.72× without it) | measured | `gnf4.kernel.dgrad-step.a6000.2026-08-06` |

**Where it loses.** Three limits on every number above:
1. **Against a CUDA-graphed baseline the fused path loses at decode** (0.949× on a 4090, 0.858× on an H100); what
   survives is the memory-traffic win at training shape on bandwidth-limited cards
   (`gnf4.kernel.graphed-baseline-decode-loses`).
2. **Unsloth wins its own regime:** 2.6–5.3× faster at prefill on an H100 with their bf16-resident kernel. The margin
   above is the 4-bit-storage regime (`gnf4.kernel.h2h-unsloth`).
3. **Known losers:** `top_k=1` cells are instance-unstable, and shapes under about 5 M weight elements lose outright; a
   dispatch floor routes them back to the dequant path (`gnf4.kernel.decode-speed-census`).

### Decode and serving (the RTX 5090 is the primary target)

- **Single-row NF4 decode.** Since 0.43.0 the bandwidth-targeted GEMV (`GNF4_GEMV_BW=auto`) serves Qwen3-30B-A3B's two
  expert shapes on GPUs with at least 160 SMs: K33 read its kernel time at 0.341× the dot-pad route's
  (`gnf4.kernel.k33-nf4-decode-gemv-bw.5090.2026-10-07`; [results](../kernel/RESULTS-k33-nf4-decode-gemv-bw.md)), and
  served it decodes one request 1.2417× as fast, and 16 requests unchanged (`e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`). Other
  shapes keep dot-pad or the scalar GEMV.
- **The decode anchor** for Qwen3-30B-A3B on the RTX 5090 class is 7.37 ms/step ±4.2% (≈130–142 tok/s); quote the
  range, not a point (`gnf4.serve.decode-anchor-5090`).
- **Paged attention** takes fp8 compute where fp8 can run (sm_89+) and f32 otherwise, and never downgrades an explicit
  request; both paths are supported (`gnf4.serve.m3-defaults-on`, `gnf4.serve.f32-arms-ran-fp8`). Windows, sinks, a
  custom scale and stride overrides serve Granite, Gemma-4 and gpt-oss (`gnf4.serve.fp8-paged-attn-windows-sinks-scale`).
- **Programmatic dependent launch** is on for launches of at most 8 rows (`gnf4.kernel.k28-pdl-decode-chain.5090.2026-10-04`;
  served 1.0404× at one request, `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04`).
- **int4-b32:** the decode GEMV (`gnf4.serve.int4-b32-gemv`, measured-private); K16's small-M GEMM, 2.91× the bf16
  dequant path on o_proj at M=16 and experts4bit-qlora's default for the attention projections
  (`gnf4.kernel.k16-smallm-int4-gemm.5090.2026-09-19`); K19's grouped small-M GEMM at 0.736× the served GEMV route at
  B=16 (`gnf4.kernel.k20-k19-plan-sweep.5090.2026-10-01`). NF4's K25 decodes with an exact select tree
  (`gnf4.kernel.k26-nf4-decode-ablation.5090.2026-10-01`).
- **Row-count invariance.** A token decoded alone and inside a 16-, 17- or 160-token call gets the same bits from
  `gemv_int4_b32`, the NF4 dot-pad GEMV and `combine_rows` (`gnf4.kernel.int4-gemv-row-invariant.5090.2026-09-24`,
  `gnf4.kernel.nf4-dotpad-gemv-row-invariant.5090.2026-09-24`, `gnf4.kernel.combine-rows-row-invariant.5090.2026-09-24`);
  the grouped int4 GEMM and the scalar NF4 GEMV differ only in summation order
  (`gnf4.kernel.int4-grouped-gemm-reorder.5090.2026-09-24`, `gnf4.kernel.nf4-scalar-gemv-splitk-reorder.5090.2026-09-24`).

### Training

- **The backward is one launch** over the packed bytes, on by default (`gnf4.kernel.dgrad`; the step is in the table).
- **The grouped-LoRA delta** pads by default. Calls of at least 16,384 routed rows are bucketed (0.42.0) and run as one
  compact autograd node (0.43.0); shorter calls, TC1's field recipe among them, keep the single padded block. Read on
  Qwen3-30B-A3B and one RTX 5090 under torch 2.12 and 2.8 (`e4b.train.pad-buckets.torch28.qwen3.5090.2026-10-06`).
- **The single-block ladder is opt-in.** `NF4_QLORA_SINGLE_LADDER=1` (#513) puts the single block on ladder rungs so
  cuBLAS stops paying a new-shape host cost on nearly every call: 0.797 of the step with fp32 adapters on a host-bound
  box, 1.015 with bf16 adapters (`e4b.train.single-ladder.field.5090.2026-10-08`). The value `auto` (#514, unreleased)
  takes the ladder exactly when the adapters are fp32.
- **The training GEMM route** (`GNF4_TRAIN_GEMM=auto`) is grouped_mm on sm_90, dense off sm_90 for calls with at most
  16 present experts, and the fused kernels everywhere else.

### Correctness contracts

- **int64 offsets:** every carrier of the expert-base promotion is observed past its own 2^31 boundary (#87, #374, #386;
  `gnf4.kernel.expert-offset-boundary.5090.2026-09-05`, `gnf4.kernel.word-boundary-wide-dotpad.5090.2026-09-23`,
  `gnf4.kernel.boundary-gathers-appenders.interp.2026-09-23`).
- **`combine_rows` and `reduce_partials`** carry an accuracy contract, not a bitwise one; `reduce_partials` is bitwise
  the slot-order sum (#393; `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23`,
  `gnf4.kernel.reduce-partials-slot-order.5090.2026-09-23`; [results](../kernel/RESULTS-b393-combine-reduce-bitwise.md)).
- **The MXFP4 prefill combine** returns the same bits on every call: 1 distinct output in 50 identical calls (#408;
  `gnf4.kernel.mxfp4-prefill-combine-ordered.a2000.2026-09-28`); the QLoRA fused combine was fixed the same way (#409).

### Past VRAM, and provenance

- **Qwen3-235B-A22B** decodes at 4.3–4.4 tok/s on 15.2 GB of VRAM from a 438 GB checkpoint. The law is per box,
  `t_token ≈ c_box + bytes/link`, with `c_box` 53.5–114.0 ms across seven hosts (`gnf4.flagship.235b-phaseB`;
  [results](../bench/phase3/flagship/RESULTS-flagship-phaseB.md)).
- **The NVMe tier is a batch tier:** at `S ≈ 3.45 GB/s` a fully cold 235B is ~2.3 s/token. It buys reachability and
  provenance, not latency (`gnf4.nvme.tier-batch-only`).
- **Pinned-tier sizing** models PyTorch's power-of-two pinned allocator (#71 closed;
  `gnf4.kernel.k29-pinned-charge-cgroup-v2.5090.2026-10-04`), and the cold cost model's `link_eff` is measured per host
  (`gnf4.calib.link-efficiency.5090.2026-09-24`).
- **Provenance.** gpt-oss-120b serves on its exact released MXFP4 bytes at ppl 26.72 against the shipped-precision
  reference 26.75, and QLoRA-trains at 9.82 GB peak with its frozen expert bytes hash-identical after training
  (`gnf4.mxfp4.serve-tax-deleted`, `gnf4.mxfp4.train-9.82gb`). The reference MXFP4 decode reproduces Kimi K3's declared
  reference exactly (`gnf4.k3.oracle-exact`).

---

## Defaults and the reads behind them

Each default moved after a registered read; most reads are experts4bit-qlora's, on Qwen3-30B-A3B and one RTX 5090.

| default | where | way back | read |
|---|---|---|---|
| fp8 compute where fp8 can run (sm_89+), f32 otherwise | paged decode attention | `GNF4_ATTN_COMPUTE=f32` | `gnf4.serve.m3-defaults-on` |
| dot-pad NF4 GEMV at its census shapes, ≥ 160 SMs | NF4 decode | `GNF4_GEMV_DOTPAD=0` | `gnf4.serve.m3-defaults-on` |
| bandwidth-targeted NF4 GEMV at Qwen3-30B-A3B's two shapes, ≥ 160 SMs | NF4 decode | `GNF4_GEMV_BW=0` (`1` forces it everywhere) | `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07` |
| programmatic dependent launch, at most 8 rows | decode kernels | `GNF4_PDL=0`; `GNF4_PDL_MAX_ROWS=0` lifts the cap | `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04` |
| int4-b32 split-K planned from N alone on every part | `int4_b32._plan` | none (a constant, `SPLITK_R_TERM_MAX_SMS`) | `gnf4.kernel.k30-splitk-r-term.l4.2026-10-05`, `gnf4.kernel.k32-splitk-r-term.a4000.2026-10-05` |
| grouped_mm on sm_90; dense off sm_90 at ≤ 16 present experts | training GEMMs | `GNF4_TRAIN_GEMM=fused` | `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.route-v2`, `e4b.train.dense-route.mixtral.5090.2026-10-04` |
| prebound Triton launches | training GEMMs | `GNF4_TRITON_PREBIND=0` | `e4b.train.prebind.qwen3.5090.2026-10-05` |
| one grouping per MoE layer pass | grouping | `GNF4_HOST_REUSE=0` | `e4b.train.host-reuse.qwen3.5090.2026-10-04` |
| pinned index ring outside capture | index transfers | `GNF4_PINNED_RING=0` | `e4b.train.host-syncs.qwen3.5090.2026-10-03` |
| prefill M-tile height from the group sizes | NF4 prefill | `GNF4_PREFILL_TILE_RULE=max` | `e4b.train.prefill-tile-rule.qwen3.5090.2026-10-03` |
| lean padded LoRA delta | grouped-LoRA delta | `NF4_QLORA_LEAN_DELTA=0` | `e4b.train.lora-delta-lean.qwen3.5090.2026-10-03` |
| bucketed padding at ≥ 16,384 routed rows (`auto`) | grouped-LoRA delta | `NF4_QLORA_PAD_BUCKETS=0` | `e4b.train.pad-buckets.auto.default-decision.5090.2026-10-06` |
| bucketed calls as one compact autograd node | grouped-LoRA delta | `NF4_QLORA_COMPACT_BUCKETS=0` | `e4b.train.compact-buckets.packed-4k.5090.2026-10-07` |

---

## What changed — retired, superseded, corrected

The register keeps every retired and superseded claim with its reason or successor. The dated account is under
[What changed](STATUS-RECORD.md#what-changed--retired-superseded-corrected) in `STATUS-RECORD.md`; the ones most
often met:
- **sm_120 is not "parked":** that roadmap line is retired (`gnf4.retired.sm120-parked`).
- **"4.67× vs the grouped-bf16 execution class" is superseded** (`gnf4.kernel.comparators-v6-execution-class`): that
  backend never ran Unsloth's own kernel; quote the head-to-head.
- **#319 was a mislabelled test arm, not an f32 kernel defect.** Its open row is retired (`gnf4.open.f32-compute-modes-triton34`);
  the old advice to gate lanes with `-k "f8dot or pf8"` excluded exactly the mislabelled arms.
- **Split-K on the NF4 dot-pad decode GEMV is refuted** (K7), its claim retired (`gnf4.retired.splitk-gemv`).
- **The int4-b32 split-K R term is off on every part:** its A2000 sweep is retired (`gnf4.serve.int4-b32-splitk-row-term.a2000.2026-09-10`),
  and K30 (L4) and K32 (RTX A4000) both read OFF.
- **Refuted designs stay closed:** a fixed fraction-of-waterfall as a law, the cold-engine "free floor" premise
  (`gnf4.cold-engine.phase0-premise-refuted`) and expert prefetch (`gnf4.flagship.prefetch-closed-negative`).
- **Exact, but not levers:** K14 (`gnf4.kernel.k14-smallm-int4-gemm-refuted.5090.2026-09-11`), K17's fused split-K reduce
  and K18's grouped expert GEMV (`gnf4.kernel.k17-fused-splitk-gemv.5090.2026-09-21`,
  `gnf4.kernel.k18-grouped-expert-gemv.5090.2026-09-22`) ship opt-in or dormant, never as defaults.

---

## What is open

- **#60:** arena staging blocks ~30% of a training step; the next layer's rows are prefetchable (`gnf4.open.issues`).
- **The single-block ladder's `auto`:** experts4bit-qlora's TC1 amendment 71 is registered to read it and has not. If
  its P215, P216 and P219 hold, `auto` becomes the default.
- **The bandwidth GEMV** is unread on other families, other cards and beside experts4bit-qlora's fused B=1 stack.
- **The cold cost model:** why two gen 4 x16 hosts read different `link_eff` is unmeasured, and `cold_dest="deadline"`
  still omits the hybrid tier's mixed-layer dispatch term.
- **`docs/context-budgets.md` is rung-one only**; full-depth confirmation is pending (`gnf4.open.context-budgets-rung-two`).
- **`docs/cold-engine/STAGE3-SYNTHESIS.md`** has one outstanding correction: quote no read count from it until gate 1 is
  re-run.
- **Every non-CUDA row is a port target:** `PROJECTIONS-multiarch.md` is arithmetic, stamped before the silicon
  (`gnf4.projection.multiarch`, projected).

---

## Reading the numbers without getting them wrong

- **Quote the hardware and the baseline.** Ratios move with the card, and almost every ratio here is against this
  project's own per-expert loop or the dequant path (bitsandbytes `dequantize_4bit` per active expert, then a bf16
  matmul), not a third party's kernel, except the head-to-head that says so.
- **Benchmark on real text.** Random token ids understate this kernel (2.75× against 4.50× on prose, on a 4090)
  (`gnf4.kernel.e2e-training-real-prose`).
- **Peak VRAM does not improve.** The fused arms peak higher; only the transient held from forward to backward shrinks.
- **Quote the register that owns the number.** A model-level figure (tok/s for a named model on a named card, a
  perplexity-gate verdict) is registered in [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora)'s
  `docs/claims.json`; quote it with that register's claim ID and status. A `measured-private` status there means not
  publicly reproducible, exactly as here, where three rows are: `gnf4.serve.int4-b32-gemv`,
  `gnf4.serve.gptq-pack-int4-b32`, `gnf4.serve.decode-glue-kernels`.

---

## Where the detail is

- [`STATUS-RECORD.md`](STATUS-RECORD.md): the dated narrative to 0.43.0, every superseded reading with its reason.
  Frozen; it is not appended to.
- [`claims.json`](claims.json) ([schema](claims-schema.md)): every claim's full text, conditions, status and evidence.
- [`CHANGELOG.md`](../CHANGELOG.md): each release and the reads behind it; [`changelog.d/`](../changelog.d/) holds what
  has merged since 0.43.0.
- `kernel/RESULTS-*.md` and the `RESULTS-*.md` under `bench/`: each lane's receipt and verdict.
- [`capabilities.json`](capabilities.json), [`SOLUTIONS.md`](SOLUTIONS.md), [`KERNEL_CONTRACT.md`](KERNEL_CONTRACT.md)
  and [`TOLERANCE_CONTRACT.md`](TOLERANCE_CONTRACT.md): entry points, layouts and the fidelity bound.
