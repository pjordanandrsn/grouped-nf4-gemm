# Status — what this kernel does, what changed, what is open

**As of 2026-10-06, `grouped-nf4-gemm` version 0.42.0.** One page. The README argues; this
page states. The positions here name their entries in
[`docs/claims.json`](claims.json), which carry the evidence paths; a line
without a claim ID records an issue closure, a correction still outstanding
in a research document, or a rule for reading the numbers.

Evidence tiers, unchanged from the rest of the repo: **confirmed** =
pre-registered, stamped, blind confirmatory run; **measured** = a run
with a committed receipt; **projected** = arithmetic, not a run. One
addition, because it was being blurred: **measured-private** = the run
happened and the number is real, but the receipt lives in a private
audit tree, so *you cannot check it from this repository*. Those are
marked. Treat them as you would any unverifiable number. The vocabulary is
`status_vocabulary` in [`claims.json`](claims.json) and `evidence_vocabulary`
in [`system-manifest.json`](system-manifest.json); each position below
names its claim ID.

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
| Training backward, one launch: gradient vs the exact per-expert loop (A2000, a correctness read) | ~0.0029 relative, inside the bf16 budget | measured | `gnf4.kernel.dgrad` |
| Fused training step with it vs the per-expert loop, Qwen3-30B-A3B, 48 layers (rented RTX A6000; experts4bit-qlora's receipt) | 2.52× (1.72× without it) | measured | `gnf4.kernel.dgrad-step.a6000.2026-08-06` |

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
any card — is **supported as of #319's close**: measured on an RTX A2000
(sm_86) and an RTX 5090 (sm_120), same errors on both, 0.00% over
tolerance (`gnf4.serve.f32-arms-ran-fp8`). The
certified single-stream anchor for Qwen3-30B-A3B on the RTX 5090 class
is **7.37 ms/step ±4.2% (≈130–142 tok/s)** — the class carries 8.5%
inter-box dispersion while each box repeats itself to 0.16%, so quote
the range, not a point (`gnf4.serve.decode-anchor-5090`).

**Row-count invariance (sm_120, lane P63).** A token decoded alone and the
same token inside a 16-, 17- or 160-token call get the same bits from three
kernels, read on an RTX 5090 at Qwen3-30B-A3B's gate_up on its own
activations and routing (experts4bit-qlora#708):
- `gemv_int4_b32`: 1,544 of 1,544 rows bit-equal to their token's own call.
  Above 64 SMs its split-K plan does not depend on the row count
  (`gnf4.kernel.int4-gemv-row-invariant.5090.2026-09-24`).
- the NF4 dot-pad decode GEMV, the default decode route there: 1,544 of 1,544
  (`gnf4.kernel.nf4-dotpad-gemv-row-invariant.5090.2026-09-24`).
- `combine_rows`: every row at T = 2 to 160
  (`gnf4.kernel.combine-rows-row-invariant.5090.2026-09-24`).

Two routes differ only in summation order: the grouped int4 GEMM against the
GEMV (max row rel L2 4.4e-4) and the scalar NF4 GEMV under
`GNF4_GEMV_DOTPAD=0`, whose split-K is planned from the rows (2.4e-4)
(`gnf4.kernel.int4-grouped-gemm-reorder.5090.2026-09-24`,
`gnf4.kernel.nf4-scalar-gemv-splitk-reorder.5090.2026-09-24`). Every path was
inside its own operand model's fp64 bound. `kernel/test_row_invariance_gpu.py`
asserts the three invariant kernels with `torch.equal` on any CUDA part (all
measured; one box, one shape).

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

- **Bucketed LoRA-delta padding is the default as `auto` (0.42.0; #490, #491,
  #492).** `NF4_QLORA_PAD_BUCKETS` unset buckets a grouped-LoRA delta call that
  carries at least 16,384 routed rows; every smaller call keeps the single
  padded block, op for op. This follows experts4bit-qlora's registered rule
  (TC1 amendments 47–50, measured there as
  `e4b.train.pad-buckets.qwen3.5090.2026-10-06` and
  `e4b.train.pad-buckets.auto.default-decision.5090.2026-10-06`): on packed
  4,096-token rows buckets lowered the step and the peak, and at the field
  recipe the gate never fired. That evidence is one model (Qwen3-30B-A3B) on one
  RTX 5090; other families that reach the gate change on this package's
  correctness tests alone. Two environments are measured. Amendments 48 and 50
  read torch 2.12.1 / triton 3.7.1. Amendment 52
  (`e4b.train.pad-buckets.torch28.qwen3.5090.2026-10-06`) read torch 2.8.0 /
  triton 3.4.0. There the buckets step 0.983 [0.974, 0.993] (matched) and 0.939
  (shipped) of the single block on packed rows, the matched peak is 4.24 GB
  lower, and held-out moves by 0.0002, so the default stands there too.
  Amendment 51's slower torch-2.8 e4b (environment ratio 0.739,
  `e4b.train.h2h.unsloth.qwen3.5090.2026-10-06.packed-4k-defaults`) is
  therefore not the buckets' cost. Reported, not scored: under torch 2.8 the
  bucketed arms left the GPU idle more of the step (median utilisation 87 %
  against 97 %), so host-side time in the bucketed delta is the open lead.
  `NF4_QLORA_PAD_BUCKETS=0` restores the single block exactly.
- **#408 is closed: the MXFP4 prefill combine returns the same bits on every
  call (#410, 2026-09-28, RTX A2000).** `Mxfp4PipelinedGptOss._forward_prefill`
  summed each token's expert outputs with `out.index_add_`, whose CUDA float
  atomics reorder a token's terms from call to call. Replayed at Kimi-K3
  geometry, 50 identical calls gave 50 different fp32 outputs. It now adds
  through `_index_add_ordered_`, which gives each `index_add_` call unique rows.
  The result is bitwise the sequential sum in ascending expert id: 1 distinct
  output in 50 at T = 6, 90 and 512, at 287–330 µs more per chunk
  (`gnf4.kernel.mxfp4-prefill-combine-ordered.a2000.2026-09-28`, measured). In nine
  Kimi-K3 processes (experts4bit-qlora#766) it was the only run-to-run
  difference in the forward, the drift #761 recorded. The MXFP4 QLoRA fused path had the same pattern, in bf16
  (#409). #416 fixed it the same way, and it ships in 0.34.0 (0.33.8 was prepared but never published).
- **#393 is closed, answered with an accuracy contract, not a bitwise one
  (lane B393, 2026-09-23, RTX 5090).** `combine_rows` (the fused MoE top-k
  weight-and-sum experts4bit-qlora runs on every MoE layer by default) and
  `reduce_partials` (the fused split-K reduce) were tested to a tolerance and
  documented as taking the torch chains' order. Measured over 414 census cases
  at the served shapes, neither is bitwise its chain: `combine_rows` differs
  in 95 of 144 cases (392 of 9.07 M elements) and `reduce_partials` in 49 of
  270. Both are within the error bound of a correct fp32 summation in every
  case, at the same max ratio as torch's own chain (0.996), so each is as
  accurate as what it replaced.
  - `reduce_partials` is bitwise the slot-order sum.
  - `combine_rows` is the slot-order sum with a fused multiply-add, exactly so
    on sm_86 and on sm_120. That is a follow-up diagnostic, not the reading.
  - **Corrected 2026-09-24:** the first read said neither the kernel's bits nor
    torch's chain are the same across architectures. That was an inference from
    counts, and output hashes refute it: all four outputs are bit-identical on
    sm_86 and sm_120 in 144/144 cases. One box's census disagreeing with two
    others stays unexplained.

  ([`RESULTS-b393-combine-reduce-bitwise.md`](../kernel/RESULTS-b393-combine-reduce-bitwise.md);
  `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23` and
  `gnf4.kernel.reduce-partials-slot-order.5090.2026-09-23`, measured.) The
  docstrings and the two tests now state and assert that contract. No kernel
  changed. **Sized end to end 2026-09-24** (experts4bit-qlora#708, lane P63,
  Qwen3-30B-A3B, T = 1 decode, `E4B_FUSE_COMBINE=0` against the default): on
  the NF4 stack KL 1.18e-04 nats/token and no argmax flips in 160 positions; on
  the int4 stack KL 1.70e-02 and 7 flips (4.4 %), between the lane's
  registered bands.
- **#386 is closed: every carrier of the int64 expert-base promotion is now
  observed past its own boundary.** The gathers (`host_gather`,
  `mxfp4_pipelined`, `mxfp4_residency`) are int64-word-addressed, and the
  `fp8_kv` appenders are byte-addressed. KERNEL_CONTRACT listed both families
  as covered carriers, but no boundary test called either one.
  `kernel/test_offset_boundary_interp.py` now puts all five kernels past their
  boundary on CPU, under the Triton interpreter, in CI. The shipped kernels
  pass 5/5, and a copy with the five straddling promotions removed fails 5/5,
  each at the wrapped address
  (`gnf4.kernel.boundary-gathers-appenders.interp.2026-09-23`, measured;
  receipts in `kernel/receipts-386/interp/`). No kernel changed.
- **#374 is closed: the word-addressed decode routes are observed past their
  own 2^31 boundary (lane B374, 2026-09-23, RTX 5090).** The wide-load and
  dot-pad routes — dot-pad is the default decode route on >= 160-SM parts at
  its census shapes —
  address the stack in 32-bit words and wrap at 2^31 words (8 GiB), which the
  2026-09-05 boundary suite's byte geometry never reached; its green wide and
  dot-pad arms were not coverage of that boundary.
  `kernel/test_offset_boundary_words_gpu.py` puts both routes past it in a
  real 16 GiB buffer with a decoy at the wrapped address: 4/4 cases pass on
  the shipped kernels, and 4/4 read the decoy when the six int64 promotions
  are stripped (pre-registered;
  [`RESULTS-b374-word-boundary-gpu.md`](../kernel/RESULTS-b374-word-boundary-gpu.md);
  `gnf4.kernel.word-boundary-wide-dotpad.5090.2026-09-23`, measured). No
  kernel changed.
- **#73 and #58 are closed; this page listed them as open until 2026-09-23.** #73 (the K3 fetch's
  host copy at ~71% of a layer) was fixed in 0.12.0 by scattering each arena row into per-segment
  staging with `preadv` (#74, #76, #79): 1113 → 56.8 ms per layer on real K3 bytes, 78% of that
  box's device bandwidth. #58's "8 requests where 2 would do" was answered by its own trace: six of
  the eight calls per layer are cache hits costing 0.02 s/step, and the ~17 GB read per step is the
  routing floor at `hot_rows=128`. The training-step stall it was pointing at is #60, still open.

- **#319 is closed, and it was never a kernel defect — the f32 shape arms were not running an f32
  kernel.** `test_fp8_paged_attn.py`'s `_modes()` built its f32 arms with no `compute` kwarg, which was
  right only while an unset `compute` meant f32. Since RESULTS-m3-default-on the default is
  capability-conditional and resolves to **fp8 on sm_89+**, so on those cards `split` and `packed` ran
  the fp8 kernel while `_close` judged them at the f32 tolerance (2e-2 against the fp8 path's own
  1.5e-1). On an RTX 5090, `compute_counts()` after one call per arm reads **`{'f32': 0, 'fp8': 4}`**,
  and the `split`/`f8dot` and `packed`/`pf8` pairs return byte-identical tensors. With every arm naming
  its mode the suite goes **27 failed → 93 passed** on that same card, and the f32 errors there are the
  same bf16 output ULPs an RTX A2000 reports (0.003906 / 0.007812 / 0.015625). The claim
  `gnf4.open.f32-compute-modes-triton34` is **retired**, replaced by `gnf4.serve.f32-arms-ran-fp8`;
  `kernel/RESULTS-319-f32-precision.md` carries the measurements. **The advice this page used to give —
  gate lanes on those boxes with `-k "f8dot or pf8"` — excluded precisely the arms that were
  mislabelled, and should not be followed.**

- **0.33.0 — two int4-b32 GEMV variants ship opt-in, and neither is a default.** K17's fused split-K
  reduce (`gemv_int4_b32(..., fused_reduce=True)` / `GNF4_GEMV_FUSED_REDUCE=1`) and K18's grouped expert
  GEMV (`gemv_int4_b32_grouped`) are both bitwise the served GEMV and neither pays: K17 saves at most
  2 µs per call at R=1 and made the consumer's decode step slower at B=1 and B=16, and K18 is 1.49× slower
  on recorded B=16 routing (the K17 and K18 reads below). With `FUSED_REDUCE=0` the served kernel's
  body is byte-for-byte the 0.32.1 one, and no default changes. The same release closes #87 with a CPU boundary test CI runs, and records that #319 was a
  mislabelled test arm (above).

- **0.32.1 — the grouped-LoRA delta's `auto` rule is structural.** Until 0.32.0 `auto` sent any adapter
  call past a 4× padding-waste ratio to the per-expert Python loop. The consumer's P46
  (experts4bit-qlora `bench/p46/RESULTS-p46.md`, Qwen3-30B-A3B at its field training recipe, RTX 5090)
  measured that guard choosing the loop for ≥ 85 % of calls every step — 24.46 s/step where the padded
  `bmm` path forced on trained the same tokens to the same loss (held-out Δ +0.0007 nats) at the same
  peak VRAM in 4.22 s/step. The loop is launch-bound; the flops padding wastes are rank-r. `auto` now
  pads unless the padded block would exceed `NF4_QLORA_PAD_BYTES_LIMIT` (2 GiB); the ratio guard is
  opt-in (`NF4_QLORA_PAD_WASTE_LIMIT`); `LORA_PAD_WASTE` records the skew for the census. No consumer
  training position moves from this — the consumer re-runs its head-to-head on this cut and quotes from
  that receipt.

Kept here because a claim that quietly disappears is worse than one that
was wrong.

- **`sm_120` is no longer "parked".** The README's roadmap once said
  three cloud provisioning failures parked Blackwell work (that line is
  gone; this page and the retired claim `gnf4.retired.sm120-parked` record it). That was true
  in July; since 0.15.0 the RTX 5090 has been the *primary* serving
  target — the M=1 config retune, the sm_120 census, the decode anchor,
  the M3 defaults, the int4 lanes and the paged attention were all
  measured on rented 5090s. **Retired as stale** (`gnf4.retired.sm120-parked`).
- **The "4.67× vs the grouped-bf16 execution class" number is
  superseded.** That backend never executed Unsloth's own kernel, and
  the proxy it did run is 1.33× slower than the real thing — so the
  comparison was against a weaker opponent than the label implied. The
  head-to-head (1.70× / 2.79×) replaces it. The old number is kept in
  the register as superseded, with that caveat attached, not rescaled
  (`gnf4.kernel.comparators-v6-execution-class`, superseded by
  `gnf4.kernel.h2h-unsloth`).
- **Split-K on the NF4 dot-pad decode GEMV is refuted** (K7: flat at
  `gate_up`, ~14% worse at `down`). That plan ships dormant behind
  `GNF4_GEMV_SPLITK` *as the evidence*
  (the retired claim `gnf4.retired.splitk-gemv`); the scalar NF4 route's
  split-K (`_decode_plan`) and the int4-b32 GEMV's split-K are separate,
  shipped plans.
- **The int4-b32 split-K planner takes the row count (0.31.0).**
  `_plan(N, K)` sized split-K from `N` alone, so at large `R` every expert
  projection ran a configuration chosen for a batch it was not in and paid
  the partials reduce to do it. `R` and the SM count are now threaded, as
  the NF4 sibling planner always had them. `R < 16` returns exactly the old
  plan, so B=1 decode — which calls at `R = top_k` — and the licensed serve
  configuration are untouched by construction. The term's constant was
  tuned by a 48-cell timing sweep on the A2000, and the row that carried
  that sweep is retired
  (`gnf4.serve.int4-b32-splitk-row-term.a2000.2026-09-10`, retired 2026-10-05:
  an A2000 timing; the A2000 is a correctness-only testbed). That sweep also
  skipped the reduce launch the served path pays at sk = 1 (K30 Amendment 1).
  **K30 read the term on a rented NVIDIA L4 (58 SMs) on 2026-10-05: OFF.**
  Over the 24 cells at R ≥ 16 the R-aware pick costs 1.0101× the N-only
  pick's summed time, and 1.1781× at its worst cell (qwen3_moe gate_up at
  R = 128, where it drops the split to sk 1). KEEP needed ≤ 0.97 and ≤ 1.02
  (`gnf4.kernel.k30-splitk-r-term.l4.2026-10-05`, measured;
  `kernel/RESULTS-k30-splitk-r-term-l4.md`). By the registered rule the term
  is off on every part (`SPLITK_R_TERM_MAX_SMS = 0`), so every card plans
  N-only at every R, as the 5090 already did after its step-level read of
  1.0064; MXFP4 keeps the N-only plan. A test enforces the default; the
  mechanism stays, tested under a monkeypatched gate. **K32 read the second
  ≤ 64-SM architecture on a rented RTX A4000 (48 SMs, sm_86): OFF too**, by
  its worst cell (1.0965×, olmoe down at R = 16) although its summed time
  passed (0.9396×) (`gnf4.kernel.k32-splitk-r-term.a4000.2026-10-05`,
  measured; `kernel/RESULTS-k32-splitk-r-term-a4000.md`). So the OFF stands
  on both architectures read, which lose at opposite ends of R.
- **K14 is refuted: at M=16 no shipped int4 arm beats dequant-then-GEMM
  on the attention projections.** On the 5090 the grouped int4 GEMM at
  its best swept configuration is 1.12–2.00× slower than the bf16 path
  and the int4 GEMV 1.46–2.40× slower; the registered mix saving is
  0.000 ms/step against a 0.5 ms bar. `q_proj`'s bf16 path is at 106 %
  of the streaming ceiling and `k`/`v` are launch-bound, so a roofline
  int4 projection kernel would be worth ≈ 1.29 ms/step and nothing shipped
  realises it (`gnf4.kernel.k14-smallm-int4-gemm-refuted.5090.2026-09-11`,
  measured).
- **K17 read (2026-09-21, RTX 5090): folding the int4-b32 GEMV's split-K reduce into its own launch is exact
  (24/24 rows bitwise, counter re-arms under graph replay) but was not the cost** — at R=1 the removed
  `_reduce_partials` launch was overlapped in the graph (savings 0–2 µs; P2 refuted by its own clause), and the
  fused epilogue is 6–12 % slower at R=128. Ships **opt-in** (`GNF4_GEMV_FUSED_REDUCE=1`), default off at every
  R. `kernel/RESULTS-k17-fused-splitk-gemv.md`; register row
  `gnf4.kernel.k17-fused-splitk-gemv.5090.2026-09-21`; the consumer's step-level read is experts4bit-qlora lane P57.
- **K18 read (2026-09-22, RTX 5090): a grouped split-K int4-b32 expert GEMV — each expert's weight slice loaded once
  for up to 4 of its rows — is exact and slower.** On experts4bit-qlora P60's recorded Qwen3-30B-A3B B=16 routing it
  runs 9.689 ms/step against the served GEMV's 6.520 (P2 refuted), and 1.00–1.65× the served call at R = 8/16 (P3
  refuted), worst where nothing can be shared. Not a lever; `gemv_int4_b32_grouped` stays dormant as the evidence.
  `kernel/RESULTS-k18-grouped-expert-gemv.md`; register row `gnf4.kernel.k18-grouped-expert-gemv.5090.2026-09-22`.
- **K20 read (2026-10-01, RTX 5090): K19 (`gemm_int4_b32_grouped_smallm`, the grouped small-M int4-b32 GEMM over
  expert-major 16-row tiles) at BLOCK_N 32 / KC 256 serves P60's recorded B=16 routing at 0.736× the served int8 GEMV
  route** (5.200 vs 7.062 ms/step including the tile build; 89 % of the byte floor). The shipped plan (64/128, chosen
  on the A2000) read 0.879×. Compiled outputs are bit-identical across plans, so 32/256 is now K19's default. K19 stays
  opt-in in the consumer until its end-to-end read (experts4bit-qlora P88).
  `kernel/RESULTS-k20-k19-plan-sweep-5090.md`; register row `gnf4.kernel.k20-k19-plan-sweep.5090.2026-10-01`.
- **K26 read (2026-10-01, RTX 5090): the per-nibble codebook lookup is about 80 % of K25's time** at the NF4 families'
  B=16 shapes (`nibble − 8` in its place: 0.223 / 0.207 of K25, Granite / OLMoE). An exact select-tree decode over
  the 16 fp32 codebook values is bit-identical to K25 and runs at 0.373 / 0.383 of it (2.189 / 3.794 ms/step, about
  0.39× the served NF4 GEMM), so K25 takes it. The served kernel's own path with the tree is not bit-identical.
  `kernel/RESULTS-k26-nf4-decode-ablation.md`; register row `gnf4.kernel.k26-nf4-decode-ablation.5090.2026-10-01`.
- **K27 read (2026-10-01, RTX 5090): K25 with the select tree at the served kernel's weight precision (fp32 weights,
  TF32 MMA) runs at 0.448 / 0.502 of the served NF4 GEMM** (Granite / OLMoE B=16 shapes), its rms error equal to the
  served kernel's. In bf16 it runs at 0.317 / 0.325, at 1.35× the served error. TF32_PATH: the next lane reads K25-tree
  in TF32 end to end on both families. `kernel/RESULTS-k27-nf4-tree-precision.md`; register row
  `gnf4.kernel.k27-nf4-tree-precision.5090.2026-10-01`.
- **K28 read (2026-10-04, RTX 5090): programmatic dependent launch (`GNF4_PDL=1`) saves 0.323 µs per gnf4
  kernel** in a CUDA-graph replay of the served Qwen3-30B-A3B B=1 layer's 912 gnf4 kernels (2.658 → 2.363 ms,
  ×1.125), with every output bitwise identical; 0.217 µs with the served step's ATen stand-ins interleaved.
  LEVER: experts4bit-qlora reads it served next, and the switch stays opt-in until then.
  `kernel/RESULTS-k28-pdl-decode-chain.md`; register row `gnf4.kernel.k28-pdl-decode-chain.5090.2026-10-04`.
- **`GNF4_PDL` on by default, capped at 8 rows (2026-10-04).** experts4bit-qlora's served lanes: P112 closed VOID
  twice; P113 read CAP_DEFAULT on SC1's int4 serving configuration (RTX 5090). Capped, it decodes identical tokens
  1.0404× as fast with one request and 1.0000× with 16; uncapped, 1.0401× and 0.9787×. Register row
  `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04` (experts4bit-qlora). `GNF4_PDL=0` turns it off,
  `GNF4_PDL_MAX_ROWS=0` removes the cap.
- **K33 read (2026-10-07, RTX 5090): the bandwidth-targeted NF4 decode GEMV (`GNF4_GEMV_BW=1`, `prmt32`) runs
  Qwen3-30B-A3B's single-row expert projections at 0.341× the served dot-pad route** over all 48 layers (2.818 →
  0.961 ms; gate_up 0.332, down 0.357), at 0.74 / 0.63 of the copy floor, `prmt32` bitwise the exact tree decode.
  Granite and OLMoE read 0.22–0.27× their scalar GEMV. Those figures are with K33's swept per-shape plans
  (`GNF4_GEMV_BW_PLAN`) and `GNF4_PDL=0`: the shipped default plan read about 0.37× on the Qwen3 pair in the same sweep,
  and PDL, on by default for these 8-row calls, made Qwen3's down projection 1.17× slower. LEVER: experts4bit-qlora
  reads it served next (P116), and the switch stays opt-in until then. `kernel/RESULTS-k33-nf4-decode-gemv-bw.md`; register row
  `gnf4.kernel.k33-nf4-decode-gemv-bw.5090.2026-10-07`.
- **`GNF4_GEMV_BW=auto` by default at Qwen3-30B-A3B's two shapes on >= 160-SM parts (2026-10-08).** experts4bit-qlora's
  served lane P116 read DEFAULT_ON on an RTX 5090: on the default `serve_paged` server, one request decoded 1.2417× as
  fast (step 10.22 → 8.22 ms) and 16 requests were unchanged, within P110's teacher-forced quality bar
  (experts4bit-qlora#1336). `_BW_SHAPES` names those two shapes and `_BW_PLANS` carries K33's selected plans. Other
  shapes and smaller parts keep dot-pad or the scalar GEMV; `GNF4_GEMV_BW=0` turns it off, `1` forces it everywhere.
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
- **Register bookkeeping corrected (0.30.2).** `gnf4.serve.m3-defaults-on`'s
  sentence described the fp8 predicate as `k_groups in (1,2,4)` — the
  predicate as it stood at the 2026-08-27 run; `fp8_compute_unsupported`
  has admitted `(1, 2, 4, 8, 16)` since 0.26.0 and the sentence now says
  so, with the measurement unchanged. Every `evidence` entry in
  `claims.json` is now a path that resolves at HEAD (at the time through
  structured forms for changelog sections, globs and cross-repository
  receipts; 0.33.2's shared schema replaced the first two — a changelog
  section is now a `CHANGELOG.md#<anchor>` location and globs are refused —
  and kept the cross-repository form: [`claims-schema.md`](claims-schema.md)); every measured, measured-private
  and confirmed row carries an ISO `measured_on` taken from its receipt or,
  where the receipt states no run date, the receipt's first commit;
  the retired `gnf4.retired.splitk-gemv` carries its `retired_reason`; and
  `scripts/check_claims_register.py` + `scripts/check_readme_claims.py`
  hold the register and this page to it in CI. No number, gate or verdict
  moves.

---

## What is open

- **The cold cost model's link term is now a per-host measurement (#400, #402; lane P69 read 2026-09-24).**
  experts4bit-qlora's lane P66 scored `cold_deadline.gpu_us` against a real gather on a gen 4 x16 RTX 5090 and found
  bytes-over-`b_link` under-predicting it 1.57–2.02x: the gather runs at the box's single-copy H2D rate, and the
  calibration's `b_link` is the 40-deep back-to-back rate. #402 gives `Costs` a `link_eff` read from a new
  `bench/calibrate.py` probe (schema `gnf4-hybrid-calib/2`: the same 64 MB copy one at a time) and a consumer-passed
  `gpu_us_fixed` for the path's own fixed per-call kernels; a /1 blob is refused unless `link_eff` is passed
  explicitly. Measured so far: **0.64** on P66's 5090 host (EPYC 7C13), **0.873 / 0.960** on P69's (EPYC 7663;
  `gnf4.calib.link-efficiency.5090.2026-09-24`, the registered [0.55, 0.75] REFUTED there). So the factor is the
  box's, not the card class's, and it lives in each box's own blob. Open: why two gen 4 x16 hosts differ (idle link
  state, NUMA placement of the pinned buffer, the DMA engine — none measured), and the hybrid tier's mixed-layer dispatch term, which `cold_dest="deadline"` still omits.
- **#60** — arena staging blocks ~30% of a training step; the next
  layer's rows are prefetchable (guarded by a CUDA event, with a
  `hot_rows` floor of two layers' experts).
- **#71 closed.** `PINNED_ROW_FACTOR`'s 1.9 was PyTorch's caching host
  allocator rounding each pinned request up to a power of two, not a per-byte
  premium. `capacity_for_bytes` models it exactly (`pinned_request_cost`). Every
  pinned request is measured at 0.4–1.0 % over its power of two, in both regimes:
  - cgroup v1 (`kernel/receipts-71/`);
  - cgroup v2 (lane K29, `gnf4.kernel.k29-pinned-charge-cgroup-v2.5090.2026-10-04`).

  (#60 is `gnf4.open.issues`.)
- **`docs/context-budgets.md` is rung-one only** (A2000-measured
  KB/token); full-depth real-weight confirmation is pending and the K3
  row is a declared gap. Its own text forbids promoting pending rows to
  the README — that still holds (`gnf4.open.context-budgets-rung-two`).
- **`docs/cold-engine/STAGE3-SYNTHESIS.md` carries one correction
  outstanding**: gate 1's published read counts are uncorrected, and no
  read count in that document should be quoted until it is re-run.
- **A Marlin-class GEMM on the int4-b32 format is the measured kernel
  lane.** K15: vLLM 0.28.0's Marlin runs Qwen3-30B's `q_proj` / `o_proj`
  1.62× / 1.99× faster than this package's bf16 dequant path at M=16 on
  the same 5090, and at matched bytes (g32) still wins — the kernel, not
  the format. Worth 0.58–0.81 ms/step, inside the band the
  pre-registration fixed as inconclusive; `k`/`v` unmeasured (workspace
  sizing, fixed after the budget closed); nothing adopted
  (`gnf4.kernel.k15-marlin-comparator.5090.2026-09-11`, measured).
- **K16 answers K15: a Marlin-class GEMM on the int4-b32 format exists and
  is measured.** `int4_smallm.gemm_int4_b32_smallm` (in-register dequant,
  per-block scaling in the tile, bf16 MMA over fat K chunks, fused split-K)
  at M=16 on the same 5090 runs `q_proj` in 6.35 µs and `o_proj` in 6.39 µs
  — 1.63× / 2.91× faster than this package's bf16 dequant path and at
  1.00× / 0.77× of K15's Marlin rows; `k`/`v` sit within 1 µs of the
  4.61 µs launch floor. P1–P3 of the pre-registration hold; P4 (a
  single-scale control) was NOT tested — the bench carried no control arm;
  P5 (≥ 0.4 ms/step at B=16) HOLDS in the consumer: −1.06 ms/step on the
  timed B=16 step (experts4bit-qlora `bench/k16/RESULTS-k16-p5.md`), and
  the route is the consumer's default from 0.36.2. Nothing in this package
  routes to it on its own
  (`gnf4.kernel.k16-smallm-int4-gemm.5090.2026-09-19`, measured).
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
