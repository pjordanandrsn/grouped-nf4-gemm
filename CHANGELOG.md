# Changelog

## Unreleased

Changes merged since the last release are one file each in [`changelog.d/`](changelog.d/); the release moves them into its section here. To add an entry, add `changelog.d/<pr-or-slug>.md`; never edit this section by hand.

## 0.41.0 — 2026-10-05 — GNF4_TRITON_PREBIND on by default (experts4bit-qlora TC1 amendments 26/30: Qwen3-30B-A3B training step 0.973 matched / 0.980 shipped on an RTX 5090, bit-identical); prebound launches cover triton 3.7

**0.41.0.** One default changes, by a rule registered and read in experts4bit-qlora (TC1 amendments 26 and 30):
`GNF4_TRITON_PREBIND` is on.

- **Prebound launches are the default.** The training GEMMs (the fused forward's M-tile, the dgrad, and the route's
  dequant) launch without Triton's per-call argument binding. They run the same compiled kernels, so outputs are
  bit-identical.
  - On an RTX 5090, with e4b's own `E4B_TRITON_PREBIND` on alongside, Qwen3-30B-A3B's training step was **0.973×** on
    the matched arm and **0.980×** [0.957, 1.003] on the shipped arm, with held-out loss unchanged.
  - P53, P54 and P55 HELD (`e4b.train.prebind.qwen3.5090.2026-10-05`). The gain is small, and the shipped interval
    reaches 1.0.
  - `GNF4_TRITON_PREBIND=0` restores Triton's own launch.
- **Triton 3.7 is covered** (#471), alongside 3.4 and 3.6. Any other Triton release keeps Triton's own launch, as do
  launch hooks, callable grids and changed globals.
- `docs/system-manifest.json` is unchanged.

### The training GEMMs' prebound launches (`GNF4_TRITON_PREBIND`) cover triton 3.7

- **Why.** torch 2.12 and Unsloth's environment ship triton 3.7.1, where `prebind` returned the kernel itself: under the default flag
  the value memos engaged there, but the fused forward, the dgrad and the route's dequant kept Triton's own launch.
- **What.** `PREBIND_TRITON` gains `(3, 7)`, with two branches that only 3.7 takes. Nothing changes under 3.4 or 3.6, and no default
  changes.
  - A registered compiler-stages hook (`knobs.runtime.add_stages_inspection_hook`) takes Triton's path: 3.7 adds its pipeline hash to
    the kernel key.
  - A `FutureKernel` is never kept. Under `AsyncCompileMode`, the 3.7 launch that compiles a key returns that proxy rather than the
    `CompiledKernel` (3.6 resolves it first); a later launch keeps the `CompiledKernel` Triton's cache then holds.
- **Read against triton 3.7.1's source.** The launch is the call 3.6 makes.
  - `JITFunction.run` passes `CompiledKernel.run` the same positional list: grid, stream, function, packed metadata, launch metadata,
    the enter and exit hooks, then every parameter in signature order.
  - Unchanged from 3.6.0: the binder, `compute_cache_key`, `CompiledKernel`, and the native specializer (`python/src/specialize.cc`,
    byte-identical). A tensor is still keyed on its dtype and 16-byte alignment, an integer on `== 1`, `% 16` and its width.
  - Unchanged knobs: `runtime.debug` (read once, at import), the launch hook chains' `.calls`, `compilation.instrumentation_mode`.
  - The NVIDIA launcher is now one C entry point instead of a module generated per signature. It takes the same arguments, skips
    constexprs by annotation and calls no hook passed as `None`.
- **Measured** on the RTX A2000 box's host (Xeon W-1250, 12 threads, load average 13–31, the bench at nice 10), torch 2.12.1 /
  triton 3.7.1. Host µs per call, each timed after a synchronize; median of 300 (whole calls) or 1,500 (`dequant_groups`)
  interleaved pairs; three runs. Qwen3-30B-A3B's experts, as in 0.40.0's entry. Flag off → on is the whole flag. Launchers only
  keeps the value memos on in both arms and differs only in the launch: what this adds under triton 3.7.

  | call | flag off → on | launchers only: Triton's → prebound |
  |---|---|---|
  | `FusedGroupedNf4.apply`, fused route, gate_up / down | 439–483 → 334–377 / 489–526 → 372–399 | 380–446 → 363–411 / 376–473 → 350–438 |
  | `FusedGroupedNf4.backward`, dgrad kernel, gate_up / down | 320–384 → 255–295 / 327–396 → 254–300 | 291–348 → 261–313 / 293–352 → 254–323 |
  | `dequant_groups` (launch and allocation, 16 groups), gate_up / down | 68–76 → 57–61 / 63–79 → 51–63 | 64–95 → 53–78 / 56–81 → 47–68 |

  Launchers only, a pair saves 16.5–52.1 µs on the forward, 20.3–39.5 on the dgrad and 9.4–17.6 on `dequant_groups`. Triton
  3.7.1's own launch is cheaper than 3.4's or 3.6's, so the saving is smaller than 0.40.0's (those runs were on another day, at
  another load).
- **Tests** under triton 3.7.1 on that A2000. `kernel/test_triton_prebind.py`: 16 passed. Two new tests cover the 3.7 branches, and
  each fails with its branch removed. The suites over these modules pass the same flag off and on: 489 passed, 25 skipped; 28
  passed in `test_shape_feasibility.py`.

### Default: the training GEMMs' prebound launches (`GNF4_TRITON_PREBIND`) are on; `=0` turns them off

- **Why.** experts4bit-qlora's TC1 amendments 26 and 30 read the prebound launches, together with e4b's own (`E4B_TRITON_PREBIND`),
  against the flags off on one RTX 5090 each, triton 3.4.
  - The training step reads 0.973× on the matched arm (fp32 adapters, `tc1-5090-69`).
  - It reads 0.980× on the shipped arm (bf16 adapters, 60-step runs, `tc1-5090-73`).
  - Held-out moves by 0.0012 or less.

  The registered decision rule (P53, P54, P55 held) makes both defaults on.
- **What.** `prebind_requested()` now reads `GNF4_TRITON_PREBIND` as on unless it is `0`. Values are unchanged: the prebound launch runs
  the same compiled kernel Triton's own lookup returns. A Triton release other than 3.4 or 3.6, a launch hook, a pre-run hook or a
  callable grid still takes Triton's own path.

## 0.40.0 — 2026-10-05 — pinned-slot fence for queued device copies (#60); opt-in GNF4_TRITON_PREBIND for the training GEMMs (bit-identical)

**0.40.0.** No default changes. One correctness fix and one opt-in.

- **Fix: pinned slots are fenced against queued device copies** (`nvme_residency`, #60).
  - A non-blocking copy out of a pinned `ColdTier` slot is queued when `segment_into` returns. A later fill into that
    slot could overwrite bytes the copy had yet to read, delivering another row's bytes under the right row IDs.
  - `segment_into` now records a CUDA event after its non-blocking copies, and a fill waits on its slot's fence.
    `stats()` gains `fence_waits` and `fence_wait_ns`.
  - Shown on an RTX A2000 with a mutation arm: with the stream held, the unfenced refill landed first.
- **Opt-in: `GNF4_TRITON_PREBIND=1`.** The training GEMMs (the fused forward's M-tile, the dgrad, and the route's dequant)
  launch without Triton's per-call argument binding, and their per-call host work is reused by value. Outputs are
  bit-identical.
  - Measured on the A2000 box's host: `FusedGroupedNf4` forward 633–654 → 433–452 µs per call, and dgrad 448 → 320 µs.
  - Not measured on a training step. experts4bit-qlora's TC1 amendment 26 is registered to read it, and the default
    stays off until a registered A/B licenses it.
- `docs/system-manifest.json` is unchanged.

### Opt-in: the training GEMMs launch without Triton's per-call argument binding, and their per-call host work is reused by value (`GNF4_TRITON_PREBIND=1`)

- **Why.** experts4bit-qlora's training step on an H100 is host-bound (device busy 0.46 in its profile). On the host, a
  `FusedGroupedNf4` call pays Triton's launcher, which binds and specializes every argument and formats a string key. It also pays
  per-call work that repeats for one grouping: an int-converted tuple key over 128 expert ids, the M-tile cost rule, and the route's
  capability query and plan (with a cumsum launch).
- **What.** Off by default; the flag is read when the modules are imported.
  - `_triton_shim.prebind(kernel)` lets the first launch of each specialization go through Triton and keeps the compiled kernel it
    returns. Its key is what Triton specializes on: each tensor's dtype and 16-byte alignment, each integer's value, every other
    argument's type and value, the launch options, the device and the debug knobs. Later launches with that key call the kernel's own
    launcher. `_gemm_nf4_grouped` (the training forward's M-tile path), `_dgrad_nf4_grouped` and `nf4_route._dequant_groups_kernel`
    launch through it. Off, `prebind` returns the kernel itself.
  - `nf4_grouped._ValueMemo` keeps recent results keyed on host int lists compared by value (one C-level list `==`). Behind it,
    `to_device_i32`'s upload memo is looked up by value, `_prefill_block_m_cost` runs once per grouping, and the grouped_mm route's
    `_plan` (expert ids and offsets) is shared by a layer's gate_up and down, forward and dgrad.
  - The grouped_mm route reads the device capability once per indexed device.
- **Unchanged values.** Same compiled binaries, arguments and stream; outputs are bit-identical. Triton's own launch serves a Triton
  release other than 3.4 and 3.6, a registered launch or pre-run hook, a callable grid, a changed global the kernel reads, and an
  argument of another type. Under triton 3.4, `TRITON_DEBUG` is read once per kernel rather than at every launch (3.6 itself reads
  it once). The value memo stores only all-int snapshots, so a hit is what the full path would build. It is never used under stream
  capture, and `GNF4_HOST_REUSE=0` turns it off with the existing memo.
- **Measured** on the RTX A2000 box's host: a Xeon W-1250 at load average 18–44 on its 12 threads, the bench at nice 10. Host µs per
  call, each timed after a synchronize; median of 300 (whole calls) or 1,500 (pieces) interleaved off/on pairs; two runs per torch.
  Qwen3-30B-A3B's experts: E=128, top-8 over 4,096 tokens (sequence 2048 × micro-batch 2), so 128 groups and 32,768 rows; gate_up
  N=1536 K=2048, down N=2048 K=768.

  | call | torch 2.8.0 / triton 3.4.0: off → on | torch 2.11.0 / triton 3.6.0: off → on |
  |---|---|---|
  | `FusedGroupedNf4.apply`, fused route, gate_up / down | 633–654 → 433–452 / 582–617 → 408–423 | 621–640 → 460–482 / 602–610 → 437–461 |
  | `FusedGroupedNf4.backward`, dgrad kernel, gate_up / down | 448–450 → 318–322 / 408–437 → 293–310 | 433–470 → 327–356 / 426–430 → 330–334 |
  | grouped_mm route: `_plan`, gate_up / down | 105–112 → 12–13 / 87–90 → 10–11 | 97–112 → 10–13 / 99–104 → 11–12 |
  | grouped_mm route: `_refuse_unless_supported` | 10–13 → 3.5–4.6 | 6–12 → 2.4–4.4 |
  | `dequant_groups` (launch and allocation, 16 groups), gate_up / down | 131–135 → 78–83 / 85–118 → 52–71 | 140–153 → 110–124 / 95–100 → 79 |

  The route's whole call is not timed: torch 2.8 has no `_grouped_mm` kernel for sm_86, and torch 2.11's synchronizes there.
- **Not measured.** No training step, and no H100 or RTX 5090 host. The default stays off until a registered A/B reads it.
- **Tests.** `kernel/test_triton_prebind.py` (wired into CI's GPU-labelled step). Against the flag off, the fused forward, the dgrad
  and the route's dequant match under `torch.equal`: odd N, K at and off 128, few and many groups, misaligned activations. Every
  prebound launch is the very compiled kernel Triton's lookup returns. A launch hook, a callable grid and an unsupported Triton
  release take Triton's path. On CPU, the value memo answers only what the full path would build, the M-tile memo is
  value-identical, and the capability is read once per indexed device. On an RTX A2000 under torch 2.8.0 and 2.11.0, flag off and
  on: 14 passed in it, and the suites over these modules pass the same either way (428 passed, 21 skipped; 28 passed in
  interpreter mode). The shape-feasibility test now also guards the name the forward launches through.

### `nvme_residency`: pinned slots are fenced against queued device copies (#60)

- **The race.** A non-blocking copy out of a pinned slot is queued, not done, when `segment_into` returns. A later fill
  into that slot could start its disk read before the copy ran: another layer's demand `ensure`, or a speculative
  prefetch. The copy then delivered the new row's bytes under the old row's IDs, finite and plausible, and no row-ID
  guard can see it. The routed training stage was ordered implicitly by its routing sync. The bulk fallback, and any
  staging made sync-free, were not.
- **The fix.**
  - `segment_into` records a CUDA event after its non-blocking copies into a CUDA destination, and fences the slots
    they read (`ColdTier.fence`).
  - A fill waits on its slot's fence before the read starts, each event once. `fence_waits` and `fence_wait_ns` in
    `stats()` count it.
  - An already-complete event costs a no-op `synchronize()`.
- **Shown on an RTX A2000.** A sleep kernel held the stream so the copy was still queued, then the same slots were
  refilled with another layer's rows. With the fence, the copy read the right layer; with fills not waiting (the
  mutation arm), it read the refill's bytes. CPU tests pin the ordering and include a mutation arm of their own.
- **Not in this change.** Whole-layer training prefetch, which #60 first proposed, was measured on 2026-08-13 and
  refuted (experts4bit-qlora `bench/host-ram-ceiling/RESULTS-prefetch.md`: 14.4 % slower at 1.9x the bytes). The
  fence was the hazard #60's thread found, and it applies to any non-blocking reader of pinned slots.

## 0.39.0 — 2026-10-04 — `GNF4_TRAIN_GEMM=auto` takes the dense route off sm_90 for training calls with at most 16 present experts (experts4bit-qlora TC1 amendment 22: Mixtral-8x7B's step 0.651x on an RTX 5090; Qwen3-like layers stay fused); `nf4_route.route_for`; engagement accounting for the training path

**0.39.0.** One default changes, by a rule registered and read in experts4bit-qlora: `GNF4_TRAIN_GEMM=auto` now takes
the dense route on cards other than sm_90 for training calls with **at most 16 present experts**. No serving kernel and
no forward output changes.

- **Why.** TC1 amendment 22 read the route on the full QLoRA training step, one RTX 5090 per family.
  - **Mixtral-8x7B** (2 of 8 experts per token): dense/fused **0.651** [0.648, 0.654], 5.52 → 3.59 s/step.
  - **Qwen3-30B-A3B** (up to 128 present experts per call): **2.947**, far slower. Its layers keep the fused kernels.
  - Held-out moved −0.0021 and −0.0001 nats.
- **What changes for you.** Few-large-expert models (Mixtral-like) train with each active expert dequantized and
  multiplied densely, on cards other than H100-class. Their values move within bf16 noise. `GNF4_TRAIN_GEMM=fused`
  restores the fused kernels. sm_90 keeps the grouped_mm route.
- **Also in this release:**
  - `nf4_route.route_for`: the training-route decision as a pure function of a device's facts, usable without a GPU.
  - Engagement accounting for the training path: `DGRAD_STATS` says which dgrad served each call and why the loop did;
    the pinned ring reports its overflow; the padded LoRA block's real bytes are recorded.
  - A docs fix: #71 is no longer described as open.
- `docs/system-manifest.json` is unchanged.

### `nf4_route.route_for`: the training-route decision without a device

- **Why.** `train_gemm_route(dev)` decided the route by asking a live CUDA device for its compute capability. A caller that
  wants the answer before it initialises CUDA, for a device it is not running on, or in a test without a GPU had to restate
  the rule, and the fused kernels' sm_80 floor was written only in the README.
- **What.**
  - `route_for(capability, *, has_grouped_mm, requested="auto", n_groups=None) -> (route, reason)`, a pure function of the
    device's facts.
    `route` is `"fused"`, `"grouped_mm"`, `"dense"`, or `None` when nothing in this module can train there, and `reason` says
    why. `n_groups` (a call's present groups, when known) reproduces #463's rule: `auto` takes `dense` off sm_90 for 1 to
    `DENSE_AUTO_MAX_GROUPS` of them.
  - `MIN_CAPABILITY = (8, 0)` and `GROUPED_MM_CAPABILITY = (9, 0)` as data.
  - `train_gemm_route` and the explicit-`grouped_mm` refusal now call it.
- **Unchanged.** Every live resolution is the same as before. Below sm_80, `auto` still answers `"fused"` and the launch is
  what fails. The refusal for an explicit `grouped_mm` keeps its wording; below sm_80 it now names the floor.
- **Tests.** `kernel/test_nf4_route_decision.py` runs on CPU. On a CUDA box it also checks agreement with the live
  resolution, per group count too: 17 passed on an RTX A2000.

### Docs: the host/NVMe streaming solution page no longer describes #71 as open (docs only)

- `docs/solutions/stream-moe-experts-from-host-or-nvme.md` named `PINNED_ROW_FACTOR` as a pinned row's cost and listed #71
  as open with cgroup v2 unmeasured. Both have been wrong since 0.38.0. The page now names `pinned_request_cost` /
  `capacity_for_bytes` and K29's register row.

### `GNF4_TRAIN_GEMM=auto` takes the dense route off sm_90 for calls with at most 16 present experts

- **Why.** experts4bit-qlora's TC1 amendment 22 read the dense route (#459) against the fused kernels on the full training step, one RTX
  5090 per family:
  - **Mixtral-8x7B** (2 of 8 experts per token): dense/fused **0.651** [0.648, 0.654], 5.52 → 3.59 s/step.
  - **Qwen3-30B-A3B** (up to 128 present experts per call): **2.947** [2.915, 2.979], far slower. The per-expert loop adds a launch per
    expert per projection to a step that is launch-bound.
  - Held-out moved by −0.0021 (Mixtral) and −0.0001 (Qwen3) nats.
- **What.** Its registered rule took the dense route under `auto` on non-sm_90 cards for calls with at most 16 present groups.
  - `train_gemm_route(dev, n_groups)` now takes the call's group count, and `FusedGroupedNf4` passes `len(sizes)`.
  - The threshold is `nf4_route.DENSE_AUTO_MAX_GROUPS`.
  - sm_90 keeps the grouped_mm route, CPU stays fused, and an explicit `GNF4_TRAIN_GEMM` value is used as given.
- **Values change** for training calls with 16 or fewer present groups on cards other than sm_90 (Mixtral-like layers), within bf16
  noise. `GNF4_TRAIN_GEMM=fused` restores the fused kernels.
- **Tested.** `test_route_env` covers the per-call choice; a new test drives `FusedGroupedNf4` with few groups (dense) and with many
  (fused). The two fused-dgrad tests pin `fused`. RTX A2000: 47 passed, 1 skipped.

### Engagement accounting for the training path: which dgrad served each call, ring overflow, the padded block's real bytes

- **`nf4_qlora.DGRAD_STATS`** counts which backward served each frozen-GEMM dgrad: the kernel, the grouped_mm route, the
  `dense` route (#459), or the per-expert decode loop. The loop is counted with its reason: an ineligible grad or shape (the `dgrad_eligible` reason),
  offload-staged storage, or `dgrad_kernel=False`. The loop used to be taken silently, so a run that asked for the kernel could
  not tell it had not had it. experts4bit-qlora's training census reads it (moe-generalize).
- **The pinned staging ring counts its overflow** (`_PinnedRing.overflow`): a call larger than `GNF4_PINNED_RING_SLOT_INTS`
  ints takes the pageable, synchronizing copy, and now says so.
- **`LORA_PAD_WASTE["last_bytes_alloc"]`** records the padded LoRA block at its real allocation itemsize. The block is
  allocated in the adapter dtype, so on fp32 adapters it takes twice the `last_bytes` that the `auto` rule compares against
  its 2 GiB limit. Recorded only: the rule is unchanged until a full-step reading says which accounting it should use.

## 0.38.0 — 2026-10-04 — pinned-tier sizing models PyTorch's power-of-two pinned allocator (`capacity_for_bytes`; #71): exact and never over budget, where the flat 1.9 wasted up to half a budget and could overshoot. Measured on cgroup v1 and v2 (lane K29)

**0.38.0.** One helper's answer changes; no kernel output changes. `capacity_for_bytes(..., pinned=True)`, the function the NVMe and host-RAM tiers tell you to size `hot_rows` with, now models what pinned memory actually costs.
- **What was wrong.** `PINNED_ROW_FACTOR = 1.9` was read as a per-byte premium on pinned memory. It is PyTorch's caching host allocator rounding every pinned request **up to a power of two**, and `alloc_landing` asks for `rows × stride + 4096`. The flat 1.9 wasted up to half a budget, and just above a power of two it could still over-allocate: 2100 MB costs 4096 MB, 1.96×.
- **What it does now.** `pinned_request_cost(n)` is the next power of two of `n + 4096`. `capacity_for_bytes` returns the largest `hot_rows` whose rounded request fits: exact, and never over the budget. At most budgets that is more rows than before.
  - `factor=PINNED_ROW_FACTOR` reproduces the old answer.
  - `pinned=False` is unchanged.
- **Measured in both cgroup regimes.** Every pinned request is charged 0.4–1.0 % over its power of two:
  - **v1** (RTX A2000, `kernel/receipts-71/`): 1.0043–1.005;
  - **v2** (rented RTX 5090, lane K29, `gnf4.kernel.k29-pinned-charge-cgroup-v2.5090.2026-10-04`): 1.0048–1.0103.

  K29 was registered as this release's gate, and it read CONFIRMED.
- **Also in this release (opt-in): `GNF4_TRAIN_GEMM=dense`** (#459). This training route works on any CUDA card. Each present expert is dequantized alone and multiplied with `torch.mm`, keeping the grouped kernels' contracts.
  - In an RTX A2000 replay of Mixtral-8x7B's expert shapes it ran forward 3.4× and dgrad 7.0–7.7× faster than the fused kernels.
  - The default (`auto`) is unchanged. A full-step A/B on an RTX 5090 decides whether `auto` takes it anywhere.
- `docs/system-manifest.json` is unchanged. Its `consumer_ci_pin` prose still trails the consumer's pin.

### K29 read: CONFIRMED. A pinned request costs its power of two on cgroup v2 as well (#71 closes)

- **Run.** `k29-5090-2`, $0.032. A rented RTX 5090 container under cgroup v2 (kernel 6.8.0-138, driver 590.48.01,
  torch 2.8.0+cu128).
- **Result.** 18 pinned rows (9 sizes from 340 MB to 4 GiB, ×2) read charged / pow2ceil(N) = **1.0048–1.0103**
  (median 1.0054). The pageable control slope is 1.0021. CONFIRMED by the registered rule ([0.97, 1.06]).
- **Predictions.** Q1, Q2 and Q4 held. Q3 (every r in [1.00, 1.01]) missed narrowly: one row read 1.0103.
- **Consequence, as registered.** This release ships the power-of-two pinned-tier model below, and #71 closes.
- **Records.** Register row `gnf4.kernel.k29-pinned-charge-cgroup-v2.5090.2026-10-04`; `kernel/RESULTS-k29-…`;
  `kernel/receipts-k29/5090/`.
- **Earlier attempts.** `k29-5090-1` ($0.06) lost its finished data to a single rsync (fixed in experts4bit-qlora#1047).

### `GNF4_TRAIN_GEMM=dense`: a per-expert dequant + `torch.mm` training route on any CUDA card (opt-in)

- **Why.** experts4bit-qlora's TC2 amendment 7 box D (Mixtral-8x7B, an RTX 5090) read e4b's reference loop, which dequantizes each expert
  and runs a dense GEMM, at 4.69 s/step against the fused kernels' 5.52. An RTX A2000 replay of Mixtral's expert shapes (1,024 rows
  per expert) shows the cause:

  | projection | forward, fused → dense | dgrad, fused → dense |
  |---|---|---|
  | gate_up | 127.5 → 37.9 ms | 283.7 → 36.7 ms |
  | down | 64.3 → 18.5 ms | 139.2 → 19.8 ms |

  At Qwen3-30B-A3B's shapes the dense loop is 1.7× forward and 4.6× dgrad at 256 rows per expert, and the fused forward is ahead at
  Qwen3's down projection with 64 rows.
- **What.** `nf4_route.dense_forward` / `dense_dgrad` keep `gemm_4bit_grouped` / `dgrad_4bit_grouped`'s contracts. Each present expert
  is dequantized alone (`dequant_groups`, bit-equal to `dequant_ref` in bf16) and multiplied with `torch.mm`, so one expert's bf16
  weight is the only transient (Mixtral's gate_up: 235 MB). It is selected only by `GNF4_TRAIN_GEMM=dense`; `auto` is unchanged.
  `ROUTE_STATS` gains `dense_fwd` / `dense_dgrad`.
- **Values.** Not bit-identical to the fused kernels (cuBLAS's accumulation order), within bf16 noise: the test bound is 5e-3 relative.
- **Tested.** `kernel/test_nf4_route.py` (dense against fused through `FusedGroupedNf4`, including an empty group, device-side sizes
  and ids, and the env value). RTX A2000: 46 passed, 1 skipped with the qlora-grad and eids-forms tests.
- **Next.** A full-step A/B on Mixtral and Qwen3-30B-A3B on an RTX 5090 decides whether `auto` takes it anywhere.

### K29 registered: what a pinned host byte costs a cgroup v2 container -- the release gate for the power-of-two pinned-tier model (prereg only; #71)

- **Why.** The pinned-tier model (`capacity_for_bytes`, this release's #71 entry below) was measured on cgroup v1. It
  hands back more rows than 1.9, so it ships only after one cgroup **v2** reading, the regime rented boxes run.
- **Method.** One rented RTX 5090. The probe from `kernel/receipts-71/` reads the container cgroup's own
  `memory.current` around fresh-process pinned and pageable allocations at ten sizes. The runner is
  experts4bit-qlora `bench/k29/` (rehearsed on v1; STOP-1 fired as designed).
- **Rule.** `r = charged / pow2ceil(N)`. VOID, then CONFIRMED (every r in [0.97, 1.06]: the model ships, #71 closes),
  then PREMIUM (median r > 1.06: the default is withdrawn before release), then MIXED. Lane ceiling $3.00.

### Pinned-tier sizing models the allocator's power-of-two rounding (`capacity_for_bytes`; #71)

- **Why.** `PINNED_ROW_FACTOR = 1.9` was read as a per-byte premium on pinned memory. It is not one. PyTorch's caching
  host allocator rounds each pinned request up to a power of two, and `alloc_landing` asks for `n + 4096`.
  - Read as the container cgroup's own charge on the stack the constant names (cgroup v1, driver 575.64.05, torch
    2.8.0+cu128, RTX A2000), pinned 340 / 700 / 1359 / 2100 / 3000 MB cost 514.5 / 1028.7 / 2057.1 / 4113.8 /
    4113.8 MB.
  - Power-of-two pinned requests and every pageable size cost ~1.00 per byte (1.0043 and 1.002), agreeing with #71's
    cap ladder (1.004).
  - So the flat 1.9 wasted up to half a budget and, just above a power of two (2100 MB: 1.96×), over-promised.
- **What.**
  - `pinned_request_cost(n)` (the next power of two of `n + PINNED_LANDING_PAD`).
  - `capacity_for_bytes(..., pinned=True)` returns the largest `hot_rows` whose rounded request fits the budget: exact
    and never over it.
  - `factor=` still gives the old flat formula (`factor=PINNED_ROW_FACTOR` reproduces it).
  - `pinned=False` is unchanged.
- **Tests.**
  - The receipts' charges.
  - Never-overshoot and maximality across budgets and strides (the old factor's overshoot is pinned too).
  - A charge test that runs a real pinned `alloc_landing` in a fresh process wherever CUDA and a cgroup charge are
    readable. On the A2000, all 30 tests in `test_nvme_residency.py` passed.
- **Released after K29 CONFIRMED** the model on cgroup v2 (the entry above). #71 is closed.

## 0.37.0 — 2026-10-04 — two defaults: programmatic dependent launch for decode launches of at most 8 rows (`GNF4_PDL`, value-identical; experts4bit-qlora's int4 serving decode 1.0404× at one request and 1.0000× at 16 on an RTX 5090) and the grouped_mm training route on compute capability 9.0 (`GNF4_TRAIN_GEMM=auto`; Unsloth/e4b 1.030 on an H100 NVL, a labelled row)

**0.37.0.** Two defaults change. One never changes values; the other changes them on compute capability 9.0 only.
- **`GNF4_PDL` is on, capped at 8 activation rows** (`GNF4_PDL=0` turns it off; `GNF4_PDL_MAX_ROWS=0` removes the cap).
  - On NVIDIA sm_90+ cards, the twelve decode-row kernels' launches of at most 8 rows become programmatic dependents of the kernel before them. Values never change.
  - experts4bit-qlora's lane P113 read it on SC1's int4 serving configuration (Qwen3-30B-A3B) on an RTX 5090, under decode-only timing. Tokens were identical, decode ran **1.0404×** as fast with one request and **1.0000×** with 16 (`e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04`). Uncapped, it cost 16 requests (0.9787×), which is why the cap is the default.
  - It is inert on CPU, under the interpreter, on ROCm and below sm_90.
- **`GNF4_TRAIN_GEMM=auto`** takes the grouped_mm training route on a compute capability 9.0 card whose torch has `_grouped_mm`, and the fused kernels everywhere else (`GNF4_TRAIN_GEMM=fused` restores the old numerics).
  - experts4bit-qlora's TC1c amendment 6 measured Unsloth/e4b at **1.030** [1.016, 1.045] on the full Qwen3-30B-A3B training step on an H100 NVL with the route on, and **1.325** with MoE activations also kept (`e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.route-v2`, `….moe-keep-route-v2`). Both are LABELLED rows. The H100 position of record stays amendment 1's 0.817 (Unsloth faster) until a default-settings box re-reads it on this release.
  - Values change on sm_90 only: at most 0.0024 relative Frobenius per GEMM in the kernel replay.
- **Also new in this release:** the grouped_mm route itself (`GNF4_TRAIN_GEMM=grouped_mm`, sm_90) with its dequant kernel at bandwidth; the `GNF4_PDL` and `GNF4_PDL_MAX_ROWS` switches the new default is built from; and K28's reading of PDL in the served B=1 decode chain (0.323 µs saved per gnf4 kernel, bit-identically).
- `docs/system-manifest.json` is unchanged. Its `consumer_ci_pin` prose still trails the consumer's pin.

### `GNF4_PDL` is on by default, capped at 8 activation rows (`GNF4_PDL=0` turns it off; `GNF4_PDL_MAX_ROWS=0` removes the cap)

- **Why.** experts4bit-qlora's lane P113 (experts4bit-qlora#1030) registered this as its CAP_DEFAULT consequence and read
  it on an RTX 5090. On SC1's int4 serving configuration (Qwen3-30B-A3B), with decode-only timing, the capped switch
  (`GNF4_PDL=1 GNF4_PDL_MAX_ROWS=8`) decoded identical tokens **1.0404×** as fast with one request (the step 4.381 →
  4.211 ms) and **1.0000×** with 16. Uncapped it read 1.0401× and 0.9787×; the 16-row step went 8.947 → 9.127 ms.
  Register row `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04` (experts4bit-qlora).
- **What changes.** `PDL_DEFAULT` is `True`, and `GNF4_PDL_MAX_ROWS` defaults to 8: unset or not a non-negative
  integer means 8, and `0` means no cap. Values do not change, because PDL never changes them. On NVIDIA sm_90+ cards,
  the twelve decode-row kernels' launches of at most 8 rows become programmatic dependents of the kernel before them.
- **Where it does nothing.** It is inert on CPU, under the interpreter, on ROCm and below sm_90, as before.
- **Scope.** The speed was read on SC1's int4 serving configuration on one RTX 5090. Elsewhere only exactness is
  established, by construction and by `kernel/test_pdl.py` on the card.
- `kernel/test_pdl.py` pins the new defaults.

### `GNF4_TRAIN_GEMM=auto`, the new default: the grouped_mm training route on compute capability 9.0, the fused kernels elsewhere

- **Why.** experts4bit-qlora's TC1c amendment 6 measured the route, with #452's dequant, on the full Qwen3-30B-A3B training step on
  an H100 NVL. Unsloth/e4b went from 0.817 with the fused kernels (amendment 1) to **1.030** [1.016, 1.045], so e4b is now faster per
  step. With MoE activations kept it went from 1.100 to **1.325** [1.296, 1.356]. The matched set stayed EQUIVALENT on both boxes.
- **The rule held.** Amendment 4 registered three conditions for this default, and all three hold: the numerics (P20), box R at or
  above 0.858, and e4b's held-out loss within 0.005 of amendment 1's (0.8481 against 0.8521 over two draws). That margin is thin:
  two of the four draw-against-draw pairings would miss it.
- **What.** `GNF4_TRAIN_GEMM` gains `auto` and defaults to it. `auto` takes the route on a CUDA device of compute capability 9.0 when
  torch has `_grouped_mm`, and the fused kernels everywhere else, including CPU; it is resolved once per device. `fused` and
  `grouped_mm` still force one, and `grouped_mm` still refuses off sm_90.
- **Values change on sm_90 only.** The route is not bit-identical to the fused kernels (at most 0.0024 relative Frobenius per GEMM in
  the kernel replay). Set `GNF4_TRAIN_GEMM=fused` for the old numerics. Other cards are unchanged.
- **Tested.** `kernel/test_nf4_route.py`'s env test now covers `auto`: the route on sm_90, the fused kernels elsewhere and on CPU.
  `test_fused_backward_matches_dequant_reference` pins `fused`, because it asserts the fused loop's exact backward. On an RTX A2000
  the three training-path files pass (36 passed, 1 skipped).

### `GNF4_PDL_MAX_ROWS=<n>`: keep programmatic dependent launch for small launches only (opt-in, with `GNF4_PDL=1`; default unchanged)

- **Why.** experts4bit-qlora's P112 closed VOID without a reading (#1026). Its first run's arms showed `GNF4_PDL=1`
  helping e4b's int4 serving step with one request (B=1 decode ×1.039 / ×1.037) and costing it with 16 (×0.985 /
  ×0.985), with identical tokens. At B=1 the switched launches carry 1 or 8 activation rows; at B=16 they carry 16 or
  more.
- **What it does.** With `GNF4_PDL=1` and `GNF4_PDL_MAX_ROWS=<n>`, a decode-row launch over at most `n` rows keeps PDL
  and a larger one launches without it. Every wrapper passes its activation rows: `R` for the quantise, GEMV, reduce,
  SwiGLU, rotary, router and norms, and the token count for the combine.
- **When it does nothing.** The cap is read with the switch, once (`pdl_refresh()` re-reads both). Unset, `0` or
  anything that is not a positive integer means no cap. The cap never turns PDL on.
- **Tested.** `kernel/test_pdl.py`, now 25 tests: the cap's parsing and gating on CPU, and an AST check that every
  launch passes its device and its rows. Values cannot change, because PDL never does.
- **No speed is claimed.** A new experts4bit-qlora lane reads PDL off, on everything, and capped, under decode-only
  timing.

### The grouped_mm route's dequant kernel runs at bandwidth (`GNF4_TRAIN_GEMM=grouped_mm`; values unchanged)

- **Why.** experts4bit-qlora's TC1c amendment 4 measured the route on the full Qwen3-30B-A3B training step on an H100 NVL, and it made
  the step **slower**: Unsloth/e4b fell from 0.817 to 0.664, and from 1.100 to 0.934 with MoE activations kept. The profiled arm put
  the route's `_dequant_groups_kernel` at 2,178 ms of device time per step (1.89 ms per call), against the fused kernels' 1,424 ms. The
  kernel gathered the NF4 LUT and the absmax per element.
- **What.** The kernel is rewritten:
  - byte tiles (BLOCK_N 16 × 256 bytes, 8 warps) are loaded coalesced;
  - both nibbles are decoded through a 16-entry register LUT (`tl.gather`) and interleaved with `tl.join`;
  - one absmax per quant block is applied by reshape-broadcast;
  - contiguous bf16 rows are stored.

  Same fp32 multiply and one bf16 rounding, so it stays bit-equal to `dequant_ref(...).to(bfloat16)`.
- **Measured on an RTX A2000** (Qwen3-30B-A3B stacks, 48 experts): gate_up 8.24 → 1.62–1.69 ms (224–233 GB/s), and down 2.64 →
  0.84–0.86 ms. That is 0.76–0.82× and 0.73–0.93× of bitsandbytes' `dequantize_4bit` on the same stacks. Outputs are bit-equal.
- **Next.** The route stays opt-in. Its full-step value on the H100 is re-measured with this kernel (experts4bit-qlora TC1c).
- **Tests.** `test_nf4_route.py` gains non-tile-multiple shapes: (17, 640) and (33, 64).

### K28 read (RTX 5090): LEVER — programmatic dependent launch saves 0.323 µs per gnf4 kernel in the served B=1 decode chain, bit-identically

Register: `gnf4.kernel.k28-pdl-decode-chain.5090.2026-10-04`; `kernel/RESULTS-k28-pdl-decode-chain.md`.
- **The reading** (`k28-5090-1`). 48 layers × 19 gnf4 kernels (912) replay in 2.363 ms with `GNF4_PDL=1` against
  2.658 ms without (×1.125). With the served step's attention and router stand-ins interleaved the
  saving is 0.217 µs per kernel. Every bitwise check held, and the probe's dependent was waiting
  19.4–19.7 µs before its 20 µs primary ended.
- **The predictions.** All seven held; Q4's 0.3–1.2 µs held at its low end.
- **What follows (registered).** `GNF4_PDL` stays off by default here. experts4bit-qlora registers a served lane (B=1 and
  B=16, tokens and tok/s), which also decides whether e4b's own decode kernels take the preamble.
- **Cost:** $0.0237.
### `GNF4_TRAIN_GEMM=grouped_mm`: dequantize the present experts and run the training GEMMs through `torch._grouped_mm` (opt-in, sm_90)

- **Why.** On an H100 NVL, experts4bit-qlora's fused training step is device-bound, at about 1.8× its comparator's device time per
  step. experts4bit-qlora's TC1c amendment 3 (`tc1c-h100-5`, H100 NVL, torch 2.8) replayed the 128 unique fused forward and dgrad
  calls of its Qwen3-30B-A3B training step, recorded through the real router:

  | | fused / dequant / grouped_mm (ms) | (dequant + grouped_mm) / fused | grouped_mm alone / fused |
  |---|---|---|---|
  | forward | 75.11 / 29.45 / 15.29 | **0.596** | 0.204 |
  | dgrad | 89.20 / 29.42 / 14.63 | **0.494** | 0.164 |

  Every call was within 0.0024 relative Frobenius error of the fused output. The registered rule (both ratios ≤ 0.80) is to take the
  route as an sm_90 opt-in.
- **What.** `kernel/nf4_route.py` adds two pieces:
  - one Triton kernel that decodes the present experts' NF4 stacks to a contiguous bf16 `[G, N, K]`, bit-equal to
    `dequant_ref(...).to(bfloat16)`;
  - `torch._grouped_mm` for the forward (`a_cat @ W_e^T`) and the dgrad (`grad_out @ W_e`) over the groups' offsets.

  `FusedGroupedNf4` picks the route at forward and remembers it, so a backward never mixes routes. `dgrad_kernel=False` and
  offload-staged storage keep their own paths. `ROUTE_STATS` counts calls.
- **Refusal.** Off compute capability 9.0 (torch 2.8's only `torch._grouped_mm` target) it refuses with the reason, and never falls
  back silently.
- **Numerics.** Not bit-identical to the fused kernels: bf16 weights, cuBLAS accumulation. It stays opt-in until a training A/B on the
  H100 decides (experts4bit-qlora TC1c amendment 4).
- **Tests** (`kernel/test_nf4_route.py`):
  - dequant bit-equality on four shapes;
  - route vs fused within 5e-3 on forward and dgrad through `FusedGroupedNf4`, with `torch._grouped_mm` stubbed by a per-group loop
    where the card has none;
  - refusal off sm_90;
  - environment parsing.

### K28 registered: does programmatic dependent launch shorten the served B=1 decode layer's gnf4 kernels in a CUDA graph, bit-identically? (bench and prereg)

`kernel/PREREG-k28-pdl-decode-chain.md`, `kernel/k28_bench.py`; tracking issue experts4bit-qlora#1015.
- **The bench.** It runs 48 layers of the served B=1 layer's 19 gnf4 kernels (912) at Qwen3-30B-A3B's shapes in one CUDA
  graph, `GNF4_PDL` off against on, on one RTX 5090. Alongside are an instrument repeat and a glued variant that adds
  the served step's attention and router stand-ins. The checks are bitwise outputs, launch accounting through a Triton
  launch hook, and an engagement probe on the global clock.
- **The rule.** VOID, FUNCTION_FAIL and NOISY come first. **LEVER** if the chain saves ≥ 0.25 µs per gnf4 kernel, which is
  0.23 ms of the served step's 4.08 ms span if it transferred whole; NO_LEVER otherwise. On LEVER, experts4bit-qlora
  registers a served lane; the default stays off here until it reads.
- **Rehearsed on the NAS A2000** (sm_86, not a reading). It read VOID by compute capability, as designed:
  - 912 gnf4 launches per capture, none with PDL;
  - every bitwise check held;
  - instrument 1.0008.

### `GNF4_PDL=1`: the decode-row kernels launch with programmatic dependent launch (opt-in; default unchanged)

- **Why.** experts4bit-qlora's SC1b census (#846) found e4b's B=1 decode step on an RTX 5090 does the same kernel work
  as llama.cpp's, 4.09 against 4.14 ms summed, but overlaps none of its 1,550 in-graph kernels. llama.cpp overlaps 95.5 %
  of consecutive pairs on one stream, which hides about 1.38 ms a step. Same-stream overlap is programmatic dependent
  launch (PDL). 913 of those 1,550 kernels are this module's decode-row kernels (SC1b's node census, `sc1d-5090-3`):
  the int4 trio 576, `_rope_norm_heads` 96, and 49 or 48 each of the norms, the router epilogue, SwiGLU and the combine.
- **What it does.** With `GNF4_PDL=1` on an sm_90+ NVIDIA card, twelve kernels launch as programmatic dependents of
  the kernel before them:
  - `_quant_x_rows`, `_quant_x_rows_gathered`, `_gemv_int4_b32`, `_reduce_partials`;
  - `_rmsnorm_rows`, `_rmsnorm_resid_rows`, `_scaled_resid_add_rows`, `_rope_norm_heads`, `_rope_heads`;
  - `_router_epilogue`, `_swiglu_rows`, `_combine_rows`.

  Each kernel's first statement waits for the previous kernel on the stream to complete, then releases the next one to
  launch. Nothing reads or writes memory before the wait, so PDL hides launch latency and cannot reorder anything.
  The two instructions (`griddepcontrol.wait`, `griddepcontrol.launch_dependents`) are inline PTX. Triton 3.4.0's own
  `tl.extra.cuda.gdc_wait` / `gdc_launch_dependents` wrappers still use the pre-3.4 `_builder` keyword and fail to
  compile.
- **What does not change.** Off (unset or `0`), every launch passes the keywords it passed before. The switch is inert
  even when set on CPU, under the interpreter, on ROCm, on cards below sm_90, and with a Triton that has no
  `launch_pdl`.
- **Read once.** The switch is read at the first launch that asks; `int4_b32.pdl_refresh()` re-reads it. Every launch
  asks, and a per-call environment read measured about 0.6 µs, which an eager decode step would pay about 900 times.
  The cached check costs about 0.06 µs.
- **Tested.** `kernel/test_pdl.py`:
  - on CPU: the parsing, the inert cases, and that every launch of the twelve kernels takes the switch and starts with
    the wait;
  - on an sm_90+ card: every wrapper, eager and captured in a CUDA graph, is `torch.equal` with the switch on and off,
    and a probe shows a dependent kernel starting inside its predecessor and leaving its wait only after it.

  No speed is claimed. Lane K28 (`kernel/PREREG-k28-pdl-decode-chain.md`) reads it on an RTX 5090, and the default
  stays off until then.

## 0.36.0 — 2026-10-04 — one MoE layer pass reuses its grouping instead of re-uploading and re-deriving it (`GNF4_HOST_REUSE`, on by default; value-identical): experts4bit-qlora's fused training step at 0.933 / 0.951 on an RTX 5090; the compact padded LoRA delta (opt-in)

**0.36.0.** One default changes, and no output changes: `GNF4_HOST_REUSE` is on (`GNF4_HOST_REUSE=0` restores the previous behaviour).
- **What it does.** One MoE layer pass stops re-uploading and re-deriving the same grouping:
  - repeat index uploads return the earlier device tensor;
  - the down LoRA delta reuses the gate_up delta's plan;
  - distinct-id adapter gathers take a scatter backward.
- **Measured.** experts4bit-qlora's TC1 amendment 20 measured its fused training step on an RTX 5090 with every other default in force. The step ran at **0.933** (shipped arm) and **0.951** (matched arm) of the flag off, with held-out loss unchanged (`e4b.train.host-reuse.qwen3.5090.2026-10-04`).
- **Also in this release (opt-in):** `NF4_QLORA_COMPACT_DELTA=1`. The padded LoRA delta saves its input instead of its padded block: 133 → 54 MB per layer at Qwen3-30B-A3B's shape with bf16 adapters, with values identical. That is what lets experts4bit-qlora keep MoE activations across a step (`E4B_MOE_KEEP_LAYERS`).
- `docs/system-manifest.json` is unchanged. Its `consumer_ci_pin` prose still trails the consumer's pin, and moves in a release that changes no other compatibility fact.

### `GNF4_HOST_REUSE` is on by default (`GNF4_HOST_REUSE=0` restores the previous behaviour)

- **Why.** experts4bit-qlora's TC1 amendment 20 registered a 5090 A/B of the flag (#444) with a decision rule: if both arms stepped at
  or below 0.99 of the flag off, it would become the default. The box (`tc1-5090-51`, AMD EPYC 7C13, RTX 5090, $0.36) ran e4b
  against itself, with every other current default in force and two draws a side in ABBA order:

  | arm | on / off | cross-draw range | s/step off (d1 / d2) | s/step on (d1 / d2) |
  |---|---|---|---|---|
  | shipped | **0.933** | 0.912 – 0.954 | 3.146 / 3.075 | 2.870 / 2.934 |
  | matched | **0.951** | 0.922 – 0.980 | 3.875 / 4.008 | 3.697 / 3.799 |

  All eight arms were VALID and every pair stable (within 3.4 %). Held-out loss was unchanged: 0.8145 → 0.8154 shipped and
  0.8500 → 0.8481 matched, within the step's existing run-to-run variation. Both predictions held (P33 band 0.92–0.99, P34
  0.93–0.99). Register: `e4b.train.host-reuse.qwen3.5090.2026-10-04`.
- **What changes.** Nothing a kernel reads. Uploads with identical integers on the same device and stream return the earlier device
  tensor. The down LoRA delta reuses the gate_up delta's plan. Distinct-id adapter gathers take the scatter backward. Captures never
  use the memo.
- **Untested.** Serving's eager decode also goes through `to_device_i32` and was not measured. experts4bit-qlora's serving and fused
  test files (76 tests) pass identically with the flag on and off.

### `NF4_QLORA_COMPACT_DELTA=1`: the padded LoRA delta saves its input, not its padded block (opt-in; default unchanged)

- **Why.** Under autograd, the lean padded delta saves the zero-padded input block `[G, widest, K]` for its first bmm, plus the
  per-expert gathered adapters. A hot expert makes the block `G x widest` rows wide however few rows are real. At Qwen3-30B-A3B's
  shape (380 tokens, top-8) one MoE layer saved 133 MB with bf16 adapters and 229 MB with fp32 adapters, 85 MB of that the gate_up
  block alone. That memory is what stops experts4bit-qlora from keeping MoE activations across a step instead of recomputing them
  under gradient checkpointing.
- **What.** `_CompactPaddedDelta` is one autograd node. Its forward runs the same ops. It saves the input (an alias), `eid`, `flat`
  and the first bmm's `[G, widest, r]` output. Its backward re-gathers the adapters, rebuilds the block with the same zero fill
  and `index_copy_`, and issues the calls autograd's own backward makes, on the same operand layouts.
- **Values.** Forward and every gradient are `torch.equal` to the autograd path. This covers distinct and repeated ids, bf16 and
  fp32 adapters, scaling 1 and 2, CPU and CUDA, with and without `GNF4_HOST_REUSE`.
- **Measured on an RTX A2000** (one `ExpertsLoRA` layer at Qwen3-30B-A3B's shape, skewed router, `GNF4_HOST_REUSE=1` in both
  arms, three interleaved pairs of 300 repetitions):
  - **Saved memory per layer:** 133.2 → 54.2 MB with bf16 adapters, and 229.1 → 54.9 MB with fp32.
  - **Forward host median:** 2.915 / 3.194 / 3.014 → 2.689 / 3.166 / 2.805 ms.
  - **Backward host median:** 2.574 / 2.772 / 2.610 → 2.526 / 2.878 / 2.696 ms.
  - **Backward device span:** 19.13 / 19.42 / 19.56 → 19.87 / 20.10 / 20.22 ms. **That is a cost**: rebuilding the block adds
    about 0.7 ms (+3.5 %) to a layer's backward. Under full-layer gradient checkpointing the flag is therefore a small device loss
    for a lower backward peak.
  - Its use is the step it enables. On a 4-layer Qwen3-30B-A3B slice (TC1 rows, mb2 x accum 4), checkpointing only attention and
    keeping the MoE activations took the step from 1.278 s to 0.975 / 0.981 s, with every trainable gradient `torch.equal` under
    deterministic mode. Peak memory rose 0.57 GB with this flag, against 1.39 GB without it.
- **Tests** (`kernel/test_compact_delta.py`): bit-identity across the grid above; at least 8x fewer bytes saved on a hot-expert
  case; off by default.

### `GNF4_HOST_REUSE=1`: one MoE layer pass stops re-uploading and re-deriving the same grouping (opt-in; default unchanged)

- **Why.** experts4bit-qlora's fused training step is host-bound on fast cards: the same RTX 5090 arm steps at very different
  speeds on different host CPUs. Within one MoE layer pass the gate_up and down projections share one grouping, yet each:
  - uploads `expert_ids` for its GEMM;
  - uploads `(rows, ids)` for its LoRA delta, then rebuilds the delta's flat padded-row index (about ten launches);
  - and the backward's dgrad uploads `expert_ids` twice more.

  That is 8 transfers per layer forward + backward, 4 of them repeats. The adapters are also gathered with advanced
  indexing, whose backward sorts its indices (a radix sort plus index arithmetic, about ten launches per adapter).
- **What.** All three changes sit behind the one flag:
  - `to_device_i32` keeps a small LRU (8 entries) keyed on the uploaded integers, the device and the current stream, and returns
    the earlier device tensor on a repeat. It is never consulted or filled under capture.
  - `lora_delta_grouped`'s lean padded path keeps a one-entry memo of its device plan (`eid`, `flat`), so the down delta reuses
    the gate_up delta's.
  - With host-known distinct expert ids, the adapters are gathered through `_GatherRows`, whose backward is a zero fill and
    one `index_copy_` (no sort, no atomics). `index_select`'s own backward, an atomic `index_add_`, measured about 0.4 ms of
    device time slower per layer backward on an A2000, so it is not used.
  - `HOST_REUSE_STATS` counts hits and misses.
- **Measured on an RTX A2000** (`bench/host-reuse/`): one `ExpertsLoRA` layer at Qwen3-30B-A3B's shape (380 tokens, r=16, bf16
  adapters, a skewed router, experts4bit-qlora's `enable_fast_train(dgrad=True)`), three interleaved off/on pairs of 300
  repetitions, on a host shared with other jobs:

  | phase | off, median host ms | on, median host ms |
  |---|---|---|
  | forward | 3.193 / 3.150 / 2.947 | 2.784 / 2.822 / 2.906 |
  | backward | 3.003 / 2.871 / 2.720 | 2.401 / 2.422 / 2.508 |

  The backward's device span also fell slightly: 18.93 / 19.16 / 19.38 ms became 18.77 / 19.01 / 19.17 ms.
- **Values.** The forward output and every adapter gradient are `torch.equal` with the flag off and on. In deterministic mode
  (`torch.use_deterministic_algorithms`) the input gradient is too. Outside it, the input gradient already varies from run to
  run with the flag off, through the atomic `index_add_` behind the caller's token gather.
- **Next.** experts4bit-qlora's TC1 measures the flag on an RTX 5090 before the default changes.
- **Tests** (`kernel/test_host_reuse.py`):
  - `torch.equal` on the forward and all gradients of a gate_up -> down twin, flag off vs on, including a repeated-id case
    that keeps advanced indexing;
  - the memo hits only identical integers with the same split, is per stream, is bounded, and is off by default;
  - a capture neither reads nor fills it.

  `test_pinned_ring.py` pins the flag off, since it watches the transfer itself.

## 0.35.0 — 2026-10-03 — three defaults for experts4bit-qlora's fused training step, all value-identical: the pinned index ring outside capture, the lean padded LoRA delta, and the prefill M-tile height from the group sizes

**0.35.0.** Three defaults change, and no output changes. Every result is bit-identical to 0.34.1's, and each previous setting is one environment variable away:
- `GNF4_PINNED_RING=0`;
- `NF4_QLORA_LEAN_DELTA=0`;
- `GNF4_PREFILL_TILE_RULE=max`.

All three came from experts4bit-qlora's training head-to-head (TC1, experts4bit-qlora#835 / #945). Each was measured on an RTX 5090 in e4b's fused training step on Qwen3-30B-A3B before its default moved:

| change | new / previous step time (shipped, matched) | register |
|---|---|---|
| the pinned index ring outside capture, with e4b's single-read grouping | 0.866, 0.847 | `e4b.train.host-syncs.qwen3.5090.2026-10-03` |
| the lean padded LoRA delta | 0.939, 0.911 | `e4b.train.lora-delta-lean.qwen3.5090.2026-10-03` |
| the prefill M-tile height from the group sizes | 0.924, 0.968 | `e4b.train.prefill-tile-rule.qwen3.5090.2026-10-03` |

- Held-out loss was unchanged in every A/B.
- One untested effect: serving's eager decode also goes through `to_device_i32`, and the ring was not measured there. `GNF4_PINNED_RING=0` restores the old build for a fully host-bound path.
- **Also in this release (tooling only):** a CI guard against merge-conflict markers.
- `docs/system-manifest.json` is unchanged. Its `consumer_ci_pin` prose still names v0.34.0. experts4bit-qlora's CI moves its pin to this release's commit in its next release, and the prose follows in the release after that, kernel first.

### The prefill M-tile height comes from the group sizes by default (`GNF4_PREFILL_TILE_RULE=max` restores the largest-group rule)

- **Why.** experts4bit-qlora's TC1 amendment 14 registered a 5090 A/B of the `cost` rule (#441) with a decision rule. If both arms
  stepped at or below 0.99 of `max`, `cost` would become the default. The box (`tc1-5090-43`, Ryzen 9 7950X) measured:

  | arm | `cost` / `max` | cross-draw range |
  |---|---|---|
  | shipped | **0.924** | 0.918 – 0.930 |
  | matched | **0.968** | 0.964 – 0.971 |

  Every pair was stable and held-out loss unchanged. `max` launched 128-row tiles on about 92 % of calls; `cost` launched mostly 64
  (64 %) and 32 (24 %).
- **What.** `_prefill_tile_rule()` now defaults to `cost`. Outputs are identical under either rule.
- **Inference prefill.** On the RTX A2000's 42-batch sweep (16–4096 tokens, three router skews), `cost` / `max` ran 0.604–1.019 per
  batch, median 0.956. It was slower than `max` only on a 64-token batch, by 1.9 %. Large groups still take the 128-row tile.

### `GNF4_PREFILL_TILE_RULE=cost`: the prefill M-tile height from the group sizes, not the largest group (opt-in; default unchanged)

- **Why.** `gemm_4bit_grouped`'s M-tile path uses one tile height for every group in a launch, keyed on `max(sizes)`. One hot
  expert therefore puts every group on 128-row tiles. The real router does this at training-sized batches: on a 4-layer slice of
  Qwen3-30B-A3B trained on experts4bit-qlora's TC1 alpaca rows, the median group was 35 rows, while the largest group in a call
  had a median of 155. On experts4bit-qlora's 5090 profile (`tc1-5090-41`), this kernel is 811 ms of a 2.05 s step's device time.
- **What.** `_prefill_block_m_cost(sizes)` picks the height in {16, 32, 64, 128} that minimises
  `sum(ceil(rows / BLOCK_M)) x (D + BLOCK_M)`, with ties going to the taller tile. Each tile pays a fixed decode of its expert's
  weight slice (D rows' worth) plus its rows' MMA. `D` defaults to 96 (`GNF4_PREFILL_TILE_D`), and one pass over the sizes costs
  about 13 µs. `PREFILL_BM_STATS` counts the heights actually launched. The default rule stays `max`.
- **Measured on an RTX A2000** (`bench/tile-rule/`):
  - 42 synthetic batches fit `tiles x (a + b x BLOCK_M)` to R² 0.998 (D ≈ 110–118). Slowdown against the best height, worst and
    geometric mean: `max` 1.656 and 1.179, `cost` 1.175 and 1.024.
  - The real-router replay takes 711.5 ms under `max`, 545.5 ms under `cost` and 538.5 ms at best.
  - The 4-layer training step, ABBA order, takes 1.619 / 1.589 s under `max` and 1.484 / 1.493 s under `cost`.
  - Outputs are bit-identical across rules.
- **Next.** experts4bit-qlora's TC1 measures the rule on an RTX 5090 before the default changes.
- **Tests** (`kernel/test_prefill_tile_rule.py`):
  - the rule is the argmin it claims, against a brute force over 7 size lists × 5 values of D;
  - a hot expert no longer sets every tile, and large groups keep the tall tile;
  - ties go to the taller tile, and D moves the pick;
  - the environment parsing works;
  - on CUDA both rules compute the same bytes, and the counter records the launched height.

### CI: `conflict-marker-guard` refuses merge-conflict markers in tracked text (tooling only)

- New workflow on push and pull_request: a positive control plants a two-sided conflict and asserts both marker lines are
  flagged, then `git grep` refuses any tracked line beginning `<<<<<<< ` or `>>>>>>> ` (the lone `=======` is a legitimate
  setext underline and is not matched; a conflict always carries the other two). `guard-allow` on the line exempts a
  deliberate quotation. Motivated by two CHANGELOG races on experts4bit-qlora in one hour (e4b#848's rebase staged an
  unresolved file; hotfix e4b#852). Lands here first, then mirrors to experts4bit-qlora. No package code changes.

### `GNF4_PINNED_RING=1`: index transfers outside a capture without a host sync (opt-in; default unchanged)

- **Why.** Outside a CUDA-graph capture, `to_device_i32` builds a pageable tensor, and that copy is `cudaMemcpyAsync`
  plus `cudaStreamSynchronize`. On the host-bound 3060 Ti step the wait was free, which is why the capture arena is
  capture-only. It is not free in experts4bit-qlora's fused training step on an RTX 5090: its profile spent
  1.27 s of a 7.28 s profiled step in `cudaStreamSynchronize`, and 8 of the 13 syncs per MoE layer pass are these
  copies (5 forward, 3 backward; experts4bit-qlora#945).
- **What.** `_PinnedRing`, a small per-device ring of pinned slots (`GNF4_PINNED_RING_SLOTS`, default 256, of
  `GNF4_PINNED_RING_SLOT_INTS`, default 16 Ki ints). Each call writes its ints into the next slot, copies with
  `non_blocking=True` and records the slot's event. A slot is rewritten only after its event completes; a wrap onto
  a copy still in flight waits for that copy and counts it in `waits`. Values are identical to the pageable path's.
  A call larger than a slot falls back to it. Captures still use the arena, whose slices are permanent.
- **Measured on an RTX A2000** (Qwen3-30B-A3B layer shape, 512 tokens, with experts4bit-qlora's single-read
  grouping): syncs per layer go from 6 forward / 3 backward to 1 / 0. Host time returns in 3.9 / 3.6 ms instead of
  waiting out the 17.5 / 21.5 ms GPU span. The training-step effect is to be measured on a 5090 before any default
  changes.
- **Tests** (`kernel/test_pinned_ring.py`, CUDA):
  - values equal the pageable path's;
  - no synchronizing call, against a control in which the pageable path is caught;
  - a 2-slot ring wrapped 12 times behind a 200 M-cycle `torch.cuda._sleep` keeps every value, and the host waits;
  - an oversize call falls back;
  - the ring is off by default.

### The pinned index ring is the default outside capture (`GNF4_PINNED_RING=0` turns it off)

- **Why.** experts4bit-qlora's TC1 amendment 10 registered a 5090 A/B with a decision rule. If e4b's single-read grouping plus
  this ring stepped the fused training step below 0.95 of the legacy path on both arms, the ring would become the default.
  The box (`tc1-5090-38`) measured **0.866** on the shipped arm (cross-draw 0.843 – 0.889) and **0.847** on the matched arm
  (0.840 – 0.854). Every pair was stable, held-out loss and peak VRAM were unchanged, and each new-path arm's ring staged
  53,680 transfers without waiting once (experts4bit-qlora#945, `e4b.train.host-syncs.qwen3.5090.2026-10-03`).
- **What.** `_pinned_ring_enabled()` is now on unless `GNF4_PINNED_RING=0`. Values are identical to the pageable build, and
  captures still use the arena.
- **Not measured here.** Serving's eager decode also goes through `to_device_i32`. A fully host-bound path pays the ring's few
  microseconds of bookkeeping per call where the sync was free; `GNF4_PINNED_RING=0` restores the old build for such a path.

### The padded LoRA delta does less work for the same bytes (`NF4_QLORA_LEAN_DELTA=0` restores the previous body)

- **Why.** experts4bit-qlora's TC1 amendment 12 profiled the fused training step on an RTX 5090 after #945's sync fix
  (`tc1-5090-41`, P19 HELD). The shipped arm spent 94 ms of its 2.05 s of device time per step in one kernel: a
  scalar-times-tensor multiply, 1,152 times per step. That is `lora_delta_grouped`'s `scaling *` over the padded
  `[G, widest, N]` block, in the forward, the checkpoint recompute and the backward. At the field recipe (r16, alpha 16) the
  scaling is 1.0. The matched arm read 0.740 device-busy, under amendment 12's 0.75 line, so its decision rule names
  launch volume in the LoRA path as the next work.
- **What.** Four changes to the padded path, each exact:
  - one flat row index (`g * widest + slot`) instead of the `(group, slot)` pair. Advanced indexing with two index
    tensors sorted its indices in the backward;
  - the gather back to row order goes through `_GatherRows`, whose backward is a plain `index_copy_` scatter. Autograd's
    own backward for `index_select` is an atomic `index_add_`, slow in bf16 on sm_86, and these rows are unique;
  - the gathered rows are the output. `a_cat` has exactly `total` rows on this path, so the zero fill and slice copy
    were redundant;
  - `scaling` multiplies the gathered rows, not the padded block, and is skipped at exactly 1 (`_scaled`; a tensor
    `scaling` is always applied). The `loop` and `grouped_mm` paths skip it at 1 too.
- **Measured on an RTX A2000** (a 2-layer Qwen3-MoE at Qwen3-30B-A3B's layer shape, r16 alpha 16, e4b's fused training step,
  ABBA order). Launches per step 1,424 → 1,344, device time 228.5 → 216.3 ms, wall 241.0 → 227.1 ms. The 5090 effect is
  to be measured by experts4bit-qlora's TC1 before it is quoted.
- **Tests** (`kernel/test_lora_delta_lean.py`, CPU and CUDA):
  - forward and all three gradients are `torch.equal` to the previous body, kept in the test verbatim. The grid covers
    4 group layouts (including empty groups), scalings 1, 2, 0.5 and 1.7, and bf16/bf16, bf16/fp32 and fp32/fp32
    activation and adapter dtypes;
  - device-tensor expert ids give the same bytes;
  - the skip fires only at exactly 1;
  - `_GatherRows`' backward equals `index_select`'s;
  - CUDA launches per forward and backward drop, and `NF4_QLORA_LEAN_DELTA=0` reaches the old body.

## 0.34.1 — 2026-10-01 — K25 decodes the NF4 codebook with an exact select tree by default (lane K26): bit-identical outputs at 0.373 / 0.383 of the paired lookup's time; lane K27 reads the tree at the served kernel's TF32 precision

**0.34.1.** One behavior change: `nf4_smallm.gemm_nf4_grouped_smallm` (K25) now defaults to `lut="tree"`. Its outputs are bit-identical to the previous default (`pair`) and to `load`, which both stay available. The rest is evidence: lanes K26 and K27 (benches, pre-registrations, results, receipts and register rows). `docs/system-manifest.json`'s `consumer_ci_pin` prose now names v0.34.0, the release whose commit experts4bit-qlora's CI installs; it had still named v0.33.0. The compatibility records are otherwise unchanged.

- **K27 read (RTX 5090): TF32_PATH. K25 with the select tree at the served precision (fp32 weights, TF32 MMA) runs at 0.448 / 0.502 of the served NF4 GEMM with the same error.** (`gnf4.kernel.k27-nf4-tree-precision.5090.2026-10-01`)
  - **ms per B=16 step (Granite / OLMoE):**
    - TF32 tree 2.588 / 4.847 (best plan 32 / 64 / 4 / 3);
    - bf16 tree 1.832 / 3.139 (32 / 128 / 4 / 2; bit-equal to the paired lookup);
    - the served kernel 5.780 / 9.660;
    - P92's K25 6.070 / 9.989.
  - **Error.** rms against fp64: the TF32 tree equals the served kernel to four digits; the bf16 arms read 1.35×.
  - **Next:** K25-tree in TF32 end to end on both families (experts4bit-qlora).
  - `kernel/RESULTS-k27-nf4-tree-precision.md`, `kernel/receipts-k27/5090/` (`k27-5090-3`; lane $0.044 with a bandwidth refusal and a guard 429 before it).

- **K27 registered: does K25 with the select tree keep its speed at the served kernel's weight precision? TF32 against bf16 at the NF4 families' B=16 shapes on one RTX 5090 (bench and prereg).** (`kernel/PREREG-k27-nf4-tree-precision.md`, `kernel/k27_bench.py`)
  - **Why.** P92 read K25 (bf16 weight operand) QUALITY_FAIL on OLMoE. K25's `dot_bf16=False` keeps the fp32 weight through TF32 MMA, the served precision class.
  - **The rule.**
    - TF32_PATH if `tree32 / served` ≤ 0.60 with `err(tree32) / err(served)` ≤ 1.10 in both families (an fp64 error proxy);
    - BF16_ONLY if only the bf16 tree clears 0.60;
    - NONE otherwise.
  - **A2000 correctness pass** (`--quick`, not a reading): every plan ran; `tree16` bit-equal to `pair16`; `err(tree32) / err(served)` 1.000; `tree32 / served` 0.46 / 0.64.

- **K25 decodes its codebook with an exact select tree by default (`lut="tree"`), as lane K26 pointed: bit-identical outputs, at 0.373 / 0.383 of the paired lookup's time on an RTX 5090 (Granite / OLMoE B=16 shapes).** (`gnf4.kernel.k26-nf4-decode-ablation.5090.2026-10-01`)
  - **What.** `nf4_smallm` loads the 16 fp32 codebook values once per program and selects each weight with a 4-level tree on the nibble's bits, in place of a load per byte (`pair`) or per nibble (`load`). Both lookups stay available.
  - **Contract.** `kernel/test_nf4_grouped_smallm_interp.py`:
    - the tree is bit-identical to `pair` and `load` across cases, plans (three tree plans added) and the masked K tail (a new test);
    - compiled, the weight read back through the MMA is the bf16 dequant.
    RTX A2000: compiled 30/30, interpreter 26 passed; a mutation that swaps one tree leaf fails 9 tests.
  - **The K26 bench** names its product arm `lut="pair"` explicitly, so a rerun keeps measuring what it measured.

- **K26 read (RTX 5090): DECODE. The NF4 codebook lookup is about 80 % of K25's time; an exact select-tree decode takes K25 to 0.373 / 0.383 (Granite / OLMoE) at bit-identical outputs.** (`gnf4.kernel.k26-nf4-decode-ablation.5090.2026-10-01`)
  - **Arms, ms per B=16 step:**
    - K25: 5.869 / 9.898;
    - `nibble − 8` in place of the lookup: 1.310 / 2.050;
    - the tree: 2.189 / 3.794;
    - the served NF4 GEMM: 5.650 / 9.540.
  - **Controls.** Both bench copies are bit-equal to the kernels they copy, within 1.1 % of their times. The served path with the tree is not bit-identical and not faster.
  - **Registered pointer:** K25 takes the select tree.
  - `kernel/RESULTS-k26-nf4-decode-ablation.md`, `kernel/receipts-k26/5090/` (`k26-5090-1`, $0.03).

## 0.34.0 — 2026-10-01 — grouped small-M tensor-core GEMMs for decode batches: K19 (int4-b32) with the K20 plan as its default, K21 (native MXFP4, with a masked K tail), K23's lean grouping glue, and K25 (NF4); the MXFP4 QLoRA fused combine is deterministic (folded in from the never-published 0.33.8)

**0.34.0.** Four kernels for decode-batch experts, all K19's structure: 16-row expert tiles, the gather folded into the load, bf16 tensor-core MMA. In experts4bit-qlora they serve its licensed batched-decode routes (K19 for the int4 store, lane P88; its lean glue, lane P89; K21 for the MXFP4 store, lane P90). K25 is opt-in there (lane P92: Granite measured, OLMoE failed its quality gate). Shipped modules changed since 0.33.7: `int4_smallm`, `int4_b32`, `mxfp4_grouped`, `nf4_smallm` (new), `mxfp4_qlora` (0.33.8's fix), `_triton_shim` (CPU-path entries) and `gguf_reader` (a docstring). Every other shipped module is identical to 0.33.7. The bench files K20–K26 are campaign instruments, deliberately not packaged.

- **K26 registered: is the NF4 codebook decode what holds K25 (and the served NF4 GEMM) back? A decode ablation on the NF4 families' B=16 shapes on one RTX 5090 (bench and prereg).** (`kernel/PREREG-k26-nf4-decode-ablation.md`, `kernel/k26_bench.py`)
  - **Why.** experts4bit-qlora lane P92 read K25's GEMM within 4 % of the served NF4 GEMM in-model. K25 shares K21's skeleton but not its decode: K21 uses integer shifts, the NF4 kernels a codebook lookup per nibble.
  - **The arms.** K25 as merged against bench-local copies of it with the decode switched:
    - the same codebook (the control);
    - `nibble − 8` (prices the lookup);
    - no scale, and bytes only;
    - an exact 4-level select tree over the 16 fp32 codebook values.
    The served kernel and a copy of it with the same select tree run beside them.
  - **The rule.** DECODE if `affine / pair` ≤ 0.60 in both families, NOT_DECODE if ≥ 0.85. The registered pointer: an exact decode qualifies only when bit-equal to the kernel it replaces and at most 0.80 (K25) / 0.90 (served) of its time.
  - **A2000 correctness pass** (`--quick`, not a reading):
    - every control bit-equal;
    - `tree` bit-equal to K25;
    - `stree` not bit-equal to the served kernel: the same layout/lowering effect `tl.gather` showed in K25.

- **K25: K19's grouped small-M tensor-core GEMM on the NF4 store (`nf4_smallm.gemm_nf4_grouped_smallm`), opt-in; no consumer, no speed claim yet.**
  - **Why.** experts4bit-qlora's lane P91 (#564) read the NF4 families' B=16 decode steps on an RTX 5090: the served grouped GEMM (`_gemm_nf4_grouped`) is 61.7 % (Granite, `r12epi`) and 71.9 % (OLMoE, `nf4`) of kernel time. That kernel gathers its rows in a separate launch, steps K 64 at a time, and multiplies TF32 on fp32-dequantised weights.
  - **What it is.** K19's kernel with the NF4 dequant, as K21 is on the MXFP4 store:
    - the grid, tile table, in-kernel gather, sorted output, K23's `scatter` / `gather_div` and K21's masked K tail are K19's;
    - each nibble decodes through the fp32 codebook (element 2j is the high nibble), is scaled by its per-64 absmax in fp32, rounded to bf16, and multiplied on the tensor cores.
  - **Arithmetic.** The weight operand is exactly `dequant_ref(...).to(bfloat16)`, the dequant-then-GEMM path's operand. That is not the served kernel's TF32, which rounds the weight less. So a consumer gates it on quality: on the A2000, at Granite and OLMoE expert shapes on synthetic weights, its rms error against an fp64 product of the fp32 dequant is 1.37× the served kernel's (1.369–1.374 over four shapes).
  - **Two codebook decodes, bit-identical:**
    - `lut="pair"` (default) loads one int64 per packed byte holding both of its fp32 codebook values;
    - `lut="load"` loads one fp32 per nibble.
    A `tl.gather` register decode was tried and left out: its weights are exact, but its outputs moved a bit at some shapes on the A2000.
  - **Contract:** `kernel/test_nf4_grouped_smallm_interp.py`, registered as an interpreter file and in CI's interpreter job:
    - within one bf16 ulp of the dequant reference, including the masked tail and K = 2880;
    - the decodes, the gather, strided stack views, `scatter` and `gather_div` are bit-identical to their plain forms;
    - deterministic; refusals;
    - compiled only: plans move no output bit, and the weight operand read back through the MMA is the bf16 dequant.
  - **RTX A2000:** 28/28 compiled; interpreter 24 passed (4 compiled-only skips). Six mutations each fail the suite: pair halves swapped, nibble order, absmax column, absmax rounded to bf16, no scatter, no `gather_div`.
  - **Shared memory.** On the A2000's 99 KB per block, `"load"` overflows at BLOCK_N × KC of 32 × 256 and 64 × 128, and `"pair"` only at 128 × 256.

- **K24 read (gpt-oss-20b, RTX 5090): VOID by its instrument again, and per-layer stores did not close the gap; descriptively K21 with its masked-tail plans reads 0.50× the served NF4 route, at 50 % of the byte floor.** (`kernel/RESULTS-k24-gptoss-per-layer.md`)
  - **Census:** step 22.50 ms; `_gemm_nf4_grouped` 17.90 ms/step (79 %).
  - **Bench, ms/step:** served 15.18; K21 best (32 / KC 128 / 4 warps / 3 stages) 7.62; default (32 / 256) 9.22;
    MXFP4 GEMV 23.84; floor 3.81.
  - **All 54 plans are bit-identical to the default**, the masked KC 128/256 plans included.
  - **Why VOID:** the bench's served NF4 kernel read 14.85 ms/step against the census's 17.90 (−17 %, band 15 %), the
    same miss as K22. K22's inferred cause (cross-layer L2 reuse) is refuted. The next candidate, inferred: the bench
    replays wikitext teacher-forced routing, while the census decodes its own prompts.
  - **Next:** an experts4bit-qlora opt-in route for the MXFP4 store's batched rows to K21, then an end-to-end lane with
    the store's KL instrument as the gate.

- **K23: the grouping glue around K19 folds into the two kernels it brackets; opt-in, no speed claim yet.**
  - **Why.** experts4bit-qlora P88 censused Qwen3-30B-A3B's B=16 step on an RTX 5090, 8 graph replays per arm. Diffing
    its K19 arm against its GEMV arm leaves 0.685 ms/step of launches that K19's route adds:
    - the tile builder, 0.430;
    - three fills, 0.096 (+144 elementwise calls per step);
    - an index kernel, 0.094 (by its count, the unsort);
    - a scatter/gather kernel, 0.065.

    Both arms also pay an `index_select` of 0.46 ms/step. By its call count and per-call time it is inferred to be
    the `[T * top_k, H]` expansion of the token rows, which `gather_div` makes unnecessary.
  - **What.**
    - `int4_b32.build_group_tiles_fused(..., lean=True)`: the builder reads the ids at their own dtype and zeroes the
      padding slots itself: one launch where the default path makes up to five calls (a cast, three
      zero-fills, the builder). Live tiles cover exactly the slots below the tile
      total, so no address is written twice.
    - `sorted_ids=True` appends `ids[order]` as a sixth output. `warps=` sets the launch's warp count.
    - `int4_smallm.gemm_int4_b32_grouped_smallm(..., scatter=order)`: K19 stores sorted row i at row `order[i]`. That is
      the caller's unsort in the store; the values are the same bits.
    - `gather_div=k` (with `order`): `x` is the step's `[T, H]` token rows, and sorted row i reads token row
      `order[i] // k`. That is what reading the expansion `x.repeat_interleave(k, 0)` reads, without the copy.
  - **Contract.**
    - `test_lean_tile_table_is_identical`: the default call's integers exactly, into poisoned buffers, at int64 and
      int32 ids.
    - `test_scatter_is_the_unsort`: bit-identical to `index_copy_`, with and without the gather.
    - `test_gather_div_reads_token_rows_as_their_expansion`: bit-identical to the expanded call, with and without
      scatter; bad shapes are refused.
    - All run under the interpreter and compiled. On the RTX A2000 (correctness only): the builder file 17/17, K19's
      22/22. Mutation arms each fail the new tests:
      - dropping the self-zeroing fails 10/10 lean cases;
      - dropping the scatter fails 3 of 4 (the fourth routing is already sorted);
      - dropping the divide fails 2 of 3 (the third is k = 1).
    - With every option off, both kernels do the same work as before: the new branches are compile-time constants,
      and the builder only gains one unused pointer argument.

- **K21 takes a masked K tail: KC need not divide K.** `gemm_mxfp4_grouped_smallm`'s KC is any of 32, 64, 128 or 256 (refused otherwise). When KC does not divide K, the last chunk is masked: activations and weights past K load as zero, so the padded columns add exact zeros. A constexpr `EVEN_K` keeps the common case mask-free.
  - **Why.** gpt-oss's K = 2880 previously capped K21 at KC 64, and K22 read K21 at only 37 % of the byte floor there. On Qwen3, KC 256 was worth 75 % → 89 % of the floor (K20).
  - **Tests.** Masked-tail reference cases (K = 2880 at KC 256 and 128, K = 96 at KC 64) and an unsupported-KC refusal. The compiled plan-identity test now runs K = 2880's KC 256/128 plans masked, and they stay bit-identical to KC 64. RTX A2000: compiled 14 passed; interpreter 23 with K19's file.

- **K22 read (gpt-oss-20b, RTX 5090): VOID by its instrument; descriptively, 79 % of the B=16 step is the NF4 expert GEMM, and K21 reads 0.68× that route at 37 % of the byte floor.** (`kernel/RESULTS-k22-gptoss-mxfp4-b16.md`)
  - **Census:** step 22.52 ms; `_gemm_nf4_grouped` 17.86 ms/step.
  - **Recorded gpt-oss routing:** `[128, 24, 16, 4]`, 18.2 distinct experts per layer per step. Kept in the receipts.
  - **Bench:** served 15.17 ms/step, K21 best 10.30 (KC 64), MXFP4 GEMV 21.65, floor 3.83.
  - **Why VOID:** the bench's served NF4 kernel read 17 % under the census (band 15 %). The likely cause, inferred: the bench shared one weight set across all 24 layers, letting L2 serve overlapping experts.
  - **Next:** a K21 masked-K-tail variant (gpt-oss's K = 2880 caps KC at 64), then a re-read with per-layer stores.

- **K21: K19's grouped small-M tensor-core GEMM on the native MXFP4 store (`mxfp4_grouped.gemm_mxfp4_grouped_smallm`), opt-in; no consumer, no speed claim yet.**
  - **Why.** experts4bit-qlora serves gpt-oss's licensed MXFP4 store with `gemv_mxfp4_b32` up to 16 rows. At B=16 a call routes 64 rows (16 × top-4), so the consumer falls back to NF4. K21 is the batched kernel the store lacks: the first kernel of the throughput push to other model families.
  - **What it is.** K19's kernel with only the dequant swapped. An e2m1 nibble decodes to twice its value as an exact integer (`gemv_mxfp4_b32`'s construction), and the per-32 e8m0 byte becomes `2^(e - 128)`, absorbing the half. So every MXFP4 weight is exact in bf16, and the MMA operand equals `dequant_mxfp4`. The grid, tile table, in-kernel gather and sorted output are K19's. The default plan is 32/256; until the masked tail above, `plan_smallm` lowered it to KC 64 for gpt-oss's K = 2880.
  - **Contract:** `kernel/test_mxfp4_grouped_smallm_interp.py`, registered as an interpreter file and in CI's interpreter job:
    - within one bf16 ulp of the dequant reference, including K = 2880;
    - the gather is bit-identical to presorting;
    - deterministic;
    - refusals;
    - compiled only: plans are bit-identical.
    RTX A2000: 10/10 compiled.

- **K20 read (RTX 5090): PROMISING. K19's default plan becomes BLOCK_N 32 / KC 256 / 4 warps / 2 stages; outputs are bit-identical across plans.** (`gnf4.kernel.k20-k19-plan-sweep.5090.2026-10-01`)
  - **The lane** (`kernel/PREREG-k20-k19-plan-sweep-5090.md`, #420; `kernel/RESULTS-k20-k19-plan-sweep-5090.md`): replays experts4bit-qlora P60's recorded B=16 routing on the 5090, one CUDA graph per decode step, sweeping 72 plans. The best plan was chosen on steps 0–7 and read on steps 8–15.
  - **Steps 8–15, ms per decode step:**
    - served route (int8 quantise + GEMV + reduce) 7.062;
    - K19 at the shipped plan (64/128) 6.206;
    - K19 at 32/256 **5.200** (0.736×; 89 % of the 4.640 ms measured byte floor);
    - the tile build 0.435 of that.
  - The instrument reproduced experts4bit-qlora P87's in-model census within the registered 15 %. Every top-ten plan has KC 256. Lane cost $0.0615.
  - **The plan is free numerically.** Compiled, all 70 plans that ran match the shipped plan bit for bit, because the MMA accumulates the same products in the same order. `test_plans_are_bit_identical_compiled` holds it (passes on sm_86 too). Under the interpreter `test_bitwise_equals_k16_per_expert` now names its plan, since numpy's fp32 dot can move a bit across KC.
  - **What follows:** an end-to-end lane in experts4bit-qlora (P88) before any consumer default moves.

- **K19: a grouped small-M int4-b32 GEMM for decode-batch experts (`int4_smallm.gemm_int4_b32_grouped_smallm`), opt-in; no default changes.**
  - **What it is.** K16's arithmetic (bf16 tensor-core MMA, the int4 tile dequantised and scaled in registers, one `tl.dot` per 128-wide K chunk) over K14's expert-major device tiles (`build_group_tiles_fused`, 16-row tiles).
  - One launch per projection covers every (tile × N block). There is no split-K, so no partials, counters or separate reduce, and activations stay bf16 (no int8 quantise).
  - The first projection's expert-major gather is folded into the load through `order`. Outputs come back in sorted order, a drop-in for K14's `gemm_int4_b32_grouped_captured`. The call is capture-legal.
  - **Contract** (`kernel/test_int4_grouped_smallm_interp.py`, registered as an interpreter file and in CI's interpreter job):
    - each sorted row is within one bf16 ulp of `x[src] @ dequant(packed[e]).T`;
    - **bit-identical to K16 (sk=1) on each expert's rows**, so grouping changes no arithmetic;
    - the in-kernel gather is bit-identical to gathering first;
    - an expert with more than 16 rows spans tiles;
    - deterministic, and a layout mismatch is refused before launch.
    - 20/20 pass under the interpreter and 20/20 compiled on the NAS RTX A2000.
  - **Exploratory A2000 probe, not a claim** (`kernel/receipts-k19-a2000-probe/`). On experts4bit-qlora P60's recorded B=16 routing (8 steps, 48 layers, gate_up + down): K19 41.9 ms/step against the served `_gemv_int4_b32` path's 79.3, tile build included. The 5090 measurement, and any default change in experts4bit-qlora behind a quality gate, are separate lanes.

- **Owner quotes and name credits are removed from the documents (docs, and one docstring).**
  - Verbatim chat quotes and name credits are removed from the kernel pre-registrations (B374, B393, K17, K18), two RESULTS pages, an upstream draft, the CI workflow comment and the `kernel/gguf_reader.py` module docstring. That docstring is text only, so no behaviour changes. Directives are paraphrased or reduced to their date; no criterion, band, measurement or date moved.
  - Two OpenTimestamps-anchored documents were edited: `kernel/RESULTS-gate2-confirmatory.md` and `kernel/RESULTS-v2-confirmatory.md`. Each now ends with a note that its `.ots` anchors the version before the edit, which git history keeps.

### Folded in from 0.33.8 (prepared 2026-09-29; never tagged or published, so it ships here)

Training through `ExpertsMxfp4LoRA` in fused mode (`mode="fused"`) no longer returns different outputs, input gradients or adapter gradients for identical inputs on CUDA. Its combine summed each token's k expert rows with one bf16 `index_add_`, which is float atomics, and it now uses the ordered add that 0.33.6 gave the prefill engine. This affects MXFP4 QLoRA training in fused mode on NVIDIA GPUs. The loop mode, inference, the prefill and decode engines, and the NF4 and int4 kernels are untouched. Upgrade if you compare training runs or need them to repeat bit for bit. Fused-mode values move at the bf16-rounding level (one fixed summation order in place of a varying one). No new number: the evidence is `test_fused_repeated_calls_are_bitwise_identical`, which passes on the RTX A2000 and fails with the bare `index_add_` restored. No floor change for experts4bit-qlora.

- **The MXFP4 QLoRA fused training path adds its combine in a fixed order, so repeated calls give the same bits (#409).**
  - **What was wrong.** `ExpertsMxfp4LoRA._forward_fused` summed each token's k expert rows with one bf16 `index_add_` over `tok_of_pair`, which repeats every token k times. That is CUDA float atomics, so identical inputs could give different outputs, dL/dx and adapter gradients from call to call. Same class as #408.
  - **What changed.** It now uses `mxfp4_pipelined._index_add_ordered_` (#410): unique rows per pass, the sequential sum bit for bit, and a gather in the backward.
  - **Test.** `test_fused_repeated_calls_are_bitwise_identical`, CUDA, k = 4 of 8 experts, 256 tokens. On the NAS RTX A2000 it passes, and fails with the bare `index_add_` restored, differing in the output, dL/dx and both B gradients.
  - The gather's backward (`hidden_states[tok_of_pair]`) proved deterministic already: with the combine fixed, every gradient repeats.
  - The loop path is unchanged, and so is `test_fused_matches_loop`.
- **The register and STATUS stop listing #409 as open.** `gnf4.open.issues` now reads #60 and #71, with its previous text in the notes. The STATUS #408 bullet says #416 fixed #409, and "What is open" drops it.

## 0.33.7 — 2026-09-29 — `kernel/fp8_kv.py`: the fused KV append writes `quantize_kv_fp8`'s bytes exactly, its quotient now IEEE-rounded (experts4bit-qlora#771, #413); the append's byte gates skip by name below sm_89 (#414); every other shipped module identical to 0.33.6

**0.33.7.** `fp8_kv_append_t1` / `fp8_kv_append_bt1`, the fused fp8 KV appends, now store exactly the bytes `quantize_kv_fp8` stores. Before, their quotient `x / scale` went through Triton's default fp32 divide, which is not IEEE-rounded. On an RTX 5090 with the hardware e4m3 cast, **21 of 5.4×10⁸ stored bytes differed under 0.33.6's append, 0 under this one** (experts4bit-qlora lane B771).
- **Who is affected.** Anyone mixing the fused append with the eager quantize in one cache, or comparing a fused-append run with an eager one: in e4b, the bucketed CUDA-graph decode against the eager runner. Rare byte flips there moved greedy tokens after tens of steps.
- **Values.** They move only where the old divide rounded differently, about once in 25 million values.
- **Nothing else changed.** No kernel besides the two appends changed, and no speed was measured.

- **The fused fp8 KV append now writes `quantize_kv_fp8`'s bytes exactly: its quotient is IEEE-rounded (e4b#771).**
  - `fp8_kv_append_t1` / `fp8_kv_append_bt1` computed each group's `x / scale` with Triton's default fp32 `/`, which is not IEEE-rounded. torch's tensor divide, in the reference, is.
  - Probed in fp32 on the NAS RTX A2000 (arch-independent): the scale matched on every row, but ~30% of quotients differed, and ~6e-8 of the stored e4m3 bytes (18 of 302M, using torch's cast in place of the hardware one).
  - That is rare, but a 48-layer Qwen3 appends ~790K values per decode step. e4b's eager step (`append_many`, the reference) and its CUDA-graph bucket step (`append_graph_bt1`, this kernel) decoded different tokens within 67–129 steps at identical rows and grouping (e4b lane P81).
  - The per-group math now lives in one jit helper, `_e4m3_group`, that both sides call. The scale keeps Triton's default `/`: it matches torch's scalar divide, and an IEEE `div_rn` would not. The quotient uses `tl.math.div_rn`: 0 fp32 mismatches in 302M values.
  - `quantize_kv_fp8`'s arithmetic is factored into `_quantize_kv_fp32`, byte-identical to before on 24 cases.
- **A gate that can see it, on any CUDA card.** `test_group_math_is_the_reference_fp32_math` compares the fused math's fp32 scale and quotient, before the e4m3 cast, with `_quantize_kv_fp32`, bitwise, over ~25K groups at three group sizes. It includes all-zero groups and single-spike groups.
  - No fp8 cast is involved, so it runs below sm_89. The byte-level gates need sm_89+ and are too small to see 6e-8.
  - On the A2000 it passes, and with the quotient reverted to `/` it fails (~27% of quotients).
  - Measured end to end in experts4bit-qlora lane B771 (RTX 5090, hardware cast): 21 of 5.4×10⁸ bytes differed before, 0 after. e4b's eager-vs-graph token divergence had a second, independent cause in e4b itself (a bucket-of-one append to a scratch slot, experts4bit-qlora#777); this fix removes the append's share.
- **The fused append's byte gates skip by name below sm_89 instead of failing to compile.** `test_bitwise_against_eager_path`, `test_untouched_bytes_stay_untouched` and `test_bt1_bitwise_against_t1_loop` gated only on "CUDA available", but the fused append's e4m3 cast (`fp8e4nv`) compiles only on sm_89+. On the NAS RTX A2000 (sm_86) the 12 of them failed with `CompilationError: type fp8e4nv not supported in this architecture`. They now skip with that reason, and the fp32 gate (`test_group_math_is_the_reference_fp32_math`, no cast) still runs on any CUDA card.

## 0.33.6 — 2026-09-29 — `kernel/mxfp4_pipelined.py`: the MXFP4 prefill combine adds in a fixed order, so identical inputs give identical bits on CUDA (#408, #410); its RTX A2000 receipts and claim (#411); every other shipped module identical to 0.33.5

**0.33.6.** The MXFP4 residency engines' prefill (`Mxfp4PipelinedGptOss._forward_prefill`, T > 1, which the NVMe, Kimi-K3 and DeepSeek-V4 subclasses inherit) no longer returns different bits for identical inputs on CUDA. Its combine added each token's expert outputs with float-atomic `index_add_`, so a forward through it could change from run to run. experts4bit-qlora's Kimi-K3 bench on an RTX A2000 drifted from process to process this way. This affects MXFP4 prefill through these engines on NVIDIA GPUs. Decode (T = 1), the NF4 grouped GEMM, the int4-b32 GEMVs and the MXFP4 QLoRA training path are untouched; the last has the same pattern (#409, open). Upgrade if you compare runs or need reproducible prefill outputs. Prefill values move at the fp32-rounding level, one fixed summation order in place of a varying one, at 287–330 µs more per chunk on the A2000 (`gnf4.kernel.mxfp4-prefill-combine-ordered.a2000.2026-09-28`). No floor change for experts4bit-qlora.

- **The MXFP4 prefill combine adds in a fixed order, so repeated calls give the same bits.**
  `Mxfp4PipelinedGptOss._forward_prefill` summed each token's routed-expert outputs with
  `out.index_add_(0, rows, ...)`. On CUDA that accumulates with float atomics, and a token that meets
  two of its experts in one chunk has its terms added in whatever order the threads win. Replaying the
  combine at Kimi-K3 geometry (topk 16 of 896, 16 slots per chunk, width 7168) on the NAS RTX A2000,
  50 identical calls gave 50 different fp32 outputs at T = 6 and at T = 90, and 2 and 4 different
  bf16 outputs after the cast. The new `_index_add_ordered_` gives every `index_add_` call unique
  rows: pass `r` adds the `r`-th occurrence of each row. Each token's terms now land in ascending
  expert id, one fp32 rounding each, which is bitwise the sequential loop on CPU and on CUDA. The
  decode path (T = 1) is unchanged. It sums the k slots with `.sum(0)` and never used `index_add_`.
  In Kimi-K3's full forward on the same card (experts4bit-qlora#761's driver, 6-token prefill), three
  processes with this file were bit-identical at every one of the 92 MoE calls, and identical to
  `torch.use_deterministic_algorithms(True)` on the shipped file. Three processes on the shipped
  file differed from each other at 3 to 8 of the 92 calls, each time with identical inputs.
  **Cost:** the replayed combine takes 287 to 330 µs more per chunk (69 → 399 µs at T = 6, 67 →
  354 µs at T = 90, 139 → 462 µs at T = 512), for the sort, one host sync and one `index_add_` per
  pass. On a 6-token Kimi-K3 prefill that is at most about 0.2 s (arithmetic: at most 6 chunks
  per layer × 92 layers × 0.33 ms), against the 175 to 185 s the prefill takes on that card.
  **New tests:** `kernel/test_mxfp4_prefill_combine.py` (CPU and CUDA: bitwise equal to the
  sequential loop; 50 repeated CUDA calls bitwise identical) and
  `test_prefill_combine_is_ordered_and_reproducible` in `kernel/test_mxfp4_pipelined.py` (the
  engine routes its combine through the helper, and repeated prefill calls give identical bits).
  Found by experts4bit-qlora#761, where Kimi-K3's p(' Paris') moved from process to process on one
  build. (#408, #410)
- **#410's receipts are registered.** `bench/prefill-combine-a2000/` holds the kernel replay (`combine_repeat`,
  `combine_pr`), the GPU test run (28 passed on the A2000 at `f180045`) and the must-fail control, with SHA256SUMS.
  New claim `gnf4.kernel.mxfp4-prefill-combine-ordered.a2000.2026-09-28` (measured). `gnf4.open.issues` adds #409. (#411)

## 0.33.5 — 2026-09-28 — `kernel/int4_smallm.py` imports on the declared Python 3.9 floor (a postponed-annotations import; no kernel body change), a static CI guard that every shipped module holds the declared floor, and documentation, issue and PR templates and package metadata (the only other shipped-module edit is `gnf4_native/build.py`'s docstring)

- **`int4_smallm` imports on 3.9.** `requires-python` is `>=3.9` and the dependencies support it (torch 2.8, triton 3.4 ships cp39), but `int4_smallm.py:115` had `dot_bf16: bool | None` in a module-level signature without `from __future__ import annotations`, which raises `TypeError` at import below 3.10. The future import fixes it; `int4_b32.py` and `nf4_grouped.py` already pair it with `@triton.jit`. On an RTX A2000 the module's contract passes 9/9 interpreted and 9/9 compiled. **New guard:** `kernel/test_requires_python_floor.py` (in CI) reads `requires-python` and fails on grammar newer than the floor or on a PEP 604 annotation evaluated at import in a module without the future import; it fails on the unfixed module at exactly line 115. CI still runs 3.11 only; the guard is static. (#406)
- **The contributor process says what happens.** `AGENTS.md` §10, `CONTRIBUTING.md` and the PR template: the maintainer
  reviews every pull request, including the maintainer's own, and squash-merges it once the required checks are green; `ci` runs on
  a pull request only after `ready-to-merge` (or leaving draft), the private-marker guard on every push; a release is a
  GitHub Release on the tag (`publish.yml`'s trigger). CONTRIBUTING no longer asks for RTX 5090 runs (sm_120 is the
  primary serving target), records the 2026-07-22 MI300X confirmatory, drops the `out-of-scope` label the repository
  does not have, and names the pre-push hook; the hardware-wanted form now applies `needs-triage` like the other two.
- **Stale statements corrected.** The README: the K16 small-M route is experts4bit-qlora's default from its 0.36.2, not
  opt-in; `examples/dequant_tax.py` needs a repository checkout (the wheel ships modules only); the torch floor is
  stated; the retired split-K is the NF4 dot-pad GEMV's (K7), not the int4-b32 GEMV's. The fp8 solution page's Install
  paragraph and `llms.txt` no longer call the f32 compute path open (#319 closed in 0.33.0). `docs/STATUS.md` moves the
  K17 and K18 reads (closed, negative) to *What changed*. `docs/INDEX.md` points at the serving kernels' solution pages
  and lists `router_probe/`, `sycl/` and `kernel/ERRATA.md`. The int4 solution page names the small-M GEMM and the two
  opt-in GEMV variants with their public claims. `docs/KERNEL_CONTRACT.md`'s status note says which parts of the Gate-0
  record are design, not the shipped signature.
- **Research records annotated, not rewritten.** Dated notes on the MXFP4 seam map (its three STOP items resolved), the
  hybrid-tier architecture notes, the tolerance contract and the NVMe determinism pre-registration; `kernel/ERRATA.md`
  cross-lists the 2026-08-15 `registered_utc` erratum.
- **Metadata.** Trove classifiers name the CUDA environment, the audiences and the Python CI tests (3.11);
  `CITATION.cff`'s `url` is the package page. `gnf4_native/build.py`'s docstring said a failed native build falls back
  to the reference path; it raises, as `kernel/cpu_grouped.py` documents.
- The historical `## Unreleased` heading between 0.10.0 and 0.9.0 is renamed: that section (the `dgrad_kernel`
  default) shipped in 0.10.0, and there was no 0.9.1.

## 0.33.4 — 2026-09-24 — `kernel/cold_deadline.py`: the GPU cost carries a measured per-host link efficiency (`link_eff`, from `bench/calibrate.py`'s new single-copy probe, schema `gnf4-hybrid-calib/2`) and a consumer-passed per-call fixed term (`gpu_us_fixed`); a /1 blob is refused unless `link_eff` is passed explicitly (#400, #402, from experts4bit-qlora lane P66); lane P69's read: the factor is 0.873 / 0.960 on one gen 4 x16 RTX 5090 against 0.64 on another and 1.0 on a gen 3 x8 A2000 — per host, not per card class (#403); every other shipped module identical to 0.33.3

- **Lane P69 read (experts4bit-qlora #741, for #400): `link_eff` is a per-host measurement, not a 5090-class constant.**
  `bench/calibrate.py` at #402's commit, run twice on a rented gen 4 x16 RTX 5090 (host EPYC 7663), reads the 64 MB
  pinned copy at 20.58 / 20.72 GB/s back-to-back and 17.97 / 19.90 GB/s one at a time — `link_eff` **0.873 / 0.960**,
  against the registered [0.55, 0.75] from lane P66's census probe on another gen 4 x16 host (0.638, EPYC 7C13): P1
  REFUTED, P2 held, P4 (repeatability) held at 9.7 %. The two hosts' difference is stated, not explained. Receipt
  `bench/cold-engine/calib-5090-p69/`; claim `gnf4.calib.link-efficiency.5090.2026-09-24`; a test loads the blob.
  Nothing moves: `Costs.from_blob` already reads each box's own figure, which this read shows is the only correct place
  for it.
- **`kernel/cold_deadline.py`: the GPU cost gets a measured link efficiency and a per-call fixed term (#400, from
  experts4bit-qlora lane P66).** Scored against a real gather on an RTX 5090 (PCIe gen 4 x16), the bytes-over-link
  term under-predicted the transfer 1.57–2.02x: the consumer's pipelined gather runs at the box's SINGLE-copy H2D rate
  (14.72 GB/s probed, 14.44 implied) while the calibration's `b_link` is the 40-deep back-to-back rate (23.07); on a
  gen 3 x8 A2000 the two agree and the model read 0.97–1.02x. `Costs` gains `link_eff` (default 1.0 = the pre-#400
  model) and `gpu_us_fixed` (default 0.0; the consumer's own measurement of its path's fixed per-call kernels — P66:
  +10 launches, +2 copies per layer whatever the cold fraction, 24.7 µs/layer captured, 168 eager on that box), and
  `gpu_us` is `bytes / (b_link · link_eff) + bytes / b_vram + gpu_us_fixed`, still flat in rows. `Costs.from_blob`
  derives `link_eff` from the blob's new `b_link.h2d_64mb_single` probe (`bench/calibrate.py`, schema
  `gnf4-hybrid-calib/2`: the same 64 MB pinned copy one at a time, synchronized on each side, median of 10) and
  **raises on a /1 blob unless `link_eff=` is passed explicitly** — `bench/cold-engine/gate2/run_gate2.py` passes 1.0
  for its /1 receipts and says so. Every consumer that builds `Costs` by keyword keeps its numbers (the defaults are
  the old model to the bit; `test_the_defaults_reproduce_the_bytes_only_model`). Not yet in any blob receipt: the
  5090 figure is P66's census probe, not this script's — a registered probe on a gen 4 box follows. Still not
  modelled: PCIe contention, and the gather's per-row compute beyond the fixed term.

## 0.33.3 — 2026-09-24 — register, tests and repository tooling only (every shipped module behaves as in 0.33.2; only `kernel/int4_b32.py`'s two docstrings changed, to state lane B393's measured contract): lane P63's row-count-invariance kernel claims with `kernel/test_row_invariance_gpu.py` in CI; lane B393 read (#393 closed) and its cross-architecture correction (#398); #386's 2^31 boundary cases observed on CPU; the CI scripts shared with experts4bit-qlora are one file each

- **Row-count invariance registered from experts4bit-qlora lane P63 (#708), kernel first.** The lane read, on one
  RTX 5090 at Qwen3-30B-A3B's gate_up on the model's own activations and routing, whether a token's rows get the same
  bits alone (T = 1) and inside a 16-, 17- or 160-token call. Every registered prediction held.
  - **Row-invariant (EXACT):** `gemv_int4_b32` (above 64 SMs its split-K plan is N-only), the NF4 dot-pad decode GEMV,
    and `combine_rows`. Claims: `gnf4.kernel.int4-gemv-row-invariant.5090.2026-09-24`,
    `gnf4.kernel.nf4-dotpad-gemv-row-invariant.5090.2026-09-24` and
    `gnf4.kernel.combine-rows-row-invariant.5090.2026-09-24`.
  - **Reorder-class:** the grouped int4 GEMM against the GEMV, and the scalar NF4 GEMV under `GNF4_GEMV_DOTPAD=0`
    (split-K planned from the rows). Claims: `gnf4.kernel.int4-grouped-gemm-reorder.5090.2026-09-24` and
    `gnf4.kernel.nf4-scalar-gemv-splitk-reorder.5090.2026-09-24`.
  - **Every path was inside its own operand model's fp64 bound.** The census is in `kernel/receipts-p63/5090/`.
  - **New test:** `kernel/test_row_invariance_gpu.py` asserts the three invariant kernels with `torch.equal` on any
    CUDA part. It includes a control that must see a split-K change. It skips on CPU, and on the NAS RTX A2000 it
    passes 18 of 18 (`kernel/receipts-p63/a2000-tests/`).
  - **B393's end-to-end size, recorded as that lane asked:** `E4B_FUSE_COMBINE=0` against the default at T = 1 is KL
    1.18e-04 nats/token on the NF4 stack and 1.70e-02 on the int4 stack (7 of 160 argmax flips). This is added to the
    note of `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23`.
  - **No kernel changed.**
- **Correction (2026-09-24): lane B393's cross-architecture statement is retracted.** The read said neither
  `combine_rows`' bits nor torch's own chain are the same on sm_86 and sm_120. That was inferred from per-case
  count differences between the lane's census and the A2000 rehearsal, not from comparing outputs.
  - **What refutes it.** The attribution diagnostic, which records a sha256 of every output, ran on a second RTX 5090
    (experts4bit-qlora P63's box). The fused output, torch's chain and the sequential sum are **bit-identical on the
    two architectures in 144/144 cases**, and `combine_rows` is the FMA slot-order sum on sm_120 too
    (`kernel/receipts-b393/5090-fma-attribution/`).
  - **What is left unexplained.** The lane box's own census disagrees with both hash-checked runs in 37 cases.
  - **Corrected in** `RESULTS-b393-combine-reduce-bitwise.md`, STATUS, the `combine_rows` docstring, the
    capabilities limitation and the claim note of `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23`.
  - **What stands.** Outcome B, the accuracy contract and both claims' values.
- **Lane B393 read (#393 closed): `combine_rows` and `reduce_partials` carry an accuracy contract, not a bitwise one.**
  The lane was pre-registered in #396 and run on one RTX 5090 for $0.0299, with teardown proven. The census covered
  414 cases at the served shapes, and the result is outcome B for both kernels.
  - **Neither kernel is bitwise its torch chain.** `combine_rows` differs in 95/144 cases (392 of 9.07 M elements) and
    `reduce_partials` in 49/270.
  - **Both are within the error bound of a correct fp32 summation in every case,** at the same max ratio as the chain.
    The claims are `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23` and
    `gnf4.kernel.reduce-partials-slot-order.5090.2026-09-23`.
  - **Attribution.** `reduce_partials` is bitwise the slot-order sum. A follow-up diagnostic on the NAS A2000 (sm_86)
    shows `combine_rows` equal bit-for-bit to the slot-order sum with a fused multiply-add. Its PTX has `fma.rn.f32`
    only (`kernel/receipts-b393/a2000-fma-attribution/`).
  - ~~**Found while reading it.** Neither the kernel's bits nor torch's own chain are the same across sm_86 and sm_120.~~
    **Retracted 2026-09-24** (see the correction entry above): output hashes show both are bit-identical on the two
    architectures. The accuracy bound holds on both.
  - **Docstrings corrected.** `combine_rows` said "fp32 in slot order, as the torch chain's is". Neither half held.
  - **Tests.** `test_combine_rows_matches_torch` and `test_reduce_partials_matches_torch` replace the whole-tensor
    `max|d| <= max|ref| * 2**-7` with the per-element bound. Under `TRITON_INTERPRET=1` the cast term is one full ULP;
    the interpreter's truncating cast was measured at 1.98× the half-ULP bound. The new
    `test_reduce_partials_is_the_slot_order_sum` asserts `torch.equal` on silicon.
  - **No kernel changed.** The end-to-end size is experts4bit-qlora#708's to measure, with `E4B_FUSE_COMBINE=0` as its
    control.
- **`kernel/test_offset_boundary_interp.py` gains five cases (#386).** KERNEL_CONTRACT listed these kernels as carriers
  of the int64 expert-base promotion covered by the boundary suite, and no boundary test called any of them. They are
  `host_gather._gather_rows`, the `mxfp4_pipelined` / `mxfp4_residency` slot gathers, and the `fp8_kv` T=1 and batched
  appenders.
  - **The gathers** multiply an int32 id or slot by `row_words`, a stride in int64 words, so they wrap at 2^31 words
    (16 GiB). An int32 wrap moves an address by 2^32 words, so each case spans ~32 GiB of address space. It is mapped
    `MAP_NORESERVE`: the heuristic overcommit check refuses a `torch.empty` that large on a 16 GB runner and does not
    charge this mapping. Resident memory stays a few hundred MiB. The verdicts are exact copies, plus an untouched decoy
    at the wrapped address for the two that write past the boundary.
  - **The fp8 appenders** are byte-addressed. Their cases check WHERE the bytes land: the true row dequantizes to the
    input, and the wrapped row keeps its decoy. The e4m3 rounding stays `test_fp8_kv_append.py`'s GPU-only bitwise
    gate. The public wrappers refuse CPU tensors, so the cases launch the kernels with the wrappers' arguments.
  - **Calibrated** in the CPU container (torch 2.8.0 / triton 3.4.0 / numpy 2.3.2). The whole file, 20 cases, passes
    on the shipped kernels. A copy with exactly the five straddling promotions removed fails all five new cases, each at
    the wrapped address (`kernel/receipts-386/interp/`, claim `gnf4.kernel.boundary-gathers-appenders.interp.2026-09-23`).
- **`docs/KERNEL_CONTRACT.md`** says where those carriers are covered. `gnf4.open.issues` drops #386. No kernel changed.
- **`scripts/check_capabilities.py` (shared with experts4bit-qlora): the serving position is read from STATUS's position
  section.** The serving-position WARN takes "the position" to be the newest `area: serve` claims STATUS quotes. It read
  the whole file, so a lane quoted under "What is open" counted too. Since 2026-09-23 that was P61's expert-GEMV cost
  split, a diagnostic read, and the rule warned on every experts4bit-qlora CI run.
  - **The fix.** It now reads only the text before STATUS's first `## What changed` heading, or the whole file when there
    is none. "What changed" records retired rows and "What is open" pending or diagnostic ones.
  - **Checked.** On experts4bit-qlora's register the position is then P58's 2026-09-22 rows, which the capability cites,
    so the warning clears without any data edit. The runtime repository's existing serving-position tests pass unchanged.
  - The rule is still the runtime role's only.
- **Four documents still called the fp8 paged kernel's f32 compute path open under #319**, although #319 closed in 0.33.0
  and the path has been `supported` since, measured on sm_86 and sm_120 (`gnf4.serve.f32-arms-ran-fp8`). They were
  `docs/SOLUTIONS.md`, `docs/INDEX.md`, the fp8 solution page's summary line and `AGENTS.md`. `AGENTS.md` also presented
  the retired `gnf4.open.f32-compute-modes-triton34` as current and called the capability `unsupported`. All four now
  match `docs/capabilities.json` and STATUS. 0.33.1's cleanup (#381) had fixed the capability, the README and the
  solution page body, but not these.
- **`scripts/check_system_manifest.py` is one file, shared with experts4bit-qlora** (now in `SHARED`). The two copies
  had forked by 734 diff lines: gnf4's had 31 rules and e4b's 39, and one check name enforced two different rule sets.
  - **How it picks its rules.** It reads its role from the manifest, the way `check_capabilities.py` does: the
    `packages` entry whose `package` is pyproject's name. It runs the rules for that role.
    - Both roles run every manifest-only rule.
    - The kernel role runs the kernel-first floor checks against its own pyproject version and LOCAL tags, and never
      touches the network.
    - The runtime role runs the consumer's `fast` / CI-pin checks.
    - Under `--sibling` each role checks the other's side of the contract.
  - **Strictness.** Where the copies differed, it keeps the stricter reading. It applies a rule found in only one copy
    to both roles wherever the rule means something in both.
  - **Parity.** Old and new copies give identical verdicts on both repositories' `main`, in every CI form. 152 mutation
    runs found no case where an old copy failed and the new file passed. 36 runs fail only under the new file; these
    are rules that now apply here, and both `main`s pass them.
  - **Newly checked here.** The `schema_version` key; an import name with no module file; an empty `why`; duplicate
    ownership ids; whitespace-only invariant fields; and, under `--sibling`, the runtime's kernel-pinning extras and its
    ownership list.
  - **Refused as ambiguous.** A bare two-part range such as `"0.37"`. The kernel's old copy read it as `0.37.*` and the
    runtime's as exactly `0.37.0`.
  - **API and CI.** `current_record` is renamed `current_kernel_record`, and `check_dependency_floor.py` imports the new
    name. The runtime's `current_record` is a different function. CI now runs the check with `--require-tags`: the
    discoverability job already fetches the tags, so the floor-vs-tag comparison can no longer be silently skipped.
- **`scripts/check_readme_claims.py` is one file, shared with experts4bit-qlora** (now in `SHARED`; 12 shared files).
  The two copies had forked by 399 diff lines. The unified file reads its role through
  `check_system_manifest.system_role` and reads the claim-id namespace from the register itself.
  - **Shared rules.** Headline numbers must be the register's current values, cited ids must exist, and an inactive id
    may appear only on a line that says so. Both roles apply these.
  - **Per-role settings (`PROFILES`).** Where the copies still differ, each role keeps its own setting, and each
    setting carries the reason in a comment. For all but one, switching to the other role's value either fails that
    repository's `main` or lets through an input its old copy failed. The runtime's exemption for anchored documents
    is kept by choice: a finding there could only be fixed by editing an anchored file.
  - **Now also checked here.** A results row must have a result column. Ids cited as `<code>` or as link text are
    checked like backticked ones. The line must name the id's own status. The status-word test ignores URLs, HTML
    comments and bare ids, so an id such as `gnf4.retired.x` cannot vouch for itself. `docs/SOLUTIONS.md` is covered.
    The manifest is required, and without it the check exits 2.
  - **Parity.** 110 mutation rows across both repositories give 0 cases where an old copy failed and the new file
    passed. On `main` and on the v0.33.2 tag, old and new give the same exit code. CI's test step passes 124/124
    unchanged.
- **`scripts/check_change_impact.py` and `scripts/check_dependency_floor.py` are one file each, shared with
  experts4bit-qlora** (both now in `SHARED`; 14 shared files, every CI script both repositories carry except the
  per-package `wheel_smoke.py`). Each reads its role through `check_system_manifest.system_role` and keeps the settings
  that still differ in a commented `PROFILES` dict.
  - **Change impact.** Both roles now diff against `git merge-base BASE HEAD`, which is this copy's rule.
    - A branch behind its base is no longer charged with what landed on the base since it forked. The runtime's old
      copy diffed against the base itself, and so both blamed such a PR for a claim change it never made and passed one
      whose missing companion the moved base happened to supply.
    - Untracked files count as added.
    - The contract is read in either shape, and every class a trigger reports must be named in it.
    - A claim's `unit` change is now a measured-result trigger alongside `status` and `value`.
    - The new-module, new-`@triton.jit` and layout-constant triggers stay kernel-only.
  - **Dependency floor.** The kernel-floor statement rule reads both copies' forms in both roles, over the union of both
    copies' documents: `requires X`, `≥`, `ci.yml`, all of STATUS and the documents INDEX lists under "Current". The
    floor's source, the historical-line markers, the anchored-document exemption and the consumer-version pins stay per
    role.
  - **Parity.**
    - 127 mutation rows. The one where an old copy failed and the new file passes is the merge-base false positive
      above.
    - 644 historical PR ranges, with no weakening wherever the new file can run. Ranges from before the manifest existed
      exit 2, because the file cannot tell which package it is there.
    - On both `main`s, every CI form gives the same exit code and findings. This repository's checker suite passes
      124/124 unchanged.
- **`gnf4.open.issues` and STATUS list #393** (filed 2026-09-23): `combine_rows` has no bitwise contract, and
  experts4bit-qlora runs it by default. The register had said the open issues were #60 and #71 only.
- **Lane B393 registered (#393): are `combine_rows` and `reduce_partials` bitwise equal to the torch chains they
  replace?** Both are tested only to a `2**-7`-relative tolerance.
  - **Why it matters.** experts4bit-qlora runs `combine_rows` on every MoE layer by default, and its call site says
    the fused path takes "the same order and roundings".
  - **The census.** `kernel/b393_bitwise_census.py` compares each kernel with the verbatim chain and with a strictly
    sequential sum, in bf16 ULPs. It covers every served family's `(top_k, hidden)` at T = 1/16/17/64, and split-K
    factors 2–16. `kernel/test_b393_census_helpers.py` checks the instrument on CPU in CI.
  - **The pre-registration.** `kernel/PREREG-b393-combine-reduce-bitwise.md` gives the decision rule: bitwise, a
    bounded reorder-class, or a defect.
  - **Why the GPU.** An interpreter dry run executed every case but is not a reading: the interpreter's own bf16 cast
    rounding puts about half the elements 1 ULP off.
  - **A rehearsal changed the metric before registration** (disclosed there, with its receipt in
    `kernel/receipts-b393/a2000-rehearsal/`, NOT a reading).
    - It ran on the NAS RTX A2000. `combine_rows` differed from the chain by up to 36 bf16 ULP near cancellation,
      while staying within the fp32 summation bound. A "≤ 1 ULP" rule would have called correct arithmetic a defect.
    - The correct/defect line is now accuracy against the exact (fp64) sum: `bound_ratio <= 1` means a correct fp32
      sum in some order.

## 0.33.2 — 2026-09-23 — documentation, register data, tests and repository tooling only (every shipped module identical to 0.33.1): the claims register has ONE schema, shared with experts4bit-qlora; lane B374 observes the word-addressed decode routes (wide loads, and dot-pad, the default at its census shapes) past their own 2^31 boundary on an RTX 5090, closing #374; #386 opened

- **`scripts/check_claims_register.py` and `docs/claims-schema.md` are one file each, byte-identical in both
  repositories** (both now in `SHARED`). The two copies had drifted into two schemas that refused each other's data (14
  findings one way, 38 the other). The converged rules, in `docs/claims-schema.md`:
  - **Locations.** An evidence path or a `quoted_in` entry is `path` or `path#anchor`. The path is a file in the git
    tree at HEAD, never a directory, annotation or glob. The anchor is `L<n>` / `L<n>-L<m>`, or the anchor
    **github.com renders** for a Markdown heading, so every location is a working link. The checker's anchors match
    github.com's on all 371 headings of both repositories' READMEs, CHANGELOGs and docs.
  - **Evidence** is a location, `{"url"}` (an issue or pull request of a system repository), or
    `{"repository": <package>, "path": <location>}` (resolved in a `--sibling` checkout). A public-run row's FIRST
    entry is a location, because the consumer site links it.
  - **Successors** are direct and named back. `superseded_by` names the active row itself, with no chains, and that
    row lists it in `supersedes`. A retired row may name its restatement.
  - **The file is closed.** Unknown row fields and top-level keys are findings, and `area` / `tier` / lane fields take
    only their values.
  - The licence, fingerprint, date and placeholder rules are the union of both repositories' old rules.
- **Tests.** One test file, identical in both repositories, produces every rule's failure. A mutation sweep disabling
  each of 32 rules in turn was caught every time.
- **This register's migration.**
  - `{"path", "section"}` objects became `CHANGELOG.md#<anchor>` locations. The note on `gnf4.kernel.dgrad`'s entry
    moved to its `notes`.
  - URL strings became `{"url"}`, and `gnf4.open.issues`'s bare tracker URL became #60 / #71 / #374.
  - `kernel/receipts-m3/` became its index `m3_manifest.txt`.
  - Heading-prefix quotes became anchors.
- **A defect the migration found.** K17's and K18's `{"section": "Unreleased"}` evidence had been resolving, under the
  old prefix rule, to an unrelated historical `## Unreleased` heading 1,400 lines down. Both now name the 0.33.0 entry
  that carries them.
- **CI** now resolves the cross-repository evidence in experts4bit-qlora's `main`, where before it was only listed as
  SKIP.

- **`kernel/test_offset_boundary_words_gpu.py` and `kernel/PREREG-b374-word-boundary-gpu.md` (#374).** The wide-load and
  dot-pad NF4 decode routes address the stack in 32-bit words, so they wrap at 2^31 words (8 GiB of packed bytes), and
  `test_expert_offset_boundary.py`'s byte geometry never reaches that. Dot-pad has been the default decode route on
  >= 160-SM parts since M3, and no test had put it past its boundary.
  - **The test.** A real 16 GiB device buffer puts the target expert past 8 GiB and a decoy at the int32-wrapped address.
    A wrap is therefore a detectable misread, not a fault.
  - **The cases.** Four: wide split 1 and 4, dot-pad and dot-pad split-K at the census gate_up shape. Each asserts its
    route, and each is skipped legibly below 17.5 GiB free. The pure geometry check runs in CI everywhere.
  - **The lane.** B374, one RTX 5090 at <= $0.66, runs the file against the shipped kernels (P1: all pass) and against
    a copy with all SIX eid promotions removed (P2: all fail).
  - **Why six.** The dot-pad kernels promote in the load itself (`tl.load(eids_ptr + g).to(tl.int64)`), so the four
    `eid = eid.to(tl.int64)` lines alone would leave dot-pad promoted and calibrate nothing.
- **Lane B374 read (2026-09-23, RTX 5090): P1 and P2 both hold, and #374 is closed.**
  - **P1.** On the shipped kernels (0.33.1 at `94a9ff4`), all 4 cases pass, none skipped.
  - **P2.** On the copy with the six promotions stripped, all 4 fail by reading the decoy at the int32-wrapped
    address, with no faults. Wide: rel 1.467 vs the true tile, 1.822e-03 vs the decoy. Dot-pad: 1.405 / 2.408e-03.
  - **Where it is recorded.** `kernel/RESULTS-b374-word-boundary-gpu.md`, receipts in `kernel/receipts-b374/5090/`,
    claim `gnf4.kernel.word-boundary-wide-dotpad.5090.2026-09-23` (measured). The run cost $0.0354 and its teardown
    is proven.
  - **Correction.** `gnf4.kernel.expert-offset-boundary.5090.2026-09-05` gains a note: its wide and dot-pad arms
    straddle 2^31 bytes, not their own boundary. Its value and its other routes stand.
  - `gnf4.open.issues` drops #374 and adds #386. KERNEL_CONTRACT had said the boundary suite covers "each
    carrier", but it never calls the gathers (`host_gather`, `mxfp4_pipelined`, `mxfp4_residency`, all
    word-addressed) or the `fp8_kv` appenders. Their promotion is correct by inspection and untested past 2^31.
    The sentence now says so. No kernel changed.

## 0.33.1 — 2026-09-23 — documentation and repository tooling only (every shipped module identical to 0.33.0): the open-issue list and #319's closure corrected across README, STATUS, the register, the capabilities and the solution pages; the CI scripts shared with experts4bit-qlora start to become one file

- **`scripts/check_shared_tooling.py`** (new, byte-identical in experts4bit-qlora): `SHARED` lists the scripts that are one
  file in both repositories. Nine of the thirteen same-named scripts had forked, so one check name enforced two rules; the
  four already identical, the two reconciled here and the checker itself are in `SHARED`. Without `--sibling` it checks
  every listed path exists; with `--sibling` it refuses a self-comparison, a sibling outside the system, a shared file
  missing there and any differing byte, and lists the scripts still forked as NOTEs. This repository is upstream for
  shared tooling (kernel-first, as for the manifest); the consumer's CI compares its copies with this `main`.
  `kernel/test_check_shared_tooling.py` produces every failure the check claims and asserts it is detected.
- **`scripts/check_readme_links.py`** now derives the README tag pin from `CHANGELOG.md`'s latest `## <version> — <date>`
  heading cross-checked against `pyproject.toml` (experts4bit-qlora's rule). The old copy read `pyproject.toml` alone, so a
  version bump without its CHANGELOG section passed; it is now refused (calibrated on this repository's files).

- **Docs: #73 and #58 were still listed as open** in README, STATUS, the `gnf4.open.issues` claim and the NVMe solution
  page. #73 was fixed in 0.12.0 (#74/#76/#79: 1113 → 56.8 ms per layer on real K3 bytes, 78% of device); #58 was answered
  by its own trace (six of eight calls per layer are cache hits, 0.02 s/step; the stall is #60). The claim also still
  named #319, closed in 0.33.0. The open list is now #60, #71 and #374 (#353 closed the same day by experts4bit-qlora#697); the claim's previous text is kept in its notes.

- **Docs: #319's closure had not reached three places.** `fp8-paged-attention-f32-compute` in `docs/capabilities.json`
  still opened with "Open, #319 (claim `gnf4.open.f32-compute-modes-triton34` …)" and still advised gating lanes with
  `-k "f8dot or pf8"`, which excluded exactly the mislabelled arms; the fp8 solution page's GPU example called the f32 path
  "open under #319"; and the README's reproduce block still ran `test_fp8_paged_attn.py` with that filter. All three now say
  #319 is closed and cite `gnf4.serve.f32-arms-ran-fp8`. Found by the consumer site's claim-reference check, which refuses a
  retired id presented without the word "retired".

- **`scripts/check_capabilities.py` is one file in both repositories** (added to `SHARED`). This repository's copy was a
  strict subset of the consumer's; the union adds the consumer's `training_support` cross-reference (inert here: this
  schema admits no such key) and its serving-position rule, which is now gated on the repository's role in
  `docs/system-manifest.json` instead of silently never matching because of a hard-coded `e4b.` id prefix. The rule
  assumes one serving position per repository and would warn falsely here (two false warnings measured when the prefix
  was derived and the gate absent). `kernel/test_check_capabilities_role.py` pins the gate from this side.

## 0.33.0 — 2026-09-23 — two opt-in int4-b32 GEMV variants, both exact and neither a default (K17 `fused_reduce=True`, K18 `gemv_int4_b32_grouped`); #319 was a mislabelled test arm, not an f32 kernel defect; #87's int64 expert-offset promotion is now guarded by a CPU test CI runs

- **K18: `int4_b32.gemv_int4_b32_grouped(xq, xs, packed, scales, eids, N, K, part=None, out=None, mt=4)` — the split-K
  int4-b32 GEMV with each program serving up to `mt` rows of ONE expert, so the expert's weight slice is loaded once per
  program instead of once per row.** New kernel `_gemv_int4_b32_grouped`: grid `(cdiv(N, BLOCK_N), R, SK)` (the served
  GEMV's, from the same `_plan`); each program derives its tile of the call's expert-major tiling in-register from the `R`
  expert ids (`tl.histogram` over `next_pow2(E + 1)` bins, `tl.cumsum`, per-expert rank — no sort, no extra launch, no
  host sync, capture-legal); rows are computed with the served GEMV's per-row arithmetic and their fp32 partials stored at
  the ORIGINAL row index, so the served `reduce_partials` runs unchanged and the result is `torch.equal` to
  `gemv_int4_b32(..., fused_reduce=False)`. Distinct from `_gemm_int4_b32_grouped` (16-row MMA tiles, no split-K), which
  P7 measured 1.92×/1.28× slower than this GEMV at decode. Pre-registered (`kernel/PREREG-k18-grouped-expert-gemv.md`):
  P1 bitwise identity, P2 ≥ 0.5 ms/step saved on experts4bit-qlora P60's recorded Qwen3-30B-A3B B=16 routing (where
  repeated rows cost 0.92 ms/step), P3 no regression beyond +3 % at R = 8/16. **Read on the RTX 5090 2026-09-22
  (`kernel/RESULTS-k18-grouped-expert-gemv.md`, lane `k18-5090-1`, $0.1269): P1 HOLDS (0 of 256 replay checks differ);
  P2 REFUTED — grouped 9.689 against served 6.520 ms/step (1.49×, 3.17 ms slower); P3 REFUTED — 1.00–1.65× at R = 8/16,
  worst where every row is a distinct expert and nothing can be shared.** Decision rule `P1 ∧ ¬P2` → exact-and-not-faster,
  not a lever: it stays dormant as the evidence, nothing routes to it. Register row
  `gnf4.kernel.k18-grouped-expert-gemv.5090.2026-09-22`. The read also withdraws P60's reading of its 0.92 ms dedup gap as
  weight re-streaming (sharing the loads while keeping every row's arithmetic did not recover it). `GROUPED_MT_DEFAULT = 4`; `GROUPED_E_MAX = 4095` bounds the histogram at 16 KiB and
  the wrapper refuses larger `E` with `UnsupportedShapeError` before the launch (KERNEL_CONTRACT's feasibility table).
- `kernel/test_int4_b32_grouped_interp.py` (bitwise vs the served GEMV: one row, all-distinct, one expert across many
  tiles, skewed, non-power-of-two `R`, ragged `N`, `SK > 1`, `mt` ∈ {1, 2, 4, 8}; compiled adds Qwen3-30B-A3B's expert
  shapes at R=128 and a CUDA-graph replay; the `GROUPED_E_MAX` refusal) — named in `ci.yml`'s interpreter job and
  guarded in `conftest._INTERP_FILES`. `kernel/test_expert_offset_boundary.py` gains an `int4_b32_grouped` case: 2 MiB
  per expert, experts 1023/1024 straddling 2^31, bitwise vs the served GEMV and within tolerance of `dequant_int4_ref`.
  `kernel/k18_bench.py` and `kernel/k18_reduce.py` are the lane's instruments (deliberately not in the wheel); receipts
  in `kernel/receipts-k18/5090/`.
- **K17: `gemv_int4_b32(..., fused_reduce=True)` / `GNF4_GEMV_FUSED_REDUCE=1` — the int4-b32 GEMV can fold its split-K reduce into
  its own launch (fp32 partials, `acq_rel` atomic counter per (row, column block), last arriver sums in split order, casts once;
  `gemv_counter_len` sizes the `cnt` workspace; `cnt`/`out` allocated when not passed). OFF by default.** Pre-registered
  (`kernel/PREREG-k17-fused-splitk-gemv.md`, #372) and read on the RTX 5090 2026-09-21 (`kernel/RESULTS-k17-fused-splitk-gemv.md`,
  lane `k17-5090-1`, $0.0888): **P1 bitwise identity HOLDS** on all 24 shape × R rows (interpreter 197/197, compiled 27/27, the
  counter reads zero after 800+ graph replays); **P2 REFUTED** — at R=1 the fused path saves 2.04 µs on `expert_gate_up`, 0.87 on
  `attn_kv` and ≤ 0.02 µs on the other four shapes (both paths 6.20 µs), so the removed launch was hidden in the graph, not on the
  critical path; **P3 HOLDS at the slow end** (fused/two-launch 1.094 / 1.061 at R=128 on the expert shapes; the attention shapes
  6–12 % slower there, unregistered). Decision rule `P1 ∧ ¬P2` → ships opt-in; no R at which it is a default. Register row
  `gnf4.kernel.k17-fused-splitk-gemv.5090.2026-09-21`. `_gemv_int4_b32`'s signature gained `cnt_ptr`, `out_ptr`, `FUSED_REDUCE`
  (constexpr); with `FUSED_REDUCE=0` the kernel body is byte-for-byte the shipped one.
- `kernel/test_int4_b32_fused_reduce_interp.py` (named in `ci.yml`'s interpreter job and guarded in `conftest._INTERP_FILES`),
  `kernel/k17_bench.py`, `kernel/k17_reduce.py` (campaign instruments, deliberately not in the wheel).
- **`kernel/test_fp8_paged_attn.py`: every shape arm now NAMES its compute mode, and a new test asserts
  the kernel's own tally per arm** (#319). `_modes()` built its two f32 arms as `("split", {})` and
  `("packed", {"pack_heads": True})` — no `compute` key. That was correct only while an unset `compute`
  meant f32; since RESULTS-m3-default-on the default is capability-conditional and `_compute_default`
  returns **fp8 on sm_89+**. So on every Ada/Hopper/Blackwell card those arms have been running the
  **fp8** kernel, while `_close` judged them at the **f32** tolerance (2e-2) because it reads the
  tolerance off the arm's *name* — the fp8 path's own envelope being 1.5e-1. Measured on an RTX 5090
  (`compute_counts()` after one call per arm = **`{'f32': 0, 'fp8': 4}`**; `split` and `f8dot` return
  byte-identical tensors, as do `packed` and `pf8`). The 27 failures that card reported for the f32
  modes were the fp8 path's documented q/p rounding (relative Frobenius 0.049, exact at T=1) checked
  against a tolerance written for a kernel that was not running. An RTX A2000 passes the identical
  suite on the identical torch 2.8.0+cu128 / triton 3.4.0, because sm_86 has no fp8 MMA to default to.
  `test_modes_run_the_kernel_they_name` holds it shut: both paths return a plausible tensor, so the
  tally is the only witness. The claim `gnf4.open.f32-compute-modes-triton34` is **retired** (kept so
  the retraction is findable) and replaced by `gnf4.serve.f32-arms-ran-fp8`.

- **`fp8_paged_attn`: the f32 split path's dot precision is a stated arm, not an inherited one**
  (`GNF4_ATTN_F32_PRECISION` = `tf32` (default, unchanged behaviour) | `tf32x3` | `ieee`; an
  unrecognised value raises). Its two dots took fp32 operands and passed no `input_precision`, so what
  they computed was whatever Triton picks per architecture. Measured on the A2000 — the class where f32
  IS the serving default — at B=25 T=4096 H=32/8 D=128: `tf32` 0.015625 worst error (one bf16 output
  ULP) at **77.9 GB/s**; `tf32x3` **exact** at 39.5 GB/s (0.51×); `ieee` exact at 4.5 GB/s (0.06×), and
  in the emitted PTX `ieee` carries **no `mma.sync` at all** — 1045 `fma.rn.f32`, off the tensor cores
  entirely, below the 4.8 GB/s occupancy-starved first version this kernel replaced. #319's suggested
  remedy was to pin `ieee`; it is priced here and refused as a default. **This knob is not a fix for
  #319 and is not offered as one.**

- **`head_dim` 256 is covered in EVERY mode** (#324, the half that was still open). The issue asked for
  it and only the fp8 half got a test; the f32 packed kernel has no pre-launch fit model, only the
  `OutOfResources` catch, and that catch had never been exercised. The new case reproduces #324's
  numbers on the A2000 to the byte — `Required: 148480, Hardware limit: 101376` — warns once, remembers
  the geometry in `_PACKED_UNFIT`, and returns the split kernel's result, which matches the oracle.

- **#87 closed: the int64 expert-offset promotion is guarded by a test CI can actually run** (#373). Every
  `@triton.jit` body was audited from the installed 0.32.1 wheel source and all expert-id and row carriers promote before
  the stride product, so no kernel changed. What was missing was a check that could fail: both boundary suites need CUDA
  and ~2.3 GiB of device memory, so they skip in full on every runner, and a port has dropped the promotion before
  (#205). `kernel/test_offset_boundary_interp.py` straddles 2^31 on CPU under `TRITON_INTERPRET=1` (15 cases, ~42 s, a
  new `ci.yml` step), with the wrapped and correct offsets both inside the mapping so a wrap misreads a decoy tile instead
  of faulting. Falsified per module with the promotion removed: `nf4_grouped` 7 failed, `mxfp4_grouped` 3, `int4_b32` 2.
  Two corrections recorded in `docs/KERNEL_CONTRACT.md`: the Triton interpreter **does** wrap int32 offset arithmetic
  (NEP 50; the 0.14.0 entry saying it cannot is superseded, and `test_interpreter_arithmetic_still_wraps` pins it), and
  the wide-load and dot-pad routes are word-addressed, so they wrap at 8 GiB of packed bytes, not 2 GiB — the GPU
  suite's wide and dot-pad arms are built on the byte geometry and do not straddle their own boundary.

## 0.32.1 — 2026-09-19 — the `auto` LoRA-delta rule is STRUCTURAL (pad unless the padded block would not fit): P46 read the 4× flop-waste guard as the defect behind the consumer's launch-bound training step

- **`nf4_qlora.lora_delta_grouped`: the `auto` rule is now STRUCTURAL — pad unless the padded block would not fit** (P46 read,
  experts4bit-qlora `bench/p46/RESULTS-p46.md`, run `p46-qwen3lora`, RTX 5090, Qwen3-30B-A3B at the field recipe). The shipped 4×
  flop-waste guard sent ≥ 85 % of the adapter calls to the per-expert loop on every step; the padded `bmm` path forced on trained
  the same tokens to the same loss (held-out Δ +0.0007 nats, median per-step |Δ| 0.0024, the 0.05 band) at the same peak VRAM
  in **4.22 s/step against 24.46** (0.173×; 365 vs 62 tok/s). The loop is launch-bound and the flops padding wastes are
  rank-r matmuls, so waste was the wrong quantity to guard. `auto` now pads unless `G · max(rows) · (K + N) · itemsize`
  exceeds `NF4_QLORA_PAD_BYTES_LIMIT` (default 2 GiB); `NF4_QLORA_PAD_WASTE_LIMIT`, when SET, re-arms the old ratio guard on
  top; `LORA_PAD_WASTE` records the last / max waste ratio and the padded bytes for the census. `grouped_mm` stays opt-in
  (torch 2.8's `_grouped_mm` is sm_90-only — the 5090 refused it, recorded). **The consumer's training position does not move
  from this entry**: experts4bit-qlora re-runs its field-recipe head-to-head (tp4 box B) on this cut and quotes from that receipt.

- **`nf4_qlora.lora_delta_grouped`: the adapter delta's path is a recorded choice** (P46, experts4bit-qlora
  `bench/p46/P46-PREREG.md`). `NF4_QLORA_LORA_PATH` = `auto` (the shipped rule, unchanged: padded bmm below the padding-waste
  limit, the per-expert loop above) | `padded` | `loop` | `grouped_mm` (two `torch._grouped_mm` calls over the jagged
  groups with cumulative offsets — no padding, no per-expert Python; CUDA/bf16; refuses with the reason where torch has no
  kernel for the part, never a silent fallback); `NF4_QLORA_PAD_WASTE_LIMIT` overrides the 4× guard; `LORA_PATH_STATS`
  counts calls per path so a training census can prove which path served a step. Why: experts4bit-qlora's P45 census read
  ~273k per-expert `aten::mm` per optimizer step on Qwen3-30B-A3B's field recipe — `auto` taking the loop under router skew
  — with the GPU 10.8 % busy. No default changes here; the arms that decide one are registered on the consumer side.

- K16 P5 read in the consumer (2026-09-19, experts4bit-qlora `k16-p5`): the small-M route saves 1.06 ms/step on the B=16 Qwen3-30B-A3B step on the 5090 (bf16 GEMM family −2.35, `_gemm_int4_b32_smallm` +1.30); **P5 holds** and experts4bit-qlora 0.36.2 makes the route its `auto` default. The claim row, `kernel/RESULTS-k16-smallm-int4-gemm.md` and `docs/STATUS.md` carry the read; no kernel code changes. P4 (single-scale control) remains untested.

## 0.32.0 — 2026-09-19 — K16: a Marlin-class small-M int4-b32 GEMM for the attention projections ships as `int4_smallm`; read on the RTX 5090 (P1–P3 hold, P4 untested, P5 pending); the consumer routes to it opt-in

### K16 — a small-M int4-b32 GEMM for the attention projections (experimental; the consumer routes to it OPT-IN only)

- `int4_smallm.gemm_int4_b32_smallm(x [M<=16, K], packed [N, K//2], scales [N, K//32]) -> [M, N] bf16`, one launch:
  in-register int4 dequantisation with the per-32-block scale applied inside the tile, bf16 tensor-core MMA over
  K chunks of 128–256 (four to eight scale blocks per dot instead of one), split-K across programs with the
  reduction fused into the same launch (the last-arriving program of a column block sums the fp32 partials in
  split order and stores bf16). Activations stay bf16 — the arithmetic is the dequant-then-GEMM path's without
  materialising the weight. `plan_smallm` legalises (BLOCK_N, KC, SK) for a shape and refuses what the format
  cannot express; `smallm_workspace` preallocates the partial buffer and the per-column counters so the launch
  is capture-legal. Under `TRITON_INTERPRET=1` the dot runs with fp32 operands (numpy has no bf16 dot); the
  compiled suite owns the bf16 numerics.
- Pre-registered as lane K16 (`kernel/PREREG-k16-smallm-int4-gemm.md`): the kernel K15 pointed at, with Marlin's
  6.37 / 8.28 µs on `q_proj` / `o_proj` at M=16 as the target (P1 ≤ 1.4× of it; P2 beats the bf16 dequant path;
  P5 ≥ 0.4 ms/step at B=16 once routed). Correctness suites: `kernel/test_int4_smallm_interp.py` (interpreter
  fp32-dot on CPU; the same file compiled on a GPU). A2000 pilot receipt `kernel/receipts-k16/a2000-pilot.*`
  (sm_86: 2.21× / 2.41× over bf16 on q/o_proj; not the registered class).
- **The 5090 lane read (2026-09-19, `kernel/RESULTS-k16-smallm-int4-gemm.md`, rows `kernel/receipts-k16/5090/`):**
  `q_proj` 6.35 µs (bf16 dequant path 10.37, K14 grouped GEMM 12.56), `k_proj` 4.37 (6.23), `v_proj` 4.49 (6.29),
  `o_proj` 6.39 (18.61, K14 20.73); launch floor 4.61 µs; relative error 0.003 against the bf16 reference. **P1 holds**
  (1.00× / 0.77× of K15's Marlin rows against a ≤ 1.4× bar), **P2 holds** (1.63× / 2.91× over the bf16 path), **P3
  holds** (k/v faster than bf16, launch-bound); **P4 NOT TESTED** (no single-scale control arm in the bench — an open
  item, not a pass); **P5 pending** the consumer route. Decision rule → the kernel-level claim is registered as
  `measured` (`gnf4.kernel.k16-smallm-int4-gemm.5090.2026-09-19`) and the consumer route is opened opt-in
  (experts4bit-qlora#578, `E4B_ATTN_INT4_SMALLM=1`). **Nothing in this package routes to it on its own.**

## 0.31.0 — 2026-09-18 — batched int4 decode plans split-K from the row count; two pre-registered kernel lanes reported (K14 refuted, K15 comparator)

**Batched decode through the int4-b32 expert GEMV gets a plan chosen for the
batch it is in.** `int4_b32._plan` sized split-K from `N` alone, so at large
row counts every expert projection ran a configuration chosen for a batch it
was not in and paid the partials reduce to do it; the plan now takes the
activation-row count and the SM count, as the NF4 sibling planner always has.
Affects batched decode (`R >= 16` activation rows, i.e. `B * top_k` through
experts4bit-qlora's collapsed decode forward) on NVIDIA parts with **64 SMs or
fewer** under Linux; B=1 decode, which calls this GEMV at `R = top_k` (4 or 8
for every shipped MoE), returns exactly the previous plan on every card, and
so does every part with more than 64 SMs (the sm_120 serving class included).
Prefill, training, the NF4 grouped GEMM and the MXFP4 GEMV are unchanged.
Upgrade if you serve batched int4 decode on an A2000/A5000-class card; no
action otherwise. A minor release because split-K changes the grouping of the
fp32 partial sums where it engages: within one box the plan is a pure function
of `(N, K, R)`, but anything asserting bitwise equality *across* boxes must pin
`sk` rather than plan it. No API removed; `_plan(N, K)` still means decode. The
consumer floor does not move — experts4bit-qlora's `[fast]` extra stays at
`grouped-nf4-gemm>=0.30.0`; its CI pin may move to this release's commit.

### The int4-b32 split-K planner takes the row count (#357, #358)

- `_plan(N, K, R=1, sm_count=128)`: for `R >= SPLITK_R_FLOOR` (16) on parts
  with `sm_count <= SPLITK_R_TERM_MAX_SMS` (64), split-K is sized so about
  `SPLITK_TARGET_BLOCKS_PER_SM` (8) blocks per SM are resident and collapses to
  `sk = 1` — no reduce launch at all — where the grid alone covers that. `sk`
  is non-increasing in `R` and never above the N-only value, so a `part`
  buffer preallocated from a plan at another `R` still fits.
- **Measured** (`gnf4.serve.int4-b32-splitk-row-term.a2000.2026-09-10`;
  `bench/int4/RESULTS-sk-r-sweep.md`, harness `bench/int4/sk_sweep.py`,
  receipts `bench/int4/rows/sk_*.json`): RTX A2000 (sm_86, 26 SMs), CUDA-graph
  replay, full cost including `reduce_partials`, 48 cells over the qwen3_moe /
  granitemoe / olmoe `gate_up` + `down` shapes at `R = 1..128`. The N-only rule
  costs 1.136× the per-cell optimum, the R-aware rule 1.011×, and it is never
  slower than the N-only rule on any cell; 1.102× at `R = 16` rising to 1.145×
  at `R = 128` summed over the six shapes, up to 1.305× on one shape
  (qwen3_moe `down` at `R = 128`, where the reduce is not launched).
- **The floor is the safety argument, not a tuning.** B=1 decode does not
  reach this GEMV at one row — `hot_residency`'s singleton branch calls it once
  per projection at `R = top_k` — so an R term without a floor would have moved
  the *licensed* serve configuration (olmoe `gate_up` 16 → 8, qwen3_5_moe 16 → 8
  and 8 → 4) as a side effect of a batch optimisation. `SPLITK_R_FLOOR = 16`
  clears every shipped `top_k`; `kernel/test_int4_b32.py` pins `R < 16` to a
  verbatim copy of the old rule across every row count below the floor and
  every SM count, re-derives the worst-cell bound (`SK_R_BOUND = 1.07`;
  measured 1.064×) from the receipts, fails if a model with `top_k >= 16` is
  added, and checks the monotonicity above.
- **Gated to the SM class it was measured on (#358).** P39 (box 1, RTX 5090,
  128 SMs; experts4bit-qlora#533) ran the R-aware plan against the N-only one
  at B=16 on an artifact-pinned pack and read NEW/OLD = 1.0064 / 1.0063
  (self-pair spread 0.0005): no step-level gain and a small real regression —
  on 128 SMs the rule collapses `sk` to 1 at `R = 128` where the old `sk = 16`
  was worth more than the reduce it saved. Parts above 64 SMs keep the N-only
  plan at every `R` until a sweep on that class sets its own target. The
  mechanism stands; the constant does not transfer across SM classes.
- **MXFP4 is measured and deliberately unchanged.** `gemv_mxfp4_b32` shares
  the planner and the grid, but on its e2m1 loop the int4 rule regressed two of
  32 cells by 15–17 % (`bench/int4/mx_sweep.py`, `rows/mx_*.json`): split-K
  keeps earning to `R >= 32..128` there. It keeps calling `_plan(N, K)` until
  an MXFP4-specific rule has its own receipt.

### K14, pre-registered and REFUTED: at M=16 no shipped int4 arm beats dequant-then-GEMM on the attention projections (#359, #360, #361)

- `gnf4.kernel.k14-smallm-int4-gemm-refuted.5090.2026-09-11`
  (`kernel/PREREG-k14-smallm-int4-gemm.md`, `kernel/RESULTS-k14-smallm-int4-gemm.md`,
  receipts `kernel/receipts-k14/`). RTX 5090, Qwen3-30B-A3B's four attention
  projections at M=16: the grouped int4 GEMM at the best of twelve swept block
  configurations is 1.12–2.00× *slower* than the bf16 dequant-then-GEMM path,
  the int4 GEMV 1.46–2.40× slower, so the registered mix saving over 48 layers
  is 0.000 ms/step against a 0.5 ms bar. `q_proj`'s bf16 path reads 16.78 MB in
  10.35 µs — 106 % of the measured 1528 GB/s streaming ceiling — and `k_proj` /
  `v_proj` are launch-bound at 1.74× the 3.57 µs floor.
- Amendment 1 swept the `gemm` arm's block configuration: at the shipped
  `bn64/w8` it ran 2.4–6.3× bf16; at its best (`bn16/w2`, `bn32/w4`) 1.12–2.00×.
  A single projection is one M-tile, so the expert path's parallelism (the
  expert count) is not there. Without the sweep the lane would have reported a
  bad default rather than the kernel.
- Two bounds the lane did not go looking for, arithmetic under a stated
  assumption rather than measurements: `o_proj` runs at 60 % of ceiling against
  `q_proj`'s 106 % on the same bytes (≈ 0.38 ms/step if matched, on the bf16
  side); unfused q/k/v spend three launches where one would do
  (≈ 0.44 ms/step at the ceiling). A roofline int4 projection kernel would be
  worth ≈ 1.29 ms/step; nothing shipped realises it — the kernel built for that
  arithmetic reaches 22 % of the bandwidth it would need. Cost $0.0653 against
  a $1.50 ceiling.

### K15, pre-registered: Marlin wins 1.6–2.0× at M=16 and the advantage is the kernel, not the format (#362, #363, #364)

- `gnf4.kernel.k15-marlin-comparator.5090.2026-09-11`
  (`kernel/PREREG-k15-marlin-comparator.md`, `kernel/RESULTS-k15-marlin-comparator.md`,
  receipts `kernel/receipts-k15/`). Same RTX 5090, M=16: vLLM 0.28.0's Marlin
  GPTQ kernel at g128 runs `q_proj` in 6.37 µs and `o_proj` in 8.28 µs where
  this package's bf16 dequant path takes 10.3 / 16.5 µs (1.62× / 1.99×) and its
  best int4 arm 12.5 / 18.9 µs. At matched bytes (g32) Marlin still wins
  (8.18 / 8.24 µs) — the kernel, not the weight format. Relative error
  0.0000–0.0005 against Marlin's own dequantised reference.
- **INCONCLUSIVE by the registered rule, informative anyway:** substituting
  Marlin where measured saves 0.583–0.812 ms/step, inside the 0.5–1.0 ms band
  the pre-registration fixed as inconclusive. experts4bit-qlora#564's 1.67 ms
  estimate is refuted as too optimistic — Marlin runs at 34–50 % of the
  streaming ceiling, not at it. Adoption is not established: engines as
  shipped across different torch/CUDA builds in different processes; using
  Marlin would mean a GPTQ repack, a dependency and a quality gate, none
  touched here.
- `k_proj` / `v_proj` are unmeasured: `MarlinWorkspace` was sized
  `N//64 × 16 = 128` for N=512, below the 170-SM count the kernel requires
  (fixed in #363, which also measures the `use_atomic_add` path); the lane ran
  out of budget before a box used the fix. Cost $1.1253 against a $1.00 ceiling
  — over by $0.13, and the lane is closed rather than extended.

### Census and repository hygiene (#356, #355, #345–#354)

- `census/shape_census.json` carries its fetch date and coverage boundary as
  data (`FETCHED`, `COVERAGE`), and `kernel/test_census_drift.py` fails if
  regenerating is not a no-op or the model set drifts (#356; the in-repo half
  of #353 — the cross-repository coverage check stays in the consumer's CI,
  which already clones this repository with `--sibling`). An unlisted `(N, K)`
  still takes `_decode_plan`'s universal constant on purpose: it is measured at
  median regret 1.000 on dense sweeps, so refusing it would be a regression.
- `private-marker-guard` self-tests every pattern it carries (#355).
  Repository-only: CI runs on `ready-to-merge` (#352); `AGENTS.md` §10 records
  how other agents work here, succession, and that nothing is filed upstream
  without the maintainer's say-so (#345–#354). None of this is in the wheel.

## 0.30.2 — 2026-09-05 — documentation and register hygiene (no API change; consumer floor unchanged)

**The claims register and the prose that quotes it are now held to their own
bookkeeping by CI.** A read-only audit of the 0.30.1 surfaces (2026-09-05)
found nothing wrong with a number, a gate or a verdict, and several things
wrong with the fields around them: receipt paths in `docs/claims.json` that
no file matched, a `measured_on` that was a month, a `retired` row without
its reason, a `quoted_in` pointing at a README section that no longer exists,
a claim sentence describing a predicate the code had since widened, and two
different "public import surface" lists for the same package. None of it was
visible to a check, because every check keyed on `status` and on version
strings. This release fixes each item and adds the two checks that would have
caught them. Affects readers of the register, the site that renders it, and
anyone adding a claim; nothing in the kernels, the wheel's contents or its
dependency floors changes. A documentation patch: no API change, no measured
number moves, and the consumer floor does not move — experts4bit-qlora's
`[fast]` extra stays at `grouped-nf4-gemm>=0.30.0`. No action needed.

### Register corrections (`docs/claims.json`)

- `gnf4.serve.m3-defaults-on`: the sentence said the fp8 path asserts
  `k_groups in (1,2,4)` — the predicate as it stood at the 2026-08-27 run.
  `fp8_paged_attn.fp8_compute_unsupported` has admitted `(1, 2, 4, 8, 16)`
  since 0.26.0 (Gemma-4's per-layer key-scale groups; `docs/capabilities.json`
  already said so); the sentence now describes the shipped predicate and the
  notes record the correction. The measurement (7.843 → 6.803 ms/step at
  Δppl −0.0021) is untouched. Its `receipts-m3/` evidence entry is now
  `kernel/receipts-m3/`, where the receipts are.
- `gnf4.kernel.sm120-census-vs-grouped-mm`: `measured_on` was `"2026-08"`, not
  a date. It is `2026-08-29`, the receipt's commit date (PR #299); the receipt
  states no run date and ran on `main` 4f1db333 (2026-08-28), so the run falls
  on 2026-08-28/29 — the notes say so.
- `gnf4.retired.splitk-gemv` carries `retired_reason` (the K7 round-2
  measurement the sentence already quoted).
- `gnf4.retired.sm120-parked`: `quoted_in` pointed at `README.md#Status /
  roadmap`, a section the README no longer has; it points at
  `README.md#What was retired` and `docs/STATUS.md#What changed`, where the
  line lives.
- Every `evidence[]` entry is a path that resolves at HEAD. The annotated
  strings became the structured forms `docs/claims-schema.md` now documents:
  `{"path": "CHANGELOG.md", "section": "<version>"}` for a changelog entry
  (`gnf4.serve.fp8-paged-attn-windows-sinks-scale`,
  `gnf4.kernel.expert-offset-boundary.5090.2026-09-05`, and
  `gnf4.kernel.dgrad`, whose A2000 cells are recorded in the 0.7.0 entry —
  no `kernel/RESULTS-*.md` carries them, which the old glob implied);
  `{"repository": "experts4bit-qlora", "path": …}` for the cross-repository
  dgrad-gate receipt; the explicit `kernel/test_int4_b32.py` where
  `gnf4.serve.decode-glue-kernels` said `kernel/test_*.py for each kernel`.
  `quoted_in` entries that named changelog versions use the
  `CHANGELOG.md#<version>` heading form.
- `measured_on` (ISO) added to every measured, measured-private and confirmed
  row that lacked one, taken from the receipt's own stated run date or, where
  the receipt states none, from the receipt's first commit — the row's notes
  say which. Bookkeeping only; no value changes.
- `gnf4.kernel.h2h-unsloth` notes: the model-level, training-axis end-to-end
  head-to-head is registered in experts4bit-qlora as
  `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05` (2026-09-05, one RTX 5090,
  Qwen3-30B-A3B: experts4bit-qlora 1.522 s/step vs Unsloth 2.151, ratio
  1.413; held-out loss comparable at N=60, lower for Unsloth at N=200, 0.2713
  vs 0.2881); a different level and regime, so the kernel-level claim is not
  superseded. `README.md` and `docs/STATUS.md` point at it beside "Unsloth
  wins its own regime".
- `gnf4.cold-engine.phase0-premise-refuted` notes state the receipt's own
  declared gap (its roofline pipe term is unmeasured) instead of the
  placeholder word "pending"; same fact.

### Contract surfaces

- `docs/capabilities.json` `project.import_names` equals the system
  manifest's `packages.kernels.import_names` (14 modules; it listed six and
  the two surfaces disagreed). Every one is a `py-modules` entry of the wheel;
  `int4_b32` imports triton at module level and so imports only where triton
  is installed (Linux), as its capability entry already says.
  `scripts/check_system_manifest.py` asserts the two lists are equal, the
  assertion the consumer's copy already made for its side. The manifest
  itself is unchanged (byte-identical in both repositories).
- `docs/claims-schema.md` documents the evidence and `quoted_in` forms, the
  `measured_on` rule and the bookkeeping fields the new check enforces.

### Checks (CI `discoverability` job; tests in the `interp-contract` job)

- `scripts/check_claims_register.py` (new): every `evidence[]` entry resolves
  at HEAD (bare path, `path`+`section`, `glob`, `repository`+`path`, or an
  issue URL under this repository); measured / measured-private / confirmed
  rows carry an ISO `measured_on` and a public (or private) receipt;
  `superseded` ⇒ `superseded_by` resolves to an active claim that names it
  back; `retired` ⇒ `retired_reason`; `quoted_in` entries resolve; no
  `pending` / `TBD` / `TODO` in the notes of an active row. `--sibling DIR`
  resolves the cross-repository paths. Tests:
  `kernel/test_check_claims_register.py`.
- `scripts/check_readme_claims.py` (new; ported from experts4bit-qlora to
  this repository's tables and wider document set): every backticked
  `gnf4.…` id in `README.md`, `docs/STATUS.md` and `docs/solutions/*.md`
  exists, a superseded or retired id appears only on a line that says so, and
  the results tables of `README.md` and `docs/STATUS.md` quote each named
  claim's current value at the document's precision with the weakest status
  in the row. Tests: `kernel/test_check_readme_claims.py`.
- `scripts/check_wheel_metadata.py` runs with explicit `--requires`
  assertions for the wheel's own floors (`numpy`, `torch>=2.8.0.dev0`,
  `triton>=3.4; platform_system == "Linux"`) beside the structural extras
  check; the consumer floor in the manifest's current record (`>=0.30.0`) is
  the consumer's metadata and is asserted in its CI, not in this wheel.
- Version 0.30.2 in `pyproject.toml`, the README's tag pins and the STATUS
  header; `llms-full.txt` regenerated.

## 0.30.1 — 2026-09-05

**Two kernel boundaries stop being run-time surprises.** Expert stacks and
fp8 KV pools past 2 GiB are addressed in int64 in every kernel that scales an
expert or block-table index by a stride, so the reader can no longer wrap
where the writer already did not; and a tile that will not fit the card's
shared memory is refused or re-dispatched before the launch with the numbers,
instead of surfacing Triton's `OutOfResources`. Affects anyone serving large
fused expert stacks or long paged contexts on NVIDIA GPUs under Linux, and
anyone on a small-LDS device (CDNA3 class) or calling the packed fp8
attention variant at Gemma-4's sliding geometry; nothing changes numerically
for stacks and pools under 2 GiB, and no measured number moves. Upgrade if
your expert stacks, KV pools or packed-attention geometries reach either
boundary; no action otherwise. A patch release: no API change, and the
consumer floor does not move — experts4bit-qlora's `[fast]` extra stays at
`grouped-nf4-gemm>=0.30.0`.

### Correctness — the 2^31 offset boundary (#87)

- The four `fp8_paged_attn` decode kernels (split and packed, f32 and fp8
  compute) widen the block-table row to int64 before the `k_row_bytes` /
  `v_row_bytes` product; the `fp8_kv` appenders already did, so a pool past
  2^31 bytes was written correctly and read from a wrapped offset.
- The M-tile kernels (`_gemm_nf4_grouped`, `_dgrad_nf4_grouped`,
  `_gemm_mxfp4_grouped`) widen the tile's `row0`, closing the sibling
  boundary at `T * max(K, N) >= 2^31` activation elements
  (`_gemm_int4_b32_grouped` already did).
- Every carrier of the expert-base pattern is now verified, not inspected:
  `kernel/test_expert_offset_boundary.py` samples the experts (or pool rows)
  whose base offsets sit just below and just above 2^31 for the NF4 decode
  GEMV in its scalar, split-K, wide-load and dot-pad forms, the NF4 M-tile
  and dgrad, the MXFP4 GEMM / GEMV / `gemv_mxfp4_b32`, the int4-b32 GEMV and
  M-tile, and the fp8 paged decode kernels, against the pure-torch references,
  each above-boundary case in its own subprocess. The expert-id cast the
  issue asked for shipped in 0.13.2 (NF4) and 0.14.0 (MXFP4); the split-K
  and dgrad kernels it flagged by inspection, and the int4-b32 and attention
  carriers it did not name, had never been exercised above the boundary.
- `docs/KERNEL_CONTRACT.md` carries the rule under "Boundaries".
- **#87 is closed by observation in every carrier.** On an RTX 5090
  (2026-09-05, torch 2.8.0+cu128 / triton 3.4.0) the boundary file's ten
  cases pass — five below-boundary controls in-process, five above-boundary
  cases in fresh processes — beside `test_offsets_2gib.py` +
  `test_int4_b32.py` (63 passed). Registered as
  `gnf4.kernel.expert-offset-boundary.5090.2026-09-05` (measured; the test
  and this entry are the public evidence, the GPU log is in the private
  receipt tree). The `docs/STATUS.md` line that called #87 "distinct from
  the 2 GiB stride fix in 0.13.2/0.14.0" was not supported by the issue
  text — the issue asked for the expert-id cast those releases shipped and
  flagged the remaining carriers by inspection — and is withdrawn; the
  position is the one above.

### Shared-memory feasibility before the launch (#324)

- `_triton_shim.UnsupportedShapeError` (a `ValueError` with `kernel`,
  `shape`, `need_bytes`, `limit_bytes`) and
  `_triton_shim.device_shared_mem_limit` (one query per device through
  Triton's driver; 0 when unqueryable, which never refuses).
- `fp8_paged_attn.packed_unsupported` decides pre-launch, from
  `packed_tile_smem_bytes` — calibrated to reproduce the 148 480 B Triton
  reported at head_dim 256 with 8 kv heads — whether the packed fp8 tile
  fits; where it does not, the split fp8 kernel serves the call with one
  `RuntimeWarning` per geometry, and the packed grid's `n_split` is no
  longer inherited by the fallback. The packed f32 kernel, which has no
  calibrated model, falls back the same way from the launch's own overflow
  (previously a raw `OutOfResources`); a split kernel that overflows raises
  `UnsupportedShapeError` with the geometry and the numbers.
- `nf4_grouped.prefill_fit` is the M-tile fit-down as a function: the same
  stages-then-`BLOCK_M`-then-stages descent the wrapper ran inline, now
  raising `UnsupportedShapeError` when even the smallest configuration
  overflows. No NVIDIA configuration moves.
- `kernel/test_shape_feasibility.py`: the selection rules under mocked
  limits on CPU, and the fit-down against `dequant_ref` under
  `TRITON_INTERPRET=1`.
- On the same 5090 run `test_shape_feasibility.py` passes compiled and under
  the interpreter (28 each) and the fp8 paged modes pass (47; the 37 f32
  modes stay deselected under #319), with the packed-tile fallback firing at
  head_dim 256 / 8 kv heads. No registered number moves, so #324 carries no
  claim entry of its own; it is noted on the #87 entry.

## 0.30.0 — 2026-09-04

### Top-k weighted combine fused

- `combine_rows(dn [T*k, H], w [T*k], k)`: the MoE combine (fp32
  weight-and-sum over the top-k slots, bf16 out) in one launch, for the
  consumer's collapsed forward whose torch chain was four dispatches and
  two kernels per layer (Qwen3 B=1 op census). Interpreter test at three
  shapes with a masked tail.

  (This section was filed under 0.29.0 in the changelog after that tag was cut; `combine_rows` ships in 0.30.0.)

### Agent discoverability layer (#336)

- README routing sections (Use this when / Do not use this when / Start here), `docs/SOLUTIONS.md` and six problem-first pages under `docs/solutions/`, `docs/capabilities.json` under a schema, `AGENTS.md`, `llms.txt` and the generated `llms-full.txt`, `docs/discovery-queries.json`, docstrings on eleven kernels and primitives, PyPI metadata (summary, keywords, MIT license expression, labelled project URLs) and a CPU-only CI job that validates all of it. No kernel behaviour changed.

### Documentation accuracy pass (#339)

- The bitsandbytes position is version-, workload- and shape-aware: upstream 5453368 ("[CUDA] New 4bit GEMM kernels for inference", 2026-05-21) is in 0.50.0 and not in 0.49.2, so recent bitsandbytes inference consumes packed 4-bit weights directly for supported ordinary 2-D matrices; routed grouped MoE remains a separate contract and the conventional backward still dequantizes for dX. Historical comparators keep their receipts.
- FP8 paged attention is split by compute path in `docs/capabilities.json`: the fp8 compute path (sm_89+, measured on sm_120) is supported; the f32 compute path (sm_80–88 default, explicit f32) is unsupported under #319 (`gnf4.open.f32-compute-modes-triton34`); decode glue is measured-private.
- The NF4 grouped-GEMM solution page carries the comparison table, the dataflow diagram and a measured-boundaries block quoting register claims by ID.

### Packaging

- `build/lib/` and the egg-info are no longer tracked, and every CI wheel proves its modules byte-identical to their sources (#338): a tracked build directory can ship stale copies when a fresh clone gives source and copy the same timestamp.
- Project-URL labels stay under PyPI's 32-character limit (enforced by `scripts/check_wheel_metadata.py`); Homepage, Documentation, Status and Solutions point at the cerinamroth.com routing pages (#337).

## 0.29.0 — 2026-09-04

### Split-K reduce + cast fused (decode GEMVs)

- `reduce_partials(part, sk, R, N)`: the fp32 split-K partials of
  `gemv_int4_b32` / `gemv_mxfp4_b32` are reduced and cast to bf16 in one
  launch instead of the `reshape().sum(0).to(bf16)` chain (three
  dispatches, two kernels per projection call). Qwen3-30B's B=1 op
  census put that chain at the top of the glue list (864 dispatches,
  ~0.6 ms of an eager step); the graph census had the reduce class at
  13% of the 5.56 ms step. Same values to bf16 rounding; interpreter test
  with a masked tail.

## 0.28.0 — 2026-09-04

### Decode-grade MXFP4 expert GEMV

- `gemv_mxfp4_b32(xq, xs, blocks, scales, eids, N, K)`: the int4-b32 decode
  GEMV structure (split-K over 32-wide groups, KU groups per iteration,
  fp32 partials, the same sm_120 plan) on the NATIVE MXFP4 store
  (`[E, N, K//2]` e2m1 nibbles, low nibble first; `[E, N, K//32]` e8m0).
  The block dot is an exact int32 sum: an e2m1 nibble decodes branchlessly
  to twice its value as an integer ({0,1,2,3,4,6,8,12}) and the 0.5 folds
  into the scale. gpt-oss's B=1 lever: its NF4 expert GEMV is 3.50 ms of a
  7.58 ms step and the v1 MXFP4 GEMV (one program per 64 rows, no split-K)
  was slower than NF4; this one has the kernel that runs ~1.1 TB/s on int4.
  Gates: e2m1 decode table (all 16 nibbles), interpreter parity against
  `mxfp4_pack_ref` on quantised activation rows, GPU gate at K=2880.

### Rotary-only fold for attention without a head norm

- `rope_heads(x [R, HEADS, D], cos, sin)`: the rotate-half rotary for all
  heads of one projection in one launch, without the per-head RMSNorm —
  GraniteMoe and Mixtral attention has no q/k norm, so the norm+rotary
  fold could not license them. Same fp32 chain and single rounding as
  `rope_norm_heads`; tested against the upstream chain at three head
  geometries under the interpreter contract.

## 0.27.0 — 2026-09-04

Two kernel additions for experts4bit-qlora's throughput-parity build-out (P30): every measured refusal on a non-Qwen family became a change. Consumers: e4b#370 (router kinds) and e4b#371 (GraniteMoe-shaped layer fold); the lane numbers that gate those are on their PRs.

### Residual fold takes a multiplier; one-launch scaled residual add

- `rmsnorm_resid_rows(x, resid, weight, eps, scale=1.0)`: the GraniteMoe
  body is `resid + x * multiplier`; the fold now takes that multiplier
  and reproduces upstream's two bf16 roundings (product, then sum). At
  the default the path is unchanged and bitwise.
- `scaled_resid_add_rows(x, resid, scale)`: the layer tail's
  `resid + x * scale` as one launch with the same two roundings (a
  single-rounding `torch.add(alpha=)` is not the upstream value).
- Tests: the residual-fold test runs at scale 1.0 and 0.22; the scaled
  add is checked bitwise on hardware and within two ULP under the
  interpreter, and the test asserts its reference differs from the
  single-rounding add so the rounding claim is actually tested.
### Router epilogue: select-on-logits mode with an optional bias

- `router_epilogue(logits, k, norm, *, select_on_logits=False, bias=None)`:
  the existing kind is softmax over all experts, then top-k, optional
  renormalisation (Qwen3-MoE, OLMoE, Mixtral). The new mode takes the
  top-k on the LOGITS (plus a per-expert `bias` when given, gpt-oss's
  router; GraniteMoe's has none) and the softmax over the selected k —
  a different function, not a reordering, when the k are not
  renormalised. The first output in that mode is the (biased) logits.
- The kernel carries `HAS_BIAS` / `SELECT_ON_LOGITS` as compile-time
  flags; the default path is unchanged and bit-identical.
- Tests: both modes against a torch reference at several (E, k),
  with and without a bias, under the interpreter contract.

## 0.26.0 — 2026-09-04

### fp8 paged attention: key scale groups up to 16; packed variant falls back instead of failing

- Both fp8 compute kernels (split and packed) now unroll **8 and 16 key
  scale groups** beside 1/2/4, so a 512-dim head keeps 32-wide key scales
  (`k_groups=16`) instead of 128-wide. Measured on Gemma-4-26B-A4B-it's
  five 512-dim layers through experts4bit-qlora's paged decode: the fp8
  cache+dot cost falls from 0.046 to 0.017 nats in the fake-quant model
  and the served path moves −0.020 nats on the same window. Synthetic
  precision is flat across group counts (the gain is real activations'
  outlier channels). `fp8_compute_unsupported` admits `(1, 2, 4, 8, 16)`
  with the `head_dim // k_groups >= 32` rule unchanged.
- The packed variant's `OutOfResources` at head_dim 256 with 8 kv heads
  (148 KB of shared memory against a 101 KB card) is caught before
  anything is written; the split fp8 kernel serves the call, the
  geometry is remembered so later calls skip the packed launch, a
  `RuntimeWarning` fires once, and `compute_counts()` tallies the call
  once (#324).
- Tests: every fp8 mode at head_dim 128/256/512 × 4/8/16 groups against
  the fp32 oracle on identical bytes; the fallback; the refusal moves to
  a count outside the unroll. Validated on a rented RTX 5090 (47 passed
  at the fp8 modes).

## 0.25.0 — 2026-09-03

### Documentation release: the README says what is measured, with each number's evidence tier

No kernel changes. This release exists so that what PyPI renders matches
the repository after the docs audit: the README is distilled from 768
lines to one page — what the kernel is and what ships around it, the
see-it-yourself script, install, the entry-point table, one table of
what is measured with each row's evidence tier, the three limits stated
up front (a CUDA-graphed baseline wins at decode; Unsloth wins its own
bf16-resident regime; the known loser classes), where the receipts are,
what was retired, what is open.

- `docs/claims.json` — a machine-readable register of 29 claims, each
  with value, unit, hardware, tier and evidence path
  (`docs/claims-schema.md`). Every number in the README's measured
  section maps to an entry, checked mechanically.
- **`measured-private`** is a tier, not a footnote: the int4-b32 GEMV
  cells, the calibrated-pack quality numbers and the decode-glue
  composition come from a private audit tree and are labelled so.
- `docs/STATUS.md` — one page: what the kernel does today, what was
  retired, what is open, how to read the numbers.
- `docs/INDEX.md` — what each of the 22 documents is for and whether
  it is current; names the gap that the serving-side kernels of
  0.14–0.24 have no reference page under `docs/` yet.
- **Retired from the README:** "Parked: sm_120 (three consecutive
  cloud provisioning failures)". True in July, false since 0.15.0 —
  the RTX 5090 has been the primary serving target for every release
  since, and the same README carried an sm_120 census.

## 0.24.0 — 2026-09-03

### Paged decode: sliding windows, attention sinks, custom scale

One change (#318). `fp8_paged_decode_attention` gains `window` (keys
older than the last `window` tokens are skipped tile-wise and masked
within the boundary tile; `0` keeps full attention), `sinks` (a
per-head `[H]` logit that joins the softmax denominator without a
value, the gpt-oss `s_aux` convention; `None` keeps the plain softmax)
and honours the `sm_scale` already accepted — Granite's
`attention_multiplier` and Gemma-4's `1.0` reach every compute mode.
All four decode kernels take the window and sink pointer as runtime
arguments, so an engine serving a model that mixes sliding and full
layers (Gemma-4: 25 of 30 layers at 1024) launches the same compiled
kernel per layer. `k_row_bytes` / `v_row_bytes` let a pool whose stride
is wider than a layer's natural row (per-layer KV geometry, one pool)
address it correctly; a stride narrower than the row is refused.
`paged_attn_ref` takes the same three options.

Tests: `test_sliding_window_keeps_only_the_last_keys` (windows 1, 5,
16, 33, 64 against a reference that drops the keys outright),
`test_window_wider_than_context_is_full_attention` (bitwise equal to
`window=0`), `test_attention_sinks_join_the_denominator_only`,
`test_window_and_sinks_together`, `test_custom_scale_reaches_the_kernel`
(the scaled-score distribution is held fixed while the scale varies:
the fp8 path's query rounding error is proportional to the scaled
score magnitude). 35/35 in the fp8 compute modes on an RTX 5090
(torch 2.8.0+cu128, triton 3.4.0). The f32 compute modes miss the
reference on that torch/triton pair on unmodified `main` as well —
tracked as #319, unrelated to this release.

Bytes on the wire and the kernels' arithmetic for `window=0`,
`sinks=None` are unchanged; a model served before this release scores
identically after it.

## 0.23.0 — 2026-09-03

### HessianAccumulator stores off-device

One change (#316). Calibrating Mixtral-8x7B's attention — 128
projections at K=4096 — held 8 GB of fp32 Hessians on a 32 GB card
beside a 23 GB model and ran out of memory. `HessianAccumulator` now
computes each batch's Gram where the activations are and accumulates
into a Hessian that lives on the requested `device` (`"cpu"` for a
whole model), moving only the K×K result per batch; allocation is
deferred to the first batch so the default storage follows the
activations. Closed-form test over unequal batch sizes. Validated on
Mixtral on a 5090: the calibration completes (the pack itself was then
refused by experts4bit-qlora's quality gate on that model, +0.09 ppl —
calibrated int4 attention is a win on Qwen3-30B-A3B, not everywhere).

## 0.22.0 — 2026-09-03

### Calibrated int4 packing on the int4-b32 grid

One packer (#313). `gptq_pack_int4_b32(w, hessian)` produces exactly
the bytes and the fp16 per-32 scales `pack_int4_b32` produces — the
same grid, the same GEMV, the same store — and differs only in *which*
grid point each weight lands on. It quantises column by column against
`H = 2·XXᵀ` accumulated over the model's own activations
(`HessianAccumulator`), pushing each column's rounding residual into
the columns that follow through the Cholesky factor of `H⁻¹`, with the
per-block scale chosen on the *compensated* weights (a scale taken from
the source weights and a round-trip through the plain packer both
measured worse than rounding). Dead input channels are pinned; 1 %
damping.

Why it exists: round-to-nearest int4 on Qwen3-30B-A3B's attention
projections failed the perplexity gate at +0.056, and fp8 e4m3 —
carrying 4.6× lower *weight* error — bought almost none of that back.
Weight error is not what the gate measures. Packed with calibration on
a C4 validation shard, the same bytes score **−0.042** on the wikitext
gate and **−0.115** on an out-of-domain C4 text (both improvements over
bf16 attention, same sign; `experts4bit-qlora` 0.28.0 carries the
serving lane and the one-sided gate for calibrated packs). Buffers
follow the weight's device (a CPU scale meeting a CUDA column was
caught in review). CPU tests: three activation regimes against
rounding, byte-identical format, identity Hessian ≈ rounding, dead
channels.

Not in this release: a row-count-aware GEMV plan (#314) that won on a
uniform-routing microbench and lost 11 % in the real serving path —
kept as a draft with the measurement.

## 0.21.0 — 2026-09-02

### Glue round 3: the router epilogue in one launch

One kernel (#311). After the router's GEMM, torch runs a softmax over
every expert, a top-k it serves with a gather plus a bitonic sort, a
sum and a divide — five launches per layer, which the shipped-stack
census put at ~0.43 ms of a 5.66 ms single-stream step.
`router_epilogue` does all of it in one: fp32 softmax, iterated-max
top-k with ties to the lower expert index, optional renormalisation,
returning probs, weights and indices with upstream's dtypes. `top_k`
need not be a power of two — the register vectors are padded and
masked, since `tl.arange` spans only powers of two and a model whose
top_k is 6 would otherwise fail at launch.

**Why this is licensed where a fused router was refused in 0.12.x:**
that fusion folded the router GEMM, so a single CTA pulled the whole
router weight matrix and lost on occupancy. This folds only the
epilogue, where a program reads E floats — 512 bytes at E=128 — so
what is left to win is launch count. Measured through the serving
package (its #329): **1.0735x at B=1** (5.657 → 5.270 ms) and
**1.0464x at B=16** (13.823 → 13.210 ms, 1,157.5 → 1,211.2 tok/s
aggregate), with the paired quality gate PASSING at **-0.01968 ppl** —
an improvement, since the fused chain softmaxes and renormalises in a
single fp32 pass.

Tests pin the selected expert set exactly, because a different set is
a routing change rather than a rounding one.

## 0.20.0 — 2026-09-01

### Glue round 2: the residual add and the rotary chain fold away

Two decode kernels (#308), both aimed by the shipped-stack censuses,
which put the remaining non-GEMM step at ~1.9 ms of the 6.5 ms
single-stream step and ~4.0 ms of the 15.5 ms batched one:

- **`rmsnorm_resid_rows`** folds a decoder layer's residual add into
  the following RMSNorm. The add rounds once to bf16 exactly as the
  bf16 `+` it replaces — bitwise on hardware, asserted as such — and
  the norm then reads that rounded value.
- **`rope_norm_heads`** folds a projection's per-head RMSNorm together
  with the rotate-half rotary chain (slice, negate, concatenate, two
  multiplies, an add) into one launch, keeping upstream's ordering and
  its pre-rotary bf16 rounding.

Both are row/head-parallel with per-program pulls of ~2-4 KB, the same
occupancy frame that licensed the fused norm in 0.19.0 and refused the
single-CTA router. Measured through the serving package's consumer
(its #326): **1.1557x at B=1** (6.575 -> 5.689 ms) and **1.0916x at
B=16** (15.305 -> 14.021 ms, 1044.7 -> 1141.1 tok/s aggregate), with
the paired quality gate PASSING at delta -0.00105 ppl over 8192
sha-matched steps.

## 0.19.0 — 2026-09-01

### A fused row-parallel RMSNorm for the decode glue

One kernel (#306). `rmsnorm_rows` folds the T=1 RMSNorm — fp32
mean-square reduce, rsqrt, weight multiply, bf16 cast — into a single
row-parallel launch. The fusion is licensed by the occupancy rule the
router kernel failed: the per-row data pull is ~4 KB, so one program
per row costs nothing in occupancy, where the refused single-CTA
fusions pulled hundreds of KB through one SM. Consumed by the serving
package's opt-in decode glue (its #324); on the composed B=1 lane the
glue round measured 8.344 → 6.469 ms per step on the rental class with
a paired quality gate PASS (Δppl +0.0136 @ 8192 steps).

## 0.18.0 — 2026-08-31

### The uniform int4-b32 serve lane, from decode GEMV to the batched step

Everything the B=16 serving campaign shipped to main since 0.17.0, all
graph-metric measured with receipts in the audit tree (INT4B16/*):

- **int4-b32 pack + decode GEMV** (`int4_b32`, `int4_pack_ref`):
  uniform symmetric grid, arithmetic unpack, int8 activations, exact
  integer accumulation. Census cells: 1,044 GB/s dense M=1, 6.9–7.2×
  over the NF4 register-LUT GEMV; grouped top-8 cells 3.8–4.3×.
- **Grouped M-tile int4 GEMM** against the captured tile contract
  (K=32 int8 MMA, `tl.interleave` nibble planes, zero-tile exit):
  gate_up 1.947× / down 4.500× over the NF4 grouped kernel at the
  B=16 census cells; defaults are the swept winner (bn64/w8).
- **Batched-slot fp8 KV append** (`fp8_kv_append_bt1`): one launch per
  side replaces the per-slot loop (12,288 → 768 calls per B=16 step),
  byte-identical to the T=1 kernel by CI gate.
- **Tail fusion**: one-launch tile table (stable counting sort,
  decode-shape contract, exact against the chained builder),
  gather-folded activation quantise, fused SwiGLU.
- **Quantise grid**: per-(row, block) launch, ~3.6× per call.

Composed on the consumer's B=16 serving step these land 39.5 → 21.8 ms
(405 → 734 tok/s aggregate). Measured refusals kept with receipts: the
format stays OFF the lm_head (+0.18 ppl) and dense attention shapes
lose to the bf16 baseline in both the GEMV and M=16 GEMM regimes
(occupancy, mechanism isolated).


## 0.17.0 — 2026-08-27

### Both decode knobs now ship ON by default (M3, PASS)

0.16.0 shipped `GNF4_ATTN_COMPUTE=fp8` OFF and said a default flip
"needs its own registration plus a longer-horizon quality window than
1024 tokens". M3 is that registration and that window.

**8192 teacher-forced tokens through the paged decode path**, four
arms on one box, one provisioning
(`kernel/RESULTS-m3-default-on.md`, receipts in `receipts-m3/`):

| arm | step | Δppl vs OFF |
|---|---|---|
| off | 7.843 ms | — |
| `GNF4_GEMV_DOTPAD` | 7.032 ms | −0.0133 |
| `GNF4_ATTN_COMPUTE=fp8` | 7.620 ms | −0.0058 |
| both | **6.803 ms** | −0.0021 |

Bar was ±0.05 on each knob AND on the composition. Read the deltas as
zero, not as gains — all four arms sit within 0.02 on a perplexity of
8.05, and bf16/e4m3 rounding is unbiased.

**K8's +0.0092 was window noise.** Over 8192 tokens the fp8 delta is
−0.0058; the sign flipped. The cost does not grow with the horizon,
which is the question 0.16.0 left open.

### The flip is capability-conditional, and that is not cosmetic

PASS licensed the defaults on **quality and speed**, not on
**applicability**. The fp8 path asserts `sm_89+`, `v_groups == 1`,
`k_groups in (1, 2, 4)`, and — on the packed-heads branch —
`block_tokens * n_kv_heads >= 32`. M3 varied none of them.

An unconditional flip would turn a working f32 install into an
`AssertionError` on **every pre-Ada GPU** (A100 sm_80, 3090 sm_86,
T4 sm_75). So:

- **Unset env** → fp8 where fp8 can run, the certified f32 path
  otherwise. Silent, and no install that worked before breaks.
- **Explicit `GNF4_ATTN_COMPUTE=fp8`** → never downgraded. You get
  the path you named, and its asserts tell you if it is unavailable.
  Silently substituting f32 under the name the caller asked for is
  how a benchmark arm gets mislabelled.
- **`GNF4_ATTN_COMPUTE=f32`** → the certified path, as before.
- **`GNF4_GEMV_DOTPAD=0`** → forces the scalar GEMV. Dot-pad needs no
  capability guard: its dispatch already requires the shape to be in
  `_DOTPAD_CONFIGS` and the part to carry ≥ 160 SMs, so a
  non-qualifying call takes the scalar path on its own.
- Both env vars now **REFUSE an unrecognised value** rather than
  treating it as off. With the defaults ON, a typo'd `GNF4_GEMV_DOTPAD=true`
  would otherwise read as a deliberate disable, which looks intentional.

One predicate, `fp8_compute_unsupported()`, backs both the default
selection and the asserts on both fp8 branches — two copies is how a
default starts choosing a path its own asserts reject.

### New: dispatch receipts

`nf4_grouped.dispatch_counts()` and `fp8_paged_attn.compute_counts()`
record which kernel actually ran, because an env var is a request:
`GNF4_GEMV_DOTPAD=1` on a part below the SM guard silently takes the
scalar path. M3's arms used these to prove each knob engaged.

**Scope:** one box, one model (Qwen3-30B-A3B), one config. M2
measured 8.5% inter-box dispersion, so the absolute numbers are that
box's; every bar above is a same-box comparison.

## 0.16.0 — 2026-08-26

Minor release: the fastest certified single-stream point this project
has measured, and an honest restatement of the slowest one.

### `GNF4_ATTN_COMPUTE=fp8` — 159.2 tok/s single-stream (K8, PASS)

`fp8_paged_decode_attention`'s fp8-COMPUTE path was built and sm_89+
gated for two releases with its quality debt stated in its own
docstring. K8 called it in: **0.217 ms off the step at a perplexity
cost of +0.0092** (bar: +0.05), measured through the paged DECODE
path over 1024 teacher-forced tokens, with the cited fp8 error bound
re-measured on the same box (mean 0.00445, p99 0.01953, max 0.10547).

- Ladder, composing on the dot-pad knob: 6.476 ms → **6.281 ms**,
  154.4 → **159.2 tok/s**.
- **Ships OFF.** An unset env is byte-identical to the certified
  path; an unrecognised value REFUSES rather than silently running
  f32 and being recorded as the fp8 arm. A default flip needs its own
  registration plus a longer-horizon quality window than 1024 tokens.
- Disclosure: all 127 greedy tokens matched between arms. That is
  reported, not relied on — the prereg deliberately refused an
  identity bar because this path's p99 element error (~5e-2) makes
  one unsatisfiable by construction, and the verdict rests on the
  perplexity delta.

### The default ladder entry is now a RANGE (M2)

Re-certifying the rental anchor on three boxes with A/A pairs found
the constant itself was **right to 0.26%** (7.369 vs 7.35) — and that
the 5090 "class" carries **8.5% inter-box dispersion** while each box
repeats itself to 0.16%. A ±3% gate was narrower than the population
it gated.

- Certified default restated: **7.37 ms ±4.2% ≈ 130–142 tok/s**
  (previously a bare "≈136"). The two knob rows are same-box
  measurements and are unchanged.
- `kernel/decode_anchor.py` is the committed source harnesses read,
  with its bounds document-locked against RESULTS-m2. An uncertified
  literal had been gating box rentals.

### `GNF4_GEMV_SPLITK` ships OFF, refuted (K7)

Split-K on the decode GEMV is **refuted**: the census pair is flat in
split factor at `gate_up`, and on `down` every split is ~14% worse
than not splitting. The kernel ships dormant because it is the
evidence for that refutation and costs nothing unused. The 9.8% the
cycle did produce is a config retune, not the registered mechanism.

### Also

- K10 attributed the census `router` row by ablation: one
  `torch.topk(k=8, sorted=True)` per layer. `sorted=False` deletes
  the sort kernel outright but changes the selected expert SETS, so
  it was refused despite a passing perplexity delta.
- No API changes. No default changes.

## 0.15.1 — 2026-08-25

Patch release, one fix.

### `fp8_kv_append_t1` refuses cleanly on non-CUDA tensors

The availability guard checked triton importability only — but triton
installs on CPU-only Linux hosts, where a launch dies inside triton's
driver with `0 active drivers`: an error naming neither the function
nor the fix. Surfaced within the hour of 0.15.0 reaching PyPI, by
experts4bit-qlora's CI (its presence-keyed fused-append resolution
began selecting the fused path on CPU-device KVs; e4b 0.21.0 fixes
its resolution to key on device — this is the kernel keeping its own
promise). The guard now requires CUDA-resident tensors, with a
CPU-runnable refusal test.

## 0.15.0 — 2026-08-25

Minor release: the M=1 decode-kernel campaign (K1–K6-B), the graph-step
tail work (F1/F2), and the capture-safe grouping API. Single-stream
decode on the reference class (Qwen3-30B-A3B, RTX 5090) moves from
~66 tok/s at 0.14.0 defaults to **~139–140 tok/s** at 0.15.0 defaults
with the e4b 0.21.0 harness, ~152 with the opt-in dot-pad knob. Every
default flip below cites an adjudicated verdict with committed
receipts.

### Performance (defaults changed)

- **M=1 decode launch configs re-tuned for sm_120** (K1, #241/#242):
  the baked winners are worth ~13% single-stream on the class box
  (65.8 → 74.3 tok/s at the K1 rung).
- **Fused one-launch T=1 paged KV append** (`fp8_kv`, #253) — the
  gnf4 half of e4b's F1-B2 default (94.2 → 133.4 tok/s rung there):
  one launch replaces the per-layer append stack, CUDA-graph-capturable.
- **f32 paged-decode attention now fuses its combine in-kernel by
  default** (RESULTS-f2-tail, #260/#261): the last-arriving CTA per
  (seq, kv-head) reduces the split partials in the same fixed order as
  the standalone kernel — bitwise-identical, token-identical over the
  127-step receipt. Cut 0.041 ms/step alone, 0.197 ms combined with
  e4b's fused QKV. Rollback: `GNF4_F32_FUSE_COMBINE=0`.

### Performance (opt-in)

- **Dot-pad GEMV** behind `GNF4_GEMV_DOTPAD=1` (K6-B, #258/#259):
  tensor-core dot with the x-vector in M-row 0 for the two flagship
  M=1 shapes; ~152 tok/s on the class box. PARTIAL verdict (two
  boxes), so **default OFF** — the 15/16 M-row waste is registered
  honestly and the knob ships disclosed.

### New API

- **Capture-safe grouped execution** (#255):
  `build_group_tiles_device` (argsort/scatter/cumsum tile construction
  with a static ceil(R/BM)+E budget, zero-row padding no-ops) and
  `gemm_4bit_grouped_captured` — T>1 expert grouping with no `.item()`,
  no host-size dependency, legal inside CUDA graph capture. 23 CPU
  tests plus source guards against host-sync ops.

### Measurement and research artifacts (no runtime behavior change)

The K-series record under `kernel/`: K2 vectorized nibbles
REFUTED-FOR-VARIANT (#243), K3/K4 streaming-floor and wide-loads
refutations (#244–#247), K5 M-tile probe STRUCTURE-REFUTED with the
graph-replay timing basis amendment (#248–#252), K6 bespoke-GEMV
frame amendments (#254/#256/#257), the low-G split revert (#236–#240),
and the F2 prereg's in-flight bar re-derivation (fused-QKV bitwise
claim falsified by its own CPU gate before registration). Receipts for
every verdict live beside their RESULTS files.

### Packaging

- `f2_verdict` classified as a campaign instrument (not shipped);
  the packaging meta-guard enforces the classification.

## 0.14.0 — 2026-08-23

Minor release. 0.13.2 shipped before a large body of hybrid-tier and residency
runtime work; this cuts all of it, plus the MXFP4 half of the 2 GiB offset fix.

### Correctness

**MXFP4 expert stacks past 2 GiB no longer fault — the port had dropped the cast
([#205](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/205)).**
0.13.2 fixed exactly this bug in the four NF4 kernels: `eid * stride_be` is
signed-int32, so a packed stack over 2^31 bytes wraps to a negative offset and
faults. The MXFP4 port of those kernels did not carry the `eid.to(tl.int64)`
promotion across. Both `_gemm_mxfp4_grouped` and `_gemv_mxfp4_grouped` now
promote `eid` to int64 before any stride product, mirroring `nf4_grouped`.

Found live, not by inspection: a P2-G1 run died with
`cudaErrorIllegalAddress` at transient-pool slot 244 (stride 8,812,800 bytes →
2.20e9, past 2^31). The regression test lives in `kernel/test_offsets_2gib.py`
alongside NF4's — deliberately **not** in `test_mxfp4_interp.py`, because
`TRITON_INTERPRET=1` evaluates offsets with int64 semantics and the overflow
cannot manifest there, so a test in that file would have validated nothing.

Also corrected: the engine accepts a gpt-oss arena's own shape rather than the
bake rewriting it (#154); ColdTier `ensure` is overlap-safe under concurrent
callers (#102); three unread Bugbot findings (#130); and a Stage-3 verdict
correction that #177 lost in transit (#179).

### New capabilities

- **Hybrid CPU tier** — `gnf4_native` now ships as a package (compile-at-
  first-use AVX-512 kernels, C source as package data) with `cpu_grouped` as
  its torch-facing wrapper: grouped GEMV and dgrad over packed NF4/MXFP4 bytes,
  a persistent pinned pool, and a fused expert FFN.
- **`RowPool`** — the weight-tier abstraction generalized to writable rows.
- **FP8 KV cache and paged decode attention** — `fp8_kv` (quantize/pack/unpack)
  and `fp8_paged_attn` (`fp8_paged_decode_attention`, plus a reference arm).
- **CPU destination for cold experts** — `cold_cpu_view.ColdCpuView`, with
  `cold_deadline.choose` supplying the time-to-contribution cost model the
  destination rule consumes.
- **Reclaimable VRAM residency** — `vram_slots.VramSlots`.
- **Observed-reuse classification** — `reuse_profile.ReuseProfile`.
- **Expert-keyed device row cache** — `dev_row_cache.DevRowCache`.
- **`preadv` direct scatter** — cold segments land straight into the
  kernel-shaped stacks; `attach_landing` closes the tier↔consumer loop.

### Performance

Measured on the paths shipped here:

- ColdTier demote is a lazy heap rather than a full scan or sort — the sequence
  #157/#158/#159/#175/#176/#182 closes **76–82% of the soft-hard gap**, after
  `_demote_locked` was measured at 93% of it.
- The `preadv` direct-scatter fill path measured **~43% faster** and materially
  steadier than the staged path.
- Native AVX-512 grouped GEMV reached **74.8% / 82.1%** of achievable bandwidth
  at flagship shapes on bare metal (gate G2), with exactness passing everywhere.

### Packaging

- **`numpy` is now a declared dependency.** It was already load-bearing in
  `nvme_reader` and became so in `cpu_grouped`; undeclared, a clean venv broke
  at import.
- `gnf4_native` is a real shipped package (`packages = ["gnf4_native"]`) with
  `*.c` as package data — previously nothing outside `kernel/` shipped.
- Nine modules added to the `py-modules` allowlist: `row_pool`, `fp8_kv`,
  `fp8_paged_attn`, `cpu_grouped`, `cold_cpu_view`, `cold_deadline`,
  `reuse_profile`, `vram_slots`, `dev_row_cache`.

### Measurement and research artifacts (no runtime behavior change)

The bulk of the diff since 0.13.2 is the Stage-3 / elastic-execution research
record under `bench/` — preregistrations, gate receipts, and results. Most of
its headline findings are **refutations**, and none of them change runtime
behavior in this release:

- Promotion mechanics do not pay as dispatched (P2-G1, #206); rank predicts
  recurrence but cannot be spent (#200); which row you evict barely matters
  (#198); ARC does not rescue this (#197); top-k does not explain when frequency
  beats recency (#192); the deadline rule loses to its own baseline (gate 2,
  #125).
- The cache is LRU and LRU is ~1.9x off optimal — where the remaining wall is
  (#185).
- Gate 1 re-measured: storage is ~2% of cold cost; the earlier 20% used the
  box's random ceiling, not its sequential one (#136/#137).
- Promotion and the CPU tier share one DRAM budget (P2-G1c, #210); the
  partitioned persistent pool — not retention — is what fails (P2-G2, #213).
- SPEC amendment for the elastic execution controller (#203, #207, #211) and
  the P2-G1b/G1c/G2 preregistrations (#208, #209, #212) are specification and
  harness text, not shipped runtime.

## 0.13.2 — 2026-08-15

### Expert stacks past 2 GiB no longer fault — int64 offset arithmetic

`gemm_4bit_grouped` (and `dgrad_4bit_grouped`) raised an illegal memory access
whenever the packed stack `B` exceeded 2^31 bytes: `eid * stride_be` was
signed-int32, so a stack of exactly 2 GiB was the last one that worked. The
boundary was measured exactly on two shapes (256 × 8 MiB passes, 257 faults;
128 × 16 MiB predicted from the stride math and hit) — it hard-capped the
batch/arena path at DeepSeek-class expert counts.

`eid` is promoted to int64 at load in all four kernels, so every downstream
stride product promotes. Verified: the new boundary test fails on 0.13.1's
kernels and passes on these (experts sampled both sides of 2^31, plus one
grouped call touching both sides in a single launch, against `dequant_ref`);
**26/26 tensors bitwise identical below the boundary**; capture ladder 6/6
with the arena guard's named refusal intact.

## 0.13.1 — 2026-08-15

### Index transfers are capture-conditional (the §11 repair)

0.13.0's pinned-arena index transfers were unconditional. Measured on whole
machines (instrument self-pair clean to ~2%): the arena path cost the
host-bound e2e step **−1.8% median** — a registered gate failure
([`RESULTS-capturability.md` §11](bench/phase1/results/dequant_forward/RESULTS-capturability.md))
— because outside a capture the syncs it removes cost nothing while its
per-call host work does.

`to_device_i32` now takes the arena path **only while the current stream is
capturing** and performs the pre-change pageable build otherwise. The arena is
touched on every CUDA call so it exists before any capture; a capture larger
than the arena refuses by name (`GNF4_PIN_ARENA_INTS`).

Verified under a pre-stamped registration
([`kernel/prereg_capture_conditional_repair.json`](kernel/prereg_capture_conditional_repair.json)):
capture 6/6 with the named refusal; **26/26 tensors bitwise identical** to
0.13.0 and all `expert_ids` forms equal; the e2e gate **PASSED** on a
whole-machine A4000 (cap/pub1 median 1.0040, inside the instrument's own
spread — parity with 0.12.0 by construction and now by measurement).

**Scope change, stated plainly:** the +6.5–15% *uncaptured* kernel-bound win
measured for 0.13.0 (§§9–10) is forfeited — uncaptured, 0.13.1 ≡ 0.12.0. That
result is re-scoped to **captured execution**, where the arena path still
runs. There is no knob; capturing is the switch.

## 0.13.0 — 2026-08-14

### The fused training path can now be CUDA-graphed — five hazards, three never named

The dequant-on-forward baseline captures into a CUDA graph cleanly; gnf4's fused
training path failed 8/8. That is a structural asymmetry against us, because
graphing is how the host floor at small batch would be removed.

CUDA's error (`operation failed due to a previous error during capture`) is what
a **later** call reports after an **earlier** illegal one already killed the
capture, so it never names the offender. Bisected with one attempt per process
(a failed capture poisons the context) in
[`bench/phase1/probe_capture_bisect.py`](bench/phase1/probe_capture_bisect.py):

| | site | what | named beforehand? |
|---|---|---|---|
| HA | `FusedGroupedNf4.forward` | `[int(e) for e in expert_ids]` — one D2H sync **per group** | yes |
| HB | `gemm_4bit_grouped` | pageable `torch.tensor(list, device=)` | yes |
| HC | `build_group_tiles` | **three** pageable transfers per call, called **twice** per step | **no** |
| HD | `dgrad_4bit_grouped` | HB again, in the backward | **no** |
| HE | `lora_delta_grouped` | two more, plus a `repeat_interleave` reading its output length off a device tensor | **no** |

HA wants a Python list and HB wants a device tensor, which is why neither
`expert_ids` form captured. HC and HD key off `sizes`, so neither named candidate
touched them. **With both named candidates removed, capture still failed.**

Fixed as call-path changes — no kernel source, tiling constant, dispatch
threshold or dtype moves, and **every output is bitwise identical** (26/26
tensors, `torch.equal`, both `expert_ids` forms, forward and backward, including
the `dgrad_kernel=False` and `_PAD_WASTE_LIMIT` fallbacks). `expert_ids` is now
accepted and passed through as a device tensor, converted once at the boundary
when a list is given; index tensors reach the device through one pinned async
transfer instead of several pageable syncing ones (8 → 4 per step, list form;
7 → 2, tensor form).

**Which construct is legal inside a capture was measured first**, and it changed
the design: pinned memory *allocated inside* the region still fails. Only a
**pre-allocated** pinned source with `non_blocking=True` is capturable, so the
staging lives in a persistent per-device arena.

The arena answers the reuse hazard the pinned-staging entry below already names
— *"a `non_blocking=True` copy is not ordered against the host writes of the next
call"* — with the event that entry prescribes: the bump pointer rewinds only when
the stream is not capturing **and** the event recorded after the last hand-out has
completed, so the host can never overwrite bytes a pending DMA has not yet read.

**Capturability is a precondition, not a speedup**, and no number here is
reported as one. MoE routing changes every step, so a replayed graph replays the
metadata it captured; making a captured graph *usable* needs a padding or
bucketing scheme with its own registration. Scope registered pre-data in
[`kernel/prereg_capturability_scope.json`](kernel/prereg_capturability_scope.json);
full write-up in
[`RESULTS-capturability.md`](bench/phase1/results/dequant_forward/RESULTS-capturability.md).

### Benchmark harness: two standing defaults

* **Every leg reports its measurement class.** The GPU-busy fraction now runs in
  every leg beside the self-pair rather than as an after-the-fact probe. A cell
  where **either** arm is below 50% GPU-busy is a `step_ratio`, not a kernel
  measurement, and is labelled that way
  ([`kernel/prereg_gpu_busy_labelling.json`](kernel/prereg_gpu_busy_labelling.json)).
  Backfilled onto legs 2 and 3 — the numbers stay, the framing changes.
* **Real prose is the fixture default**; random token ids are opt-in and
  understate the fused advantage by 1.6–1.7×. Routing occupancy, cv, and the
  fixture's *name* land in every receipt.

## 0.12.0 — 2026-08-14

### The package would not import at all on a platform it declares support for

`pyproject.toml` pins triton as `triton>=3.4; platform_system == 'Linux'` — a
deliberate marker, since there is no triton wheel for arm64 Darwin at all
(`pip install triton` there: "No matching distribution found"). A non-Linux
install is therefore a **supported** configuration by this package's own
packaging. It simply did not work: `nf4_grouped`, `mxfp4_grouped` and
`host_gather` each ran a bare `import triton` at module scope, so importing any
of them raised `ModuleNotFoundError: No module named 'triton'`.

The failure landed in exactly the wrong place. What this package promises
without a GPU is specific and pure-torch: `dequant_ref` (whose docstring already
says "Runs on CPU (no CUDA/Triton)"), the README's CPU quickstart, and the
**taught** refusal — "requires CUDA tensors ... use `dequant_ref(packed, absmax,
N, K)`" — that `test_cpu_refusal` pins as doctrine. A raw import error preempted
all three, so a CPU-only user following the README got precisely the unhelpful
error that guard exists to prevent, one import too early for it to speak.

`kernel/_triton_shim.py` centralizes the import. Where triton is present it
binds the real modules and nothing else changes, so the CUDA path is unaffected
by construction. Where it is absent, `@triton.jit` still **defines** the kernels
— they are built at import, so the decorator must succeed — while a *launch*
raises, naming the CPU alternative for the module in hand: `dequant_ref` for
NF4, `mxfp4_pack_ref.dequant_mxfp4` for MXFP4, and for `host_gather`, the honest
answer that a device-side gather over UVA has no CPU equivalent.

This had also disarmed the contributor checklist: `test_readme_cpu_block.py` and
`test_cpu_refusal.py` are an "Always" item in the PR template *because* they are
the CPU-only tests anyone can run — and on a machine without triton they could
not even be collected. Whole kernel suite on such a box went from 196 passed
with 7 collection errors to 235 passed with none. (#78)

### The arena tier could not read DeepSeek-V4's own scale dtype

`nvme_residency._ST_TO_TORCH` had no entry for `F8_E8M0`, so staging a real
DeepSeek-V4 MXFP4 arena died with `KeyError: 'F8_E8M0'` in `segment_geometry` —
and again in `segment_tensor`, which resolves the same table.

The gap was narrow and internally inconsistent: `nvme_bake_nf4._MXFP4_BYTE_DTYPES`
and `mxfp4_residency._PACKED_BYTE_DTYPES` both already listed the tag, with
comments saying V4 labels its MXFP4 experts `I8`/`F8_E8M0` where Kimi K3 labels
both `U8` — same bytes, different label. So this package could **bake** such an
arena and **serve** from it, and only the path through `segment_geometry` — the
one a training tier's geometry check takes — could not read it back.

Found on a rented 3090 after a 149 GB download and a 147 GB relocation bake of
`deepseek-ai/DeepSeek-V4-Flash`; the bake and the load both succeeded and the very
next call raised. Maps to `uint8`, not `float8_e8m0fnu`: these tags label **bytes
to hand back unchanged**, and materializing an e8m0 exponent as a float then
casting yields the value rather than the exponent byte, scaling every block by
`2**-127`. (#75)

### `preadv` scatter: rows land in per-segment staging by DMA, no host copy

`fetch_raw` now issues ONE scattering read per row — `os.preadv` takes an iovec list, so the
kernel writes each segment straight into its staging slot. The CPU never touches the bytes.

That matters twice over. Profiling a layer fetch found the host memcpy at **39.7%** of it,
and — separately — that a CPU write to pinned memory makes the FOLLOWING H2D **~6x slower**
(70.5 ms vs 11.65 ms for the same 281 MB, same tensors, same process). A host copy is
charged once to make and once as a penalty on the transfer. Both are gone.

Measured on a real K3 layer (896 experts, top-16 decode, 281 MB):

| | per layer | achieved | of device |
|---|---|---|---|
| original | 1113.1 ms | 0.757 GB/s | 11% |
| + single fetch (#74) | 387.3 ms | 0.725 GB/s | — |
| + pinned staging (#76) | 111.4 ms | 2.52 GB/s | 11% |
| **+ scatter** | **54.3 ms** | **5.17 GB/s** | **75%** |

`scatter == copy` **bitwise on real released K3 bytes**, checked on the pod, not only against
toy fixtures.

**It refuses rather than guesses.** O_DIRECT needs every iovec base and length
`align`-aligned, and `preadv` fills sequentially so inter-segment gaps and row padding need
scratch. `_scatter_layout()` returns None unless every segment length and every gap is
align-aligned, and `fetch_raw` records `last_fetch_path` so a fallback is visible — a silent
one would be a silent multi-x regression. K3 qualifies (all six lengths are multiples of
4096, `row_stride == row_bytes`); the shared toy fixture does NOT, which is why the scatter
tests carry their own aligned fixture and assert the path taken.

Short reads resume INSIDE the buffer they stopped in (`_advance`), not at the next one —
getting that wrong would drop or duplicate a segment's bytes and produce a plausible tensor.

### `ArenaExpertSource.fetch_raw` lands rows in pinned staging — 3.5x

`fetch_raw` copied every byte on the host **three times** before the device saw it:
`bytearray(mv[...])` once per SEGMENT per expert (a required copy, since the landing
buffer is reused, but a Python-level one), then `torch.stack`, then a pageable
`.to(device)`.

The measured symptom was that the path ran at **~0.72 GB/s regardless of the device** —
identical on a 6.88 GB/s and a 22.71 GB/s NVMe. That is a host ceiling, not a read limit.

Now the reader's bytes go straight into a **pinned `[E, length]` staging tensor per
segment**, reused across calls, and each segment moves to the device in one transfer. This
is `nvme_residency.segment_into`'s shape applied to the serving path, which predated it.

Measured on a real K3 layer (896 experts, 17.5 MB rows, top-16 decode), same slice, same
bench, same pod class:

| | per layer | achieved |
|---|---|---|
| before | 387.3 ms | 0.725 GB/s |
| after | **111.4 ms** | **2.52 GB/s** |

**3.5x**, and 10x cumulative with the amplification fix above (1113 -> 111 ms). Bytes read
are unchanged at 18,529,910,784 — the fix removes host copies, not reads, and the counter
confirms it.

Returned tensors never alias the staging: `.to()` is a no-op when the source is already on
the target, so on the DEFAULT `device="cpu"` the result would have handed back the reused
buffer and the next fetch of the same expert count would rewrite a caller's earlier result
in place (Cursor Bugbot). Detected by pointer identity rather than by comparing device
strings, which get `cuda` vs `cuda:0` wrong.

The device transfer is **synchronous on purpose**: staging is reused, and a
`non_blocking=True` copy is not ordered against the *host* writes of the next call, so the
CPU could overwrite staging mid-DMA. Making it async needs an event recorded here and
waited on before reuse. Still ~9x under this box's device rate (22.94 GB/s at qd=16), which
stays open in #73.

### `moe_layer_forward` read every expert row three times

A row carries all six segments, but `moe_layer_forward` called `fused_stacks`
once per projection and each call independently ran `fetch_raw` — reading the
whole row, returning two segments, discarding four. Measured on a real 1-layer
K3 slice (896 experts, 17.5 MB rows, top-16 decode): **842 MB read where 281 MB
is needed**, confirmed to the byte by the reader's own counter.

One `fetch_raw` now serves all three projections. `fused_stacks` gained
`raw=` so a caller wanting more than one projection can fetch once; passing it
is what removes the amplification, and single-projection callers are unchanged.

Measured alongside it and NOT fixed here: the path achieves **0.757 GB/s against
6.88 GB/s** available on the same file, same pod, same queue depth — a further
9.1x that is neither the amplification nor the device (gnf4#73).

The gate lives in its own `test_arena_fetch_amplification.py` rather than in
`test_arena_experts.py`, which the packaging guard allowlists as "needs CUDA" and
CI therefore never runs. It stubs the GEMM, because the question is how many
times the bytes are read, not what the kernel computes.

## 0.11.0 — 2026-08-13

### `bake_nf4 --absmax-dtype bf16`: 5.6% off every arena row, bitwise lossless

absmax is **11.1% of a Qwen3-30B row** (294,912 of 2,654,208 B) and shipped as fp32.
For a bf16 checkpoint it can be stored bf16 with **no change to any computed value**:
absmax is `|w|.amax()` over a block, so it *is* one of the source magnitudes, and the
maximum of a set of bf16 values is a bf16 value. Measured on the real Qwen3-30B —
**80/80 expert tensors bitwise identical** after a bf16 round-trip, with an fp32-source
control correctly *not* identical.

Against ground truth (the original bf16 weights), where the NF4 quantization floor is
12.0766% relative RMSE:

| absmax storage | rel. RMSE | vs the floor | row |
|---|---|---|---|
| fp32 | 12.0766% | — | — |
| **bf16** | **12.0766%** | **+0.00%** | **−5.6%** |
| int8 (per-256 linear) | 12.0852% | +0.07% | −8.3% |

int8 was the obvious candidate and is the worse trade: 2.7 more points of row for a
numerics change, a re-bake accepted as a different quantization config, and a kernel
contract that excludes nested absmax.

- **`--absmax-dtype {f32,bf16,auto}`**, default **`f32`** — unchanged behaviour. The index
  is self-describing, but *older readers are not*: one predating this refuses the segment,
  so flipping the default would break them on a library upgrade alone.
- **`auto`** decides from the **source dtype** (a proof) rather than by sampling values (a
  guess that can pass on the experts it looked at).
- **`cast_absmax` refuses an inexact cast** instead of rounding quietly, and is applied in
  `bake_nf4` after the quantizer — so an *injected* `quantize_fn` cannot return a width the
  row geometry did not budget for, which would write a short row and shift every later
  offset.
- **`segment_into` widens** bf16/fp16 segments into an fp32 destination via a converting
  `copy_` instead of a memcpy. VRAM and the kernel contract are untouched: bf16 on disk,
  fp32 absmax in VRAM. `widening_casts()` is exported so consumers test the same table;
  **narrowing is not in it and must not be added.**

### The arena reader's queue depth scales with the host's CPU budget

`ArenaReader(qd=...)` and `ColdTier(qd=...)` now default to `None`, which resolves to
`clamp(cpus // 4, 4, 16)` via `nvme_reader.default_qd()`.

The old fixed `qd=4` was measured optimal on a 12-core box that sat at load ~9.8, where 8
and 16 came back *worse*. Re-measured on an idle 32-vCPU L40S against the same arena and
the same scattered pattern, that inverts:

| qd | O_DIRECT | vs qd=4 |
|---|---|---|
| 1 | 2.04 GB/s | 0.38x |
| 4 | 5.31 GB/s | — |
| 8 | 5.95 GB/s | +12% |
| 16 | 6.13 GB/s | +15% |

So 4 was tuned to a CPU-starved regime rather than to the device.

**This cannot regress a smaller host:** the divisor is coarse and the floor is 4, so
anything under ~20 CPUs gets exactly the depth it got before. The cap is 16 because that
is where the measurement stops.

`cpu_budget()` reads the **cgroup quota** before `sched_getaffinity`/`cpu_count`: in a
container with a CPU quota and no cpuset both of those report the *host's* cores — 256 on
the box above, where the real budget was 27.2.

The serving path (`arena_experts.ArenaExperts`) already defaulted to `qd=16` and is
unchanged; the measurement covers the training access pattern only.

## 0.10.0 — 2026-08-13

### `bake_nf4` handles fused expert layouts (Gemma-4, GraniteMoe)

A fused checkpoint ships **one 3-D `[E, X, Y]` tensor per layer** instead of per-expert
2-D tensors, and was unbakeable. Now detected rather than flagged: when per-expert
discovery finds nothing, `E` is read off `shape[0]`.

The arena row format needed no change — `bake_nf4` already quantizes `cat[gate;up]` as one
`[2I, H]` matrix and Gemma-4 ships it pre-concatenated (`[128, 1408, 2816]` slices to
exactly that). Neither did the provenance schema: the record is already
`(file, byte range, sha256)`, and a fused slab is a byte range like any other, so
`verify --against-source` re-checks exactly the bytes consumed.

Verified end-to-end on `google/gemma-4-26B-A4B`: **8/8 arena segments byte-identical** to
what experts4bit-qlora's loader builds, **8/8 provenance ranges** re-read and matched. A
full 3840-row / 12.85 GB bake then trained end-to-end, with step-0 loss bit-identical to
the host-resident arm — the check that catches a wrong expert ORDER, which no hash would.

**Refuses what it cannot bake correctly.** Per-slab and whole-stack quantization coincide
only when each expert's numel is a multiple of the 64-element block. That is asserted per
projection, because a checkpoint failing it would bake rows the loader silently cannot
reproduce.

### `capacity_for_bytes` no longer over-promises rows for pinned tiers

It returned `usable_bytes // row_stride`, assuming a row costs exactly its stride. For a
pinned tier — the default, and the one the docstring points callers at — that hands back a
`hot_rows` that OOMs partway through the first training step.

`pinned` now defaults to `True` and applies `PINNED_ROW_FACTOR = 1.9`; `pinned=False` gives
the old arithmetic for the mmap tier; `factor=` overrides.

**The constant is conservative and not well determined.** A follow-up measurement across
five `hot_rows` values found the relationship is not linear in `hot_rows` over the range
that matters — 3.1 GB of extra pinned buffer between 2048 and 3216 rows did not move the
requirement at all, and three explanations for that were tested and refuted. Every
measurement says 1.9 **under**-promises, so nothing built on it is unsafe, but it should be
read as a safe bound rather than a measured cost. See grouped-nf4-gemm#58.

Not this module's doing either way: the same effect reproduces on a bare
`torch.empty(n).pin_memory()` with no gnf4 code in the process.

### The bake says what it searched for

Discovery matching nothing surfaced as `ValueError: max() arg is an empty sequence`, naming
none of the three things that decide the match. That cost a diagnosis twice — Kimi K3's
`.weight_packed`, then Gemma-4's fused layout. The error now prints the prefix, marker and
key it searched for, plus either the near-miss names the checkpoint really has or, when the
layout is fused, that this path does not support it. `--prefix` and `--moe` are exposed on
the CLI; `bake_nf4()` always accepted them.

## Shipped in 0.10.0: the `dgrad_kernel` default (merged 2026-08-12, after 0.9.0, under an "Unreleased" heading; there was no 0.9.1)

**`dgrad_kernel` now defaults to `True`.** The single-launch dgrad has been
opt-in since 0.7.0, so the QLoRA backward took the per-expert decode loop unless
a caller asked otherwise.

- **Why it flipped.** The loop decodes through `dequant_ref`, so its gradient is
  exact — that was the entire case for the old default, and it never priced
  itself against the gap 0.7.0 had already measured: **5.92 ms vs 61.78 ms**
  (gate_up, E=256) and **3.28 ms vs 85.12 ms** (down, E=256) on an A2000 at
  T_cat=4096, with the composed training step at **403.7 → 26.5 ms**. The loop
  materializes a decoded expert per group, which is exactly the round trip the
  fused forward exists to avoid: the shipped default was paying the forward's
  thesis back in the backward.
- **Fidelity.** ~2.9e-3 relative against the exact loop — an order of magnitude
  inside bf16's own mantissa budget (eps ~3.9e-3, and a K-term dot accumulates
  ~sqrt(K) of it). Not zero, which is why this is a changelog entry and not a
  silent tweak.
- **Escape hatch unchanged.** `dgrad_kernel=False` restores the exact loop — use
  it for a bit-exact A/B against a reference trainer, or convergence forensics.
  Every guard still declines to the loop on its own: ineligible shapes, non-bf16
  gradients, evicted storage, and offload-staged weights on another device.
- `test_dgrad_kernel_is_off_by_default` is inverted to
  `test_dgrad_kernel_is_on_by_default` and now pins **both** halves — the
  default must be the kernel, and `dgrad_kernel=False` must still reach the
  exact loop, so the escape hatch the new default depends on is itself tested.
  Mutation-verified: restoring the old default fails it.
  `test_fused_backward_matches_dequant_reference` now pins `dgrad_kernel=False`
  explicitly, so its exactness assertion keeps meaning what it says instead of
  silently re-scoping to whatever the default becomes.

## 0.9.0 — 2026-08-12

**The arena grew a staging seam: `segment_into` fills a destination the caller owns.**

- **`nvme_residency.segment_into(tier, index, layer, experts, suffix, out, rows=…, non_blocking=…)`.**
  `segment_tensor` is the *serving* seam and allocates its own `[R, *shape]` result,
  which a staging path cannot use: staging holds one reusable buffer (or writes
  straight to the device) and fills only the routed rows of a full-shaped
  `[E, …]` destination. A pageable result is also a quiet correctness trap —
  copying from pageable memory silently downgrades `non_blocking=True` to a
  synchronous copy, so a caller believes it overlapped a transfer it did not.

  When the tier is pinned this is genuinely zero-bounce. `ColdTier` already lands
  rows in pinned memory, so the segment is read out of the pinned slot itself:
  disk → slot → `out`, with no intermediate host allocation. `segment_tensor`
  cannot do that at all — `torch.frombuffer` needs a writable buffer, so it copies
  through a `bytearray` first. Unpinned tiers keep that fallback, correct but with
  the extra copy.

  Bytes move as `uint8`, so bit-identity holds by construction rather than through
  a dtype-reinterpretation step that could disagree with `segment_tensor`'s.

- **`nvme_residency.segment_geometry(index, suffix)`** — `(dtype, shape_per_expert,
  seg_off, length)` without touching the tier, so a caller can size its landing
  buffer at setup rather than after the first row is resident.

- **Destinations that cannot be filled correctly are refused, not mangled.** A
  mismatched dtype reinterprets the bytes, a wrong trailing shape shifts every
  row, and a non-contiguous `out` makes `reshape(-1)` a copy that is silently
  discarded. Each raises with the mismatch named.

- **13 tests, wired into CI's NVMe step.** `test_packaging_covers_kernel` caught
  that the new file would otherwise have run nowhere — the guard working as
  designed. The pinned branch is exercised against a stand-in tier whose
  `pinned_tensor()` is an ordinary CPU tensor, because the `[slot, off:off+len]`
  arithmetic is where a skew hides and both a wrong stride and a dropped segment
  offset produce plausibly-shaped output. Two mutations confirmed the suite is
  armed: dropping the segment offset on the pinned path, and ignoring `rows=`.

  A real pinned `ColdTier` needs CUDA and is **not** exercised on CPU CI.

Consumer: `experts4bit-qlora`'s arena-backed training path
(`enable_nvme_train_residency`) stages every layer through this.

**Also, CI-side — no effect on the published wheel:**

- **The README link check stopped calling throttling a dead link.** It opened a
  fresh TLS connection for each of ~35 links in a tight loop and GitHub's edge
  dropped some of that churn, so the step failed on load — SSL handshake timeouts
  here, and on `experts4bit-qlora` a run reporting 28 of 28 links dead on a tree
  where every path existed. Established by measurement, not assumption: a URL that
  failed four `urlopen` attempts in a row answered 200 three times in a row under
  `curl`. One pooled keep-alive connection per host fixes it — 35/35 in 16 s.
  Retry is a backstop, scoped to answers that are not verdicts: **404 and 403 are
  never retried into a pass**, because a gate that turns dead links green is worse
  than one that is merely flaky.
- **The link check no longer forwards `Authorization` across origins.** Replacing
  `urlopen` with a hand-rolled `http.client` loop silently dropped its cross-host
  header stripping, and GitHub 302s assets to `*.githubusercontent.com` and object
  storage — so the Actions token would have followed. Same-origin only now (scheme
  **and** host; an `http` downgrade counts as foreign). Caught by Cursor Bugbot on
  #47.

## 0.8.3 — 2026-08-12

**Test isolation enforced, not just documented.**

- **`pytest kernel/` on a GPU box now refuses up front instead of aborting mid-run.**
  `TRITON_INTERPRET` is read when triton is first imported and latches for the life of
  the process. Two test files set it at module scope, so collecting one flipped the
  global knob and the process then died with `Cannot call @triton.jit'd outside of the
  scope of a kernel` — a stack dump, not a test failure. A fixture cannot fix it (triton
  has already read the variable before any test runs), so `conftest.py` rejects the mixed
  run and prints the split commands. The constraint was documented; nothing enforced it.

  Gated on a CUDA device actually being present. The crash needs a test that launches a
  real kernel, and with no device those skip, so mixing is harmless — which is exactly
  CI's "CPU-reachable suites" step, running `test_mxfp4_interp.py` alongside eight
  compiled-path files and passing. Refusing on filenames alone would have broken that
  green step.

## 0.8.2 — 2026-08-11

Backfilled: this entry was missing when 0.8.2 shipped.

- **Malformed k-quant input raises `ValueError` with diagnostics rather than tripping a
  bare `assert`.** Asserts vanish under `python -O`, so on-disk validation stated as an
  assert is validation that silently disappears in exactly the deployment that strips it.

## 0.8.1 — 2026-08-11

Backfilled: this entry was missing when 0.8.1 shipped.

- **The fused path is refused below triton 3.4** — it crashed there, and the obvious
  guard then made it silently *wrong* rather than absent. Both halves fixed (#45).

## 0.8.0 — 2026-08-10

**GGUF k-quant decode lane.** Reads released GGUF files and computes their bytes
directly — never a re-quantization — so a llama.cpp-format checkpoint can be served
from the exact weights its publisher shipped.

- **`kernel/kquant_ref.py`** — pure-torch dequant for `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/
  `Q6_K`/`Q8_0` plus `F32`/`F16`/`BF16` passthrough, dispatched **by ggml type per
  tensor**. That is what makes any publisher's file work through one table: a
  "Q4_K_M" file is a mix (attention in Q4_K, some ffn in Q6_K, norms in F32), and
  dynamic quants re-mix per tensor. Scope was set by parsing real released headers,
  not filenames. IQ i-quants refuse explicitly rather than guess a codebook.
- **`kernel/gguf_reader.py`** — GGUF v2/v3 header parse (metadata, tensor table,
  absolute byte extents). Every length is bounds-checked before use and a truncated
  header raises `NeedMoreBytes(minimum)` instead of guessing, so the same parser is
  safe against a ranged prefix as against a local file.
- **Oracle-adjudicated bit-exactness.** `kernel/test_kquant_ref.py` compares against
  gguf-py (the llama.cpp project's own numpy implementation) with int32-view equality
  — disagreement is STOP, not tolerance. A synthetic arm always runs; an env-gated arm
  checks sha256-pinned tensors range-fetched from real released files by
  `scripts/fetch_gguf_fixtures.py`.
- Validated on real bytes at scale: 27 sampled tensors across two publishers' 30B
  GGUFs (every quant type, layers 0 through 51) decode bit-exact and finite.

## 0.7.1 — 2026-08-06

Docs-only patch: the PyPI page for 0.7.0 froze a warning that has since been resolved
by measurement, and the training work had no README presence at all.

- **The dgrad layer-composed caveat is retired.** 0.7.0 shipped "layer-composed fidelity
  is unmeasured — gate a real run on your own parity check." Measured same-day at 16 and
  48 layers from the published wheels (experts4bit-qlora `bench/dgrad-gate/`): dgrad adds
  nothing to the fused lane's composed gradient error (4.97e-2 → 4.99e-2 mean at 48
  layers), an fp32-truth arm shows every lane on the composed bf16 noise floor (the fused
  lane *closest* to truth at 16 layers), and a 20-step real-data trajectory gate passes at
  a third of its band with dgrad at 2.87x the reference's step rate.
- **sm_120 verified.** 66 kernel tests at the v0.7.0 tag pass on an RTX PRO 4500
  Blackwell (capability 12.0 — the same arch as the RTX 5090); `_DGRAD_DEFAULT` tuned on
  sm_86 holds there, every swept config bit-identical, and dgrad measures 67–103x over
  the Python decode loop (vs 10–26x on sm_86).
- **README documents the 0.7.0 training work** — `dgrad_4bit_grouped` in the entry-point
  table, a training section with the measured numbers, and the opt-in's semantics.
- **Attribution**: the comparison baseline in the 0.7.0 notes (`enable_batched_train`)
  is @jiwoon-ahn's whole-stack-dequant approach from experts4bit-qlora#38; now credited.

No code changes; the kernel is byte-identical to 0.7.0.

## 0.7.0 — 2026-08-06

**Both per-expert Python loops in the training lane are gone.** They were the
dominant cost of a fused training step and each hid the other: removing one alone
buys little, because whichever remains dominates.

**`dgrad_4bit_grouped` — the backward of `gemm_4bit_grouped`, in one launch.**
There was no backward kernel at all, so `FusedGroupedNf4.backward` looped the
active experts in Python with a `dequant_ref` + matmul each. At 256 experts over
40 layers that is ~10k decode+matmul pairs per step, measured at 78-84% of an
experts4bit-qlora training step.

The transposed contraction cost nothing structurally: the weight tile is
`[BLOCK_N, BLOCK_K]` in both directions from the same pointer arithmetic, and
with `BLOCK_K` dividing 64 the whole output tile sits in one quant group, so the
absmax column index is a scalar rather than a gather. Against the per-expert
decode oracle on an A2000, T_cat=4096: gate_up E=256 **5.92 ms vs 61.78 ms
(10.4x)**, down E=256 **3.28 ms vs 85.12 ms (26.0x)**. A tile sweep put the
default config at 0.91x of the *forward* kernel's time on the same problem — it
reaches the forward's ceiling. Every config in the sweep produced bit-identical
output, so the config knob is speed, not fidelity.

It materializes nothing: the decode happens in registers inside the GEMM as the
forward does, preserving "packed bytes are the only residency". The whole-stack
dequantize alternative also beats the loop but spends ~1.6 GB per layer at
production width.

**Opt-in** via `dgrad_kernel=False` on `FusedGroupedNf4`,
`gemm_4bit_grouped_train`, and `fused_grouped_lora`. The default stays the loop,
whose gradient is EXACT (it decodes with the same oracle the reference uses, and
a test asserts `grad_rel == 0.0`); the kernel accumulates fp32 in a different
order and lands near 2.9e-3 — inside the bf16 budget, not zero. Opted in it
declines rather than fails: `dgrad_eligible()` is askable before launch, and the
fallbacks are non-bf16 gradients, a `BLOCK_K` that does not divide the quant
blocksize, empty/evicted storage, and offload-staged weights on another device —
where the kernel would need the whole stack resident, which is what offload
exists to avoid.

**`lora_delta_grouped` is batched.** It ran a Python loop over experts in the
*forward*, putting `2E` matmul nodes per projection per layer on the autograd
graph and paying for them again in backward. Padding the groups and running two
`bmm`s measured **2.96x on the end-to-end training step** at E=256 (403.7 → 136.5
ms) for +36% peak memory, gradients agreeing to 1.6e-3. Past `_PAD_WASTE_LIMIT`
(4x real rows) the loop is used instead, so pathological router skew cannot cost
more than it did before; the loop survives as `_lora_delta_grouped_loop` and is
the oracle the tests compare against.

**Together**, on the same A2000 step at E=256: **403.7 → 26.5 ms (~15x) at 134 MB
peak**. For scale, experts4bit-qlora's kernel-free `enable_batched_train` runs
that step in 25.0 ms but at 417 MB — this lane now matches it at under a third of
the memory.

That comparison baseline is not ours: `enable_batched_train` implements
@jiwoon-ahn's whole-stack-dequant approach from
pjordanandrsn/experts4bit-qlora#38. Measuring against it is what made the size of
the backward gap visible in the first place — see #34.

Layer-composed fidelity of the dgrad path is unmeasured. This repo has seen a
per-op-more-accurate path cost +0.023% perplexity through 16 layers, so gate a
real training run on your own parity check before flipping it on.

## 0.6.0 — 2026-08-02

**`bake_nf4(source="fp8")`: block-scaled FP8 checkpoints can be baked.** Until now the bake
read `bf16` or `mxfp4`. DeepSeek ships *both* formats under the same tensor names —
V4-Flash's experts are MXFP4 (137 GiB), **V4-Flash-Base's are block-scaled FP8 e4m3
(258 GiB)** — so the Base checkpoint could not be baked at all, and the `source="mxfp4"`
path pointed at it produces a correct-shaped arena of nonsense rather than an error.

Two things differ from the MXFP4 path and both are silent if crossed:

* **The on-disk shape is already logical.** MXFP4 packs two nibbles per byte so the bake
  doubles its K back; FP8 is one byte per element, and doubling here would describe a
  matrix twice as wide as the model has.
* **The scale is an F32 per `[128, 128]` tile**, not an e8m0 byte per 32 elements, and it is
  already the multiplier — no `2**(x-127)`. `read_fp8` rejects an `F8_E8M0` scale (that
  means MXFP4) and a non-`F8_E4M3` weight, rather than reading either as the other.

Validated against the real 149 GB `DeepSeek-V4-Flash-Base`: the reader is **bit-identical**
(max relative error `0.000e+00`) to `experts4bit-qlora`'s independently written
`dequantize_fp8_blocks`; geometry resolves to the correct `43L x 256E, I=2048, H=4096`; the
full arena bakes to **155.8 GB in 4890 s**; and the served model answers
`"The capital of Japan is"` with ` Tokyo` at p=0.90.

5 tests on a synthetic FP8 snapshot (no checkpoint needed), 3 of which fail against 0.5.1.

## 0.5.1 — 2026-08-01

**0.5.0's `source="mxfp4"` bake could not read a Kimi K3 checkpoint, which is the one it
was fixed for.** 0.5.0 made `read_mxfp4`'s tensor suffixes a parameter (`mxfp4_suffixes`)
because K3 spells them `.weight_packed`/`.weight_scale` where DeepSeek-V4 says
`.weight`/`.scale`. It left the two places that go looking for those tensors — expert
DISCOVERY and the geometry probe — hardcoded to `.weight`. So on a K3-spelled checkpoint
nothing matched, and the bake died one line in with
`ValueError: max() arg is an empty sequence`.

Parameterizing the read was necessary and not sufficient. The signature tests 0.5.0 shipped
passed either way; **only running it on real K3 bytes found this.**

Verified on the A2000 against the real 1.4 TB `moonshotai_Kimi-K3` checkpoint: discovery
now finds all **896 experts/layer**, geometry resolves to I=3072 / H=3584, and a 4-expert
slice bakes in 5 s. The baked NF4 matches the source MXFP4 it came from at **cosine 1.0024,
mean relative error 0.079** — which is NF4 re-quantization error, as expected, not agreement
by construction.

Two tests, both on a synthetic K3-spelled MXFP4 snapshot so they need no checkpoint: one
that the bake completes and its provenance chain still closes against the source, and one
that the WRONG (V4) suffix pair raises rather than producing an empty or half-built arena.
The first fails against 0.5.0 with that same `max()` error.

## 0.5.0 — 2026-08-01

**DeepSeek-V4's experts, read and served from a native MXFP4 arena.** This is the half of
`experts4bit-qlora` 0.8.0's V4 path that lives here: `enable_mxfp4_nvme_residency` imports
`Mxfp4NvmeResidencyV4` and `V4_RESIDENCY_KINDS` from this package, so without it the
documented V4 arena path raises `ImportError`.

`nvme_bake_nf4` gains a `source="mxfp4"` bake — a **relocation** of the released bytes
rather than a re-quantization, which is why it is both smaller and faster to produce than
the NF4 lane (V4-Flash: 147 GB and ~80 s, against 156 GB and a full quantize pass) — plus
`proj=` for V4's `w1`/`w3`/`w2` spelling and `moe=` for its block name.

`Mxfp4NvmeResidencyV4` is a third epilogue, and it is neither parent's: gpt-oss's **clamps**
with SwiGLU's **combination**, over a **clean-concat** `gate_up` (like K3, unlike gpt-oss's
interleaved columns). Three independent choices, each of which produces a correctly-shaped
tensor when taken from the wrong parent.

It also evaluates the GLU in **fp32** and casts back only for the down projection, because
V4's reference does (`self.w1(x).float()`); the sibling epilogues stay in compute dtype
because *theirs* do. Reproducing an epilogue means reproducing its precision, not only its
shape — the same correction made across all five execution engines in
`experts4bit-qlora` 0.8.0.

`test_mxfp4_v4.py` gates all of it (pure python, no GPU, wired into CI): the transcribed
reference, the one-sided gate clamp, not-gpt-oss's-GLU, clean-concat-not-interleaved, and
the fp32 evaluation — the last asserted structurally, since the cast back to compute dtype
is larger than the difference a numeric test would be trying to see.

## 0.4.0 — 2026-07-31

**Version-number correction. No code change from 0.3.1.**

0.3.1 shipped `#26` (prefill — many tokens per call) alongside a packaging fix and
described itself as "nothing else changed". A new capability went out under a patch
label, so anyone reading versions rather than diffs had no signal it existed. 0.4.0
is the same tree under the number semver says that feature warranted, and 0.3.1's
entry now describes what it actually contained.

Nothing to migrate: if you are on 0.3.1 you already have prefill.

## 0.3.1 — 2026-07-30

**0.3.0 announced two modules it did not ship.** `mxfp4_residency` and
`nvme_residency` were absent from `pyproject`'s `py-modules` allowlist, so they
were never in any wheel — while 0.3.0's release notes described
`K3_RESIDENCY_KINDS`, `fuse_gate_up_segments` and `Mxfp4NvmeResidency` as
shipped. Anyone who followed those notes got `ModuleNotFoundError`.

`py-modules` is an explicit allowlist: adding a file under `kernel/` does not
package it, and nothing warned. `kernel/test_packaging_covers_kernel.py` now
diffs the directory against the allowlist and fails naming the missing modules,
so the next gap lands on the pull request instead of on a user. A module that
genuinely should not ship goes in `_DELIBERATELY_UNPACKAGED` with a reason,
which keeps that decision visible rather than silent.

**Correction (added after release): 0.3.1 also shipped a new capability, and its
notes said it did not.** `#26` — prefill: the engine takes many tokens per call —
merged to `main` before the packaging fix and was swept into this tag. The line
"nothing else changed" was written from the packaging work alone rather than from
the full `v0.3.0..main` delta, and it is wrong.

What that feature does: the engine was decode-only (`a_buf.copy_(x.expand(k, -1))`
broadcasts ONE token's hidden state across the k slots). The *kernel* never was —
`gemm_mxfp4_grouped`'s `sizes` is a per-group token count and already switches to
the tiled path above one row — so this is engine plumbing. Prefill is not decode in
a loop, and the difference is I/O: stepping T tokens re-reads the whole dense side T
times and every routed row T times; entering each layer once for the prompt reads
each *distinct* expert once. Measured on Kimi K3 at full depth, 7 tokens: dense
108.76 GB once vs 761 GB; expert rows 7,080 vs 10,304 (31 % deduped by route
overlap); **233 GB vs 942 GB of I/O, 187.4 s vs 643 s**. VRAM peak unchanged at
**3.59 GB**.

By semver that warranted a minor bump, not a patch. See 0.4.0.

## 0.3.0 — 2026-07-30

**If you are on triton 3.2, the pipelined MXFP4 engine did not work at all.**
Both kernel factories imported `triton.language as tl` into their *locals*. With
`from __future__ import annotations`, `BLOCK: tl.constexpr` is the **string**
`"tl.constexpr"`, which triton resolves against the jitted function's
`__globals__` — triton 3.4 tolerates it, 3.2 raises `NameError('tl is not
defined')` from inside the compiler. Moving `tl` to module globals takes
`test_mxfp4_residency`'s companion suite from **7 failed to 7 passed** on a
triton-3.2 box.

**Experts now serve from NVMe, and the arena you already baked is readable
whatever order it was baked in.** `Mxfp4NvmeResidency` reads gate_up at one
computed offset, so it needs the two blocks segments adjacent and the two scales
segments adjacent — while `arena_experts.K3_KINDS`, the released-K3 spelling,
interleaves per projection. Both orders are legitimate; they are for different
consumers. Rather than force a re-bake of 1.45 TB, the gather takes a per-piece
`(src, dst, len)` table and lands segments where the engine expects them: no
extra bandwidth, nothing on disk touched, **any bake order readable**. The
identity case keeps the original contiguous kernel, so the previously measured
path is untouched.

`K3_RESIDENCY_KINDS` carries the real tensor names in the order this engine
wants, and the two constants cross-reference each other so the trap is visible
from either file. A mis-ordered arena is now refused with a message naming the
order that works, instead of "trailing dims differ".

**The k slots are shared across layers.** They were per-layer, so VRAM scaled
with depth — on a 92-layer model that is the difference between fitting and not.

**Also:**

- `nvme_reader`: a pinned tensor is **not** reliably page-aligned. O_DIRECT
  needs the buffer address aligned, and assuming `pin_memory()` delivers that
  is how a good checkpoint reads as a corrupt one.
- K3's SiTU activation is registered from the **release source** — none of the
  guesses matched.
- `moe_layer_forward` passed **model-global** expert ids into the kernel's
  stack index, which reads out of bounds silently. Every toy fixture used ids
  smaller than the group count, so they indexed validly by coincidence; on
  896-expert K3 with top-16 it would have corrupted essentially every layer.
- The equivalence fixture quantized random nibbles against random e8m0 scales,
  whose magnitudes a GLU squares into overflow — and `torch.equal` is False for
  identical NaNs, so byte-identical outputs compared unequal. It now quantizes
  realistic weights and compares bitwise.

**Receipts** (`docs/`): the Phase-1 oracle passes — decode bit-identical to
compressed-tensors across 33,030,144 elements, max delta 0. A real-bytes arena
round-trip on released Kimi-K3 matched 48/48 segments against the shipped
safetensors, with a byte-flip negative control. Each carries a note on what its
OpenTimestamps anchor does **not** prove: the stamps were applied after the runs,
so they establish the text has not changed since, not that the protocol predated
the data.
