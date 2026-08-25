# Changelog

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

## Unreleased

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
