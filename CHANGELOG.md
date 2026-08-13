# Changelog

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
