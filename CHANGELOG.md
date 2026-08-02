# Changelog

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
