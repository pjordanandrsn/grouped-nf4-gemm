# Changelog

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
