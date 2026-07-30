# Changelog

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

Nothing else changed. Everything 0.3.0 describes is accurate; two of its modules
were simply unreachable from an installed wheel.

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
