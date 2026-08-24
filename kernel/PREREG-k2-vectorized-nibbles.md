# PREREG — K2: the vectorized dual-nibble GEMV mainloop

Registered 2026-08-24, before any measurement. The follow-on
RESULTS-m1-decode-config fixed the baseline for: after K1's config
winners, the decode GEMV runs **3.48 ms/step equivalent at ~18% of the
measured 1573.9 GB/s** (winner pair-time 72.4 µs; floor 13.5 µs/pair).

## The mechanism, quoted from the kernel

Both decode GEMV kernels (`_gemv_nf4_grouped` and its split-K variant)
load packed bytes as:

    bytes_ = tl.load(b_base + (kk[None, :] // 2), ...)
    nib = tl.where((kk % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)

BLOCK_K lanes address BLOCK_K/2 distinct bytes — consecutive lane
PAIRS hit the SAME address, so half the load lanes are redundant and
the duplicate addressing defeats wide vectorization; per 32-lane
transaction only ~16 useful bytes arrive.

## The registered edit (value-identical by construction)

Load the packed row HALF-WIDTH and contiguous, then expand both
nibbles in registers:

    by  = tl.load(b_base + k0 // 2 + offs_kb, ...)      # BLOCK_K/2 bytes
    nib = tl.interleave((by >> 4) & 0xF, by & 0xF)      # hi,lo,hi,lo…

bnb packs element 2j into the HIGH nibble and 2j+1 into the LOW, so
`interleave(hi, lo)` reproduces the element order EXACTLY — the same
nibbles reach the same LUT gather in the same lanes ⇒ **bitwise-equal
outputs at any fixed config**. That is the hard fidelity gate this
cycle (unlike K1's cross-config ulp allowance).

Both kernels gain a `VEC_LOADS: tl.constexpr` branch; the wrapper
selects it when `tl.interleave` exists (triton ≥ 3.0; the same
bind-once shim pattern as `_TL_GATHER` keeps triton 3.2 importable)
and `GNF4_GEMV_SCALAR_LOADS=1` forces the legacy path — the A/B arm
and the fallback in one knob. The M-tile prefill kernel is OUT of
scope (B=16 sanity guards it).

## Instruments

- The K1 sweep bench re-times the K1 winner configs under BOTH paths
  (legacy env vs default) — same registered median estimator.
- CI: an interpreter-mode bitwise test (both paths, random
  packed/absmax, `torch.equal`) — with the known caveat that interp
  mode masks int-width bugs, so the ON-BOX bitwise arm is the real
  gate and CI is the tripwire.
- e2e: the b1d graph harness, arms g1 (legacy env) / g2 (vectorized).

## Bars (before any number)

- **G-B (bitwise, hard)**: on-box, winner configs, both kernels:
  outputs `torch.equal` legacy-vs-vectorized, AND e2e tokens
  bit-identical between the arms. ANY mismatch ⇒ REFUTED regardless
  of speed (the construction claim is false).
- **H-K**: winner-config pair-time under the vectorized path ≤
  **45 µs** (from 72.4; ×0.62). PARTIAL (45, 58]; > 58 ⇒
  REFUTED-FOR-VARIANT: legacy stays default, and the lane escalates to
  the layout question (row-interleaved packing) with a fresh
  registration.
- **H-E**: graph step ≤ **12.6 ms** (≥ 79 tok/s). KERNEL-WIN-NOT-AT-
  WALL if H-K passes without it.
- **GS**: B=16 in the certified band (M-tile untouched).
- Consequences: all pass ⇒ CERTIFIED — vectorized is the default (it
  already is, guarded by availability; the RESULTS PR records it);
  props suite 48/48 under the default required as in K1.

## Verdict calculator

`k2_verdict.py`, self-tested both directions; receipts in
`kernel/receipts-k2-vecnib/`.
