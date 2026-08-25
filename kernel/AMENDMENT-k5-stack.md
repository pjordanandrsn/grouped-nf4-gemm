# AMENDMENT — K5: the registered sweep is uncompilable on the receipts stack

Registered 2026-08-24, after attempt 1 returned a REFUSE receipt
(`receipts-k5-probe/attempt1-refuse-t33/`) and before any new-stack
timing exists.

## What attempt 1 established

On the campaign receipts stack (torch 2.7.0+cu128 / **triton 3.3.0**,
RTX 5090 = sm_120, box 48617509, EPYC 9654, driver 575.64):

1. **Every M-tile config fails to compile — the path does not exist on
   this stack.** VARIANT=1 (register-LUT) dies in triton's
   `OptimizeThreadLocality.cpp:227` `setOptimizedGatherLayout:
   isWarpLocal()` assertion at **all** BLOCK_M ∈ {16, 32, 64, 128};
   VARIANT=0 (v5 mainloop) dies in `AccelerateMatmul.cpp:42`
   `getMMAVersionSafe` — triton 3.3.0 cannot lower `tl.dot` for
   sm_120 at all. This is a third terminus the prereg did not
   enumerate: not slower, not faster — absent. (v6 certified the
   M-tile path on an A5000 = sm_86; the campaign's e4b prefills never
   exercised this path on the 5090 boxes.) It also retroactively
   explains #78's `triton>=3.4` floor: the campaign image's 3.3.0 is
   **below the product's declared floor**, so this is a
   below-floor-install exposure, not a product bug.

2. **The box failed the GEMV anchor.** The K1-winner GEMV pair read
   108.2 µs vs the 72.9 µs K1/K4 anchor (+48%) with the noise gate
   failing, on a driver-575 host that rejects `nvidia-smi -lgc`.
   Per the burst-clocks law the box is disqualified for timing;
   its compile facts (clock-independent) stand.

## ERRATUM (2026-08-25, added after F1 Stage A)

Item 1 above says "the campaign image's 3.3.0 is **below the product's
declared floor**". That is right for the KERNEL-ONLY cycles (K3, K4,
K5), which installed gnf4 alone and pinned torch 2.7. It is WRONG as a
statement about the campaign as a whole: the e4b end-to-end cycles
(b1, b1c, b1d — including the certified 74.3 tok/s ladder) provisioned
with `pip install -e gnf4 && pip install -e e4b` and **no**
`--no-deps`, so pip upgraded torch to satisfy gnf4's own
`torch>=2.8` / `triton>=3.4`. Those runs were ALWAYS on the floor
stack. The kernel-only cycles never exercised the M-tile prefill path,
which is why triton 3.3's missing Blackwell lowering stayed invisible
until K5 asked for `tl.dot` directly.

Discovered when an F1 Stage A run on a deliberately 2.7-pinned box
died in the PREFILL path with the same gather-layout assert, on code
that had run fine throughout b1d. Nothing in K5's conclusions moves:
the ratio 1.303 refutation and the 0.4% cross-stack GEMV agreement
were both measured on the floor stack, and "the floor is exactly
right" is if anything better supported. What changes is the scope of
the claim above, and the operational rule: **provision e4b cycles
WITHOUT `--no-deps` so the declared floor resolves, and assert
`triton >= 3.4` rather than pinning torch 2.7.**

## Amended Stage A (decision map UNCHANGED)

The probe moves to the lowest **floor-compliant** stack and becomes a
two-stack cycle on one box:

- **Stack diag first**: compile-check VARIANT ∈ {0,1} × BLOCK_M ∈
  {16, 128} on torch 2.8.0 / triton 3.4 (the declared floor). If the
  floor stack still cannot compile sm_120, step to the current stack
  (torch 2.13 / triton 3.7) and record that the pyproject floor is
  understated (follow-up to bump it; not this cycle's change).
- **GEMV baseline on BOTH stacks** (same box, same binary sweep): the
  receipts-stack GEMV anchors comparability to K1–K4; the new-stack
  GEMV discloses the cross-stack delta. If the two differ by >10%,
  any absolute-µs conclusions carry that disclosure.
- **M-tile sweep on the new stack only** (it exists nowhere else),
  same grid as registered (+ BLOCK_M ∈ {16, 32} now explicit).
- **The registered ratio is within-new-stack** (M-tile best sum /
  new-stack GEMV sum). Thresholds and consequences are unchanged:
  ≤0.6 ⇒ K5-B routing knob (which ships behind the package's
  EXISTING `triton>=3.4` floor — no new requirement); 0.6–0.9 ⇒ lane
  pauses for fusion; ≥0.9 ⇒ structure refuted for existing kernels.

## Box gates added (hygiene, from attempt 1)

- Driver ≥ 580 at rent time (cu130 wheels require it).
- **GEMV anchor gate at provision**: receipts-stack GEMV pair within
  ±10% of the 72.9 µs K1/K4 anchor, else the box is destroyed and
  re-rented (codifies the burst-clocks law for this cycle).
- The bench now records driver, clocks.sm (sampled post-timing), and
  a `--stack-tag` into every report.
