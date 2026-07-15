# Cross-architecture sweep (11 GPUs, sm_80 → sm_120) — decode wins everywhere, zero code bugs, and the prefill gate_up regime is a low-SM floor

**2026-07-15 · Exploratory (NOT a blind confirmatory)** — a breadth shakedown to
surface cross-architecture bugs and stumbling blocks before external users do.
Eleven RunPod SECURE cards, one unmodified image (`torch 2.8.0+cu128 / triton
3.4.0 / bitsandbytes 0.49.2`), the frozen v6 tree. Each pod ran: env fingerprint
→ property suite (44) → the v6 variant matrix (`bench/phase2/v6_prefill_matrix.py`)
→ the paired 3-backend harness (`dequant_grouped fused_nf4 fused_v5loop`,
prefill_s2048 + decode_bs1). Receipts + `SHA256SUMS` in `sweep_20260715/`.

## Coverage

Four GPU generations, compute capability 8.0 → 12.0, SM counts 22 → 148:

| generation | cards (SM count) |
|---|---|
| Ampere (sm_80/86) | A100 (108), A4000 (48), A4500 (56) |
| Ada (sm_89) | RTX 2000 Ada (22), RTX 4000 Ada (48), L4 (58), L40 (142) |
| Hopper (sm_90) | H200 (132), H100 NVL (132 — see flake) |
| Blackwell (sm_100/120) | B200 (148), RTX PRO 4500 Blackwell (82) |

## Bug ledger: clean

**Zero code bugs across all 11 architectures.** Property suite **44/44** on all
ten cards that ran to completion; the v6 variant matrix and the paired harness
both exited 0 on all ten. The kernel compiles and is numerically correct from
Ampere through Blackwell on stock dependencies, unmodified.

**One environmental flake, not a defect:** H100 NVL returned `CUDA-capable
device(s) is/are busy or unavailable` during the suite and matrix (41/44 suite,
matrix rc=1) — shared-tenant contention on that specific pod. The card was
detected correctly (cc 9.0), the contention cleared before the harness ran
(rc=0), and **H200 — the same Hopper sm_90 — passed 44/44**. Recorded here because
an external user on a busy cloud instance will hit this and may misattribute it
to the kernel: `device busy` is a host condition, distinct from a compile or
numeric failure. (Prior sibling flake, for cross-reference: a container missing a
C compiler yields a *different* signature — triton can't build launcher stubs, so
fused cells skip and the suite fails ~40/44 with `Failed to find C compiler`.)

## Decode — the primary product surface — wins on every card

Fused vs the bitsandbytes dequant path, batch-1 decode, across the 8 census
cells (min–max, median):

| SM | card | cc | decode fused / dequant |
|---|---|---|---|
| 22 | RTX 2000 Ada | 8.9 | 1.22–3.58 (med 2.00) |
| 48 | RTX 4000 Ada | 8.9 | 1.80–4.54 (med 2.51) |
| 48 | A4000 | 8.6 | 1.80–4.00 (med 2.36) |
| 56 | A4500 | 8.6 | 1.62–4.21 (med 2.13) |
| 58 | L4 | 8.9 | 1.52–5.08 (med 2.46) |
| 82 | RTX PRO 4500 Blackwell | 12.0 | 1.87–5.67 (med 3.30) |
| 108 | A100 | 8.0 | 1.25–5.80 (med 2.25) |
| 132 | H200 | 9.0 | 1.10–3.91 (med 1.97) |
| 142 | L40 | 8.9 | 2.26–5.71 (med 4.57) |
| 148 | B200 | 10.0 | 1.22–4.05 (med 2.23) |

**Every census decode cell on every card is a fused win** (worst single cell
1.10× on H200). Decode down-projections at prefill likewise win on all cards ≥48
SMs (1.04–8.95×). This is the claim the product rests on, and it holds across the
rentable NVIDIA line without a single per-architecture change.

## Prefill gate_up — the one regime split, and it's a low-SM floor

The gate_up projections at prefill are the sole cell class that ever loses to the
dequant path, and the sweep localizes exactly where:

| SM band | gate_up prefill fused / dequant | verdict |
|---|---|---|
| ≤ ~26 (RTX 2000 Ada; A2000 from the v6 A2000 leg) | 0.43–0.89× on the large gate_ups | **loses** |
| 48–58 (A4000, 4000 Ada, A4500, L4) | 1.02–1.59× | wins |
| ≥ 82 (Blackwell, A100, H200, L40, B200) | 1.25–4.79× | wins, growing with SM count |

The loss is a **compute-poverty floor**, not a gradient across the line:
compute-bound prefill on ≤~26-SM cards favors cuBLAS's full-rate bf16 tensor
cores over the fused path's in-loop decode. Everything with ≥48 SMs already wins.

## The bf16-MMA variant is refuted as a fix — everywhere

The v6 matrix carried `fused_v5loop` (v5 loop) and, in the standalone matrix, the
opt-in bf16-MMA prefill variant (V3). Across all 10 cards, **V3 loses to the
shipped V1 register-LUT mainloop on every one** (V3/V1 = 0.40–0.95×, never ≥ 1.0).
The sm_86 exploratory finding — bf16-MMA costs the fidelity edge and buys no
speed — generalizes to the whole line. A device-keyed bf16 dispatch is off the
table; **V1 is the correct fused variant universally** (V1 > V0 > V3 on every
card), so the v6 ship decision needs no revision.

## What this motivates

A **device-keyed prefill dispatch floor**: on cards below ~40 SMs, route the
gate_up-prefill class to the dequant path (which wins there), mirroring v4's
tiny-cell floor. That converts the one honest prefill loss into "≥ baseline
everywhere," and the sweep gives it a real 10-card threshold rather than a
two-point guess. Registered separately as a blind confirmatory (see
`kernel/prereg_v7_confirmatory.json`); nothing here is a stamped claim.

## Teardown

All 11 pods DELETE → 404-verified by the collector; stragglers expired at the
2.5 h deadline; wedged 5090 (a fourth GeForce provisioning wedge) self-deleted.
Total sweep spend ≈ $13.
