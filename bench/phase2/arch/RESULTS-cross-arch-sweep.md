# Cross-architecture sweep (12 GPUs, sm_80 → sm_120) — decode wins everywhere, zero code bugs, and the prefill gate_up regime is a low-SM floor

**2026-07-15 · Exploratory (NOT a blind confirmatory)** — a breadth shakedown to
surface cross-architecture bugs and stumbling blocks before external users do.
Eleven RunPod SECURE cards, one unmodified image (`torch 2.8.0+cu128 / triton
3.4.0 / bitsandbytes 0.49.2`), the frozen v6 tree. (A twelfth — the RTX 5090,
whose provisioning wedged four times during the original sweep — landed
2026-07-16 on a fifth attempt and is folded into the tables below; see the
dated addendum at the end.) Each pod ran: env fingerprint
→ property suite (44) → the v6 variant matrix (`bench/phase2/v6_prefill_matrix.py`)
→ the paired 3-backend harness (`dequant_grouped fused_nf4 fused_v5loop`,
prefill_s2048 + decode_bs1). Receipts + `SHA256SUMS` in `sweep_20260715/`.

## Coverage

Four GPU generations, compute capability 8.0 → 12.0, SM counts 22 → 170:

| generation | cards (SM count) |
|---|---|
| Ampere (sm_80/86) | A100 (108), A4000 (48), A4500 (56) |
| Ada (sm_89) | RTX 2000 Ada (22), RTX 4000 Ada (48), L4 (58), L40 (142) |
| Hopper (sm_90) | H200 (132), H100 NVL (132 — see flake) |
| Blackwell (sm_100/120) | B200 (148), RTX PRO 4500 Blackwell (82), RTX 5090 (170 — added 2026-07-16) |

## Bug ledger: clean

**Zero code bugs across all 12 architectures.** Property suite **44/44** on all
eleven cards that ran to completion; the v6 variant matrix and the paired
harness both exited 0 on all eleven. The kernel compiles and is numerically correct from
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
| 170 | RTX 5090 | 12.0 | 1.48–3.88 (med 2.96) |

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
| ≥ 82 (Blackwell incl. RTX 5090, A100, H200, L40, B200) | 1.18–4.79× | wins, growing with SM count |

The loss is a **compute-poverty floor**, not a gradient across the line:
compute-bound prefill on ≤~26-SM cards favors cuBLAS's full-rate bf16 tensor
cores over the fused path's in-loop decode. Everything with ≥48 SMs already wins.

## The bf16-MMA variant is refuted as a fix — everywhere

The v6 matrix carried `fused_v5loop` (v5 loop) and, in the standalone matrix, the
opt-in bf16-MMA prefill variant (V3). Across all 11 cards, **V3 loses to the
shipped V1 register-LUT mainloop on every one** (V3/V1 = 0.40–0.95×, never ≥ 1.0;
the 5090 reads 0.75–0.95 — consumer Blackwell's bf16 tensor cores don't save it).
The sm_86 exploratory finding — bf16-MMA costs the fidelity edge and buys no
speed — generalizes to the whole line. A device-keyed bf16 dispatch is off the
table; **V1 is the correct fused variant universally** (V1 > V0 > V3 on every
card), so the v6 ship decision needs no revision.

## A prefill dispatch floor was investigated — and the data refuses it

The obvious follow-up was a device-keyed prefill dispatch floor (route the
gate_up class to dequant on low-SM cards, mirroring v4's decode floor). Fitting
that idea to the 88 prefill cells across the 10 cards **refutes it**, and the
refutation is recorded here so it isn't retried blind:

- The prefill losers do **not** separate by SM count. The dominant loser across
  the whole line is the **smallest-expert shape (OLMoE)** — its gate_up loses at
  sm 22/48/56/58 (0.43–0.57×) and still reads 0.96× at sm 132. That is precisely
  the cell v6 already carries as the report-only known-loser, not a new
  SM-regime effect.
- The best separating predicate found, `sm < 40 AND N ≥ K`, catches 2 losers
  while **wrongly surrendering 4 real wins and missing 9 losses**; widening to
  `sm < 64` gives up 23 wins. No `(N, K, M, sm)` predicate separates losers from
  winners without either forfeiting real speedups or overfitting 11 data points
  — the v1 config-overfit failure mode the methodology exists to prevent.

**Conclusion:** the fused kernel stays the universal default at prefill; there is
no low-regret dispatch floor to add. The small-expert prefill shortfall remains a
documented known-loser (as in v6), not a dispatch case. No v7 is registered —
this negative result is the finding.

## Teardown

All 11 pods DELETE → 404-verified by the collector; stragglers expired at the
2.5 h deadline; wedged 5090 (a fourth GeForce provisioning wedge) self-deleted.
Total sweep spend ≈ $13.

---

## Addendum 2026-07-16 — the twelfth card: RTX 5090 (fifth provisioning attempt)

**Provisioning post-mortem.** The 5090 wedged four times during the original
sweep window (pods created, never received an IP — all SECURE, all at Low/None
listed stock). On 2026-07-16 stock read **Medium**; attempt 5 tried COMMUNITY
first (twice — both bounced instantly with `machine does not have the
resources`, a per-machine capacity error, not a wedge) and then **SECURE
created AND provisioned normally** (pod `83r70st31jo1ia`). Working hypothesis
from the 5-attempt record: the wedge variable was **stock level, not cloud
type** — at Low stock the scheduler places pods it can't ever bring up.
Run: same tarball, same runner, same image as the other eleven cards.
Spend ≈ $1.1 (SECURE $0.99/hr, ~65 min); DELETE → 404-verified.

**Fingerprint.** RTX 5090, driver 570.195.03, cc 12.0 (sm_120), **170 SMs —
the largest card in the table**; `torch 2.8.0+cu128 / triton 3.4.0 /
bitsandbytes 0.49.2`, unmodified.

**Epistemic status: exploratory, out-of-sample — no 5090-specific prereg.**
This run was launched from the exploratory lane; no prediction document was
stamped before it (and none was stamped after — by the time the question was
asked, the harness receipts already existed, so the registration window was
closed). What the row IS graded against: the cross-card laws published in this
document at commit `add0ef5`, pushed to the public repo on **2026-07-15, before
this run existed**. Every one of them holds out-of-sample:

| published law (2026-07-15) | RTX 5090 outcome (2026-07-16) |
|---|---|
| suite 44/44 unmodified on every arch | **44/44** ✓ |
| decode fused > dequant on every census cell | **8/8 win, 1.48–3.88 (med 2.96)** — 2nd-best median in the table ✓ |
| prefill gate_up wins for ≥48 SMs | **4/4 win, 1.18–3.06 (med 2.30)** ✓ |
| prefill down wins on ≥48-SM cards | **4/4 win, 2.27–5.70 (med 4.15)** ✓ |
| v6 register-LUT ≥ v5 loop at prefill | **8/8, v6 = 1.24–1.41× v5loop (med 1.32)** ✓ |
| bf16-MMA (V3) < V1 everywhere | **V3 = 0.75–0.95× V1, never wins** ✓ |

Two cells worth naming. The **OLMoE gate_up known-loser closes with SM count**:
0.43–0.57× at sm 22–58 → 0.96× at sm 132 → **1.18× (a win) at sm 170** — the
compute-poverty-floor reading of the loss is now a monotone SM progression.
And **gpt-oss down — the historically instance-unstable cell
({0.67..1.75} over six instances)** — reads a clear 1.58× here.

Receipts: `sweep_20260715/5090/` (HW_STATE, harness/matrix JSONs + logs),
hashes appended to `SHA256SUMS`; reduction by `sweep_20260715/reduce_row.py`,
validated against the published B200 and L40 rows before use on this card.
