# RESULTS — leg 2: the small-batch axis with the shared floor removed

**Grades `kernel/prereg_dequant_forward_floorfree.json`**, OTS-stamped pre-data
at `e9be409` (stamp `b121faa`). Adjudicated mechanically by
`bench/phase1/reduce_dequant_forward_floorfree.py`.

## VERDICT: NOT CONFIRMED — but the question leg 1 left open is answered

| | H100 80GB (sm_90) | RTX 4090 (sm_89) |
|---|---|---|
| cells live | **32 of 32** | 10 of 32 |
| F1 (primary) | **PASS** — 7 of 8 at bar, median **1.588**, band **HIT** | not adjudicable (1 live cell) |
| B1 leg-1 replication | PASS | FAIL (1 cell of 5 outside band) |
| Q1 self-pair | PASS (0% void) | **FAIL (69% void)** → device VOID |
| Q2 wiring / Q4 fidelity | PASS / PASS | PASS / PASS |
| **device** | **CONFIRMED** | **VOID** |

`FLOORFREE_CONFIRMED` requires both devices, so it is **false** — the same
shape as leg 1, where one device graded clean and the other did not.

---

## What leg 1 left open, and what this settles

Leg 1 reported its own small-batch criterion as untrustworthy and said why: at
`decode_m8` the identical `lora_delta_grouped` call **both** arms make was
54–66% of the fused arm's measured time, against cells only 1.2–1.6 ms long. A
cost added equally to both arms pins every ratio near 1.0. Leg 1 registered a
predicted median of **1.3–3.0×**, measured **1.070×**, and reported the miss
rather than adjusting anything.

This leg re-asked the question on the quantity the two kernels actually differ
in — the frozen 4-bit projection's forward and backward, with no adapter delta
on either side — **against leg 1's own band, unchanged**. On the H100:

| regime | floor-free `D_base/G_base` | with the shared floor | LoRA delta ÷ fused base arm |
|---|---:|---:|---:|
| `decode_bs1` | **1.855** (7/8 at bar) | 1.208 | **1.78×** |
| `decode_m8` | **1.588** (7/8) | 1.123 | **1.54×** |
| `decode_m32` | **1.564** (7/8) | 1.132 | **1.54×** |
| `tokbudget_2048` | **2.486** (8/8) | 2.195 | 0.28× |

**1.588 is inside 1.3–3.0.** Leg 1's miss was the shared floor, not the
kernels. Because the band was registered before leg 1 ran and carried into this
prereg unchanged, that conclusion was not available to be tuned after the fact.

Three things fell out of it:

**The adapter delta is the dominant term at small batch.** `lora_delta_grouped`
alone costs **1.54–1.78× the entire fused base projection** below 32 tokens per
expert, collapsing to 0.28× by 2048 tokens. That is real work a real trainer
pays — it just does not distinguish the two arms, so putting it inside the
timed region measured the harness.

**The one loss gets worse, not better, when the floor is removed.** gpt-oss
`gate_up` reads **0.727** floor-free against 0.787 with the floor at
`decode_m8`. The shared cost was flattering the loss as well as the wins. This
is the standing wide-N loser class, now measured without the flattery.

**Fidelity is unchanged from leg 1**, as it should be: `b_rel` G/D median 0.762
(H100) and 0.758 (4090) across all 64 cells, matching leg 1's 0.763. The arms
changed how they are *called*, not what they compute.

---

## The 4090 leg is void, and why that is a finding rather than an accident

22 of 32 cells failed the self-pair or drift band. The failures are not random
— they track **cell size**, sharply:

| median cell (`G_base`) | H100 voids | 4090 voids |
|---|---:|---:|
| 0.35 ms / 0.64 ms (`decode_bs1`) | 0 of 8 | 7 of 8 |
| 0.44 ms / 0.99 ms (`decode_m8`) | 0 of 8 | 7 of 8 |
| 0.45 ms / 0.74 ms (`decode_m32`) | 0 of 8 | 6 of 8 |
| 6.74 ms / 9.28 ms (`tokbudget_2048`) | 0 of 8 | 2 of 8 |

**A rented consumer GPU cannot hold a ±3% self-pair on sub-millisecond cells;
a datacenter card with locked clocks holds it at 0.35 ms.** That is now
twice-replicated: leg 1's run 1 voided all eight `decode_m8` cells on a 4090
while the H100 voided one, and leg 1's amendment-1 diagnosis (the first cell
after a fixture build measuring clock recovery) was itself only visible on the
consumer part.

The honest scope statement this forces: **the small-batch training axis is
measurable to this protocol's registered precision on a datacenter card and not
on a rented consumer card.** Holding it on consumer hardware would need longer
blocks, multiple reps with a different reduction rule, or locked clocks — each
a protocol change requiring its own registration, not something to reach for
after seeing a void.

The 4090's ten live cells are listed in the per-cell receipt. They read *higher*
than the H100's (`decode_m8` 2.285, `tokbudget_2048` median 3.06), consistent
with leg 1's finding that GDDR6X punishes the dequant round-trip's extra bytes
more than HBM3 does — but the device is VOID and **none of those numbers is
claimed**.

## B1: are the two legs measuring the same thing?

Registered to gate the *description*, not the number: leg 2's full arms must
reproduce leg 1's within [0.85, 1.15] at `decode_m8` on the same device class.

- **H100: PASS.** Leg 2's `D_full/G_full` median 1.123 against leg 1's 1.070.
- **4090: FAIL on one cell of five** — gemma-4 `gate_up`, leg 2 1.482 vs leg 1
  1.076, ratio 1.378. On a device that is void anyway, so it carries little,
  but it is reported rather than dropped.

There is a known reason the two legs can diverge, and it is larger at high
expert counts: leg 2 passes `expert_ids` as the **documented list form** while
leg 1 passed a CUDA tensor, costing one device sync per group in both arms. At
`decode_m8` that is 4–8 syncs; at `tokbudget_2048` it is 128, which is why leg
2's 2048 figures (2.486 median) sit well above leg 1's (1.709) and why B1 is
registered only at `decode_m8`. `E1` measures that cost directly.

---

## What this does and does not license

- **Licensed:** a statement about the fused NF4 projection versus the
  dequant-on-forward pattern **at the projection**, at small batch, on the
  H100, on synthetic weights, with adapters excluded from the timed path.
- **NOT licensed: a QLoRA training-step claim.** Removing the adapter delta is
  what makes this measurement clean and is exactly what makes it not a
  training-step number. The full-arm column speaks to a step; that is leg 1's
  story.
- **NOT licensed: a two-device claim.** One device confirmed, one void.
- **NOT licensed: quoting the 4090's live cells.** That device is VOID.
- **NOT licensed: superseding leg 1**, or any restatement of the Unsloth
  comparison, or any claim about a tuned grouped GEMM.
- **NOT licensed: calling the shared floor a defect in either kernel.** It is a
  property of how this harness attached LoRA, and it is real work.

## Receipts

`leg2/H100/`, `leg2/ADA/` — 32 cells each with gates, fidelity, energy and all
timings; `verdicts-leg2.json`; `bf16_redprec.txt` per device (see
`RESULTS-dequant-forward.md` for what that probe settled). Property suite 49/49
and this leg's 20 CPU tests green on both devices.
