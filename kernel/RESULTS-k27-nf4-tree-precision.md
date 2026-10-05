# K27 — results: **TF32_PATH**. K25 with the select tree runs at 0.448 (Granite) and 0.502 (OLMoE) of the served NF4 GEMM's time at the served kernel's weight precision, with the same error

Registration: `kernel/PREREG-k27-nf4-tree-precision.md` (#434, `908a2ca`). Issue: experts4bit-qlora#564. Runner:
experts4bit-qlora `bench/k27/` (#833).

**Verdict by `k27_bench.verdict`: TF32_PATH.** The next lane reads K25-tree in TF32 end to end on both families, as a
default candidate.

```
K27_VERDICT TF32_PATH: granite tree32/served 0.448 err 1.000, olmoe tree32/served 0.502 err 1.000; granite tree16/served 0.317, olmoe tree16/served 0.325 -- next: K25-tree in TF32 end to end on both families (<= 0.6, err <= 1.1)
```

## The arms (ms per decode step on steps 4–7; tree arms at their best plan, selected on steps 0–3)

| arm | Granite | OLMoE | rms error vs fp64 (Granite / OLMoE) |
|---|---:|---:|---:|
| `pair16`, K25 as P92 ran it (default plan) | 6.070 | 9.989 | 3.426e-4 / 4.425e-4 |
| `tree16`, bf16 (best plan 32 / 128 / 4 / 2) | 1.832 | 3.139 | 3.426e-4 / 4.425e-4 |
| **`tree32`, TF32 (best plan 32 / 64 / 4 / 3)** | **2.588** | **4.847** | **2.528e-4 / 3.281e-4** |
| `served` (`_gemm_nf4_grouped`) | 5.780 | 9.660 | 2.528e-4 / 3.281e-4 |

- **At the served precision, the tree keeps most of its speed.** `tree32 / served` is 0.448 and 0.502, under the 0.60
  bar. Its rms error equals the served kernel's to four digits (ratio 1.000 in both families). The bf16 arms read
  1.35×.
- **`tree16` is bit-equal to `pair16`** on both projections. At its best plan it runs at 0.317 / 0.325 of the served
  kernel: faster than K26's default-plan reading (0.39 / 0.40), because KC 128 beat KC 256 here.
- **TF32 costs the tree about 1.4–1.5×** against bf16 (2.588 vs 1.832; 4.847 vs 3.139). Its best plan is the smallest
  K chunk (KC 64, 3 stages). Every plan of both arms ran; the select-step medians are in `k27.json`.

## Against the predictions

| prediction | outcome |
|---|---|
| `err(tree32) / err(served)` 0.95–1.05 | held: 1.000 / 1.000 |
| `tree32 / served` 0.45–0.65 | held: 0.448 (at the band's floor) / 0.502 |
| `tree16 / served` 0.35–0.45 | **0.317 / 0.325, below the band**: the plan sweep found KC 128 |
| TF32_PATH at about 55 % | TF32_PATH |

The A2000 rehearsal read `tree32 / served` 0.63 / 0.64, and this card read 0.45 / 0.50. The scaling the PREREG named
(about 0.53) was in the right direction and conservative.

## The run

- **`k27-5090-3`:** one RTX 5090 (driver 595.91.07, power.limit 545 W) on an AMD host. grouped-nf4-gemm at
  `908a2ca`, torch 2.8.0, triton 3.4.0.
- **Premise:** K25's contract compiled on the card, 30/30.
- **Cost:** $0.0269, destroyed and proven absent.
- **Two earlier attempts:**
  - `k27-5090-1` was refused by the launcher's bandwidth pre-flight (26.2 MB/s < 40; $0.0166);
  - `k27-5090-2`'s guard did not arm (a Vast API 429; $0.0004).

  Both were destroyed and proven absent. **Lane cost: $0.0439.**

## What follows

The registered pointer: **K25-tree in TF32, end to end on both NF4 families**.
- In experts4bit-qlora, the K25 route runs at the served precision (`dot_bf16=False`) at this lane's TF32 plan.
- A P-lane on P92's design re-reads Granite and OLMoE: speed at B=16 and B=1, and two-text K8.
- If both read LICENSED, the route's default moves.

## Receipts

[`receipts-k27/5090/`](receipts-k27/5090/) holds `k27.json`, summary, forensics, versions, the bench and contract logs,
the teardown proof and `SHA256SUMS`.

## Erratum (2026-10-05): the speed band and the verdict lean were seeded by A2000 timings

The PREREG's "What was seen before this page" records two A2000 runs: the bench's `--quick` correctness pass and the
runner's full-mode rehearsal (experts4bit-qlora `bench/k27/`). Both are labelled not a reading, but registered
statements draw on their timings:
- the `tree32 / served` band, 0.45–0.65, is justified partly by OLMoE being the tighter family "as on the A2000";
- the verdict lean, TF32_PATH at about 55 %, scales the rehearsal's A2000 TF32 ratios by the 5090-to-A2000 bf16 ratio;
- the section reports which verdict the A2000 runs would have read.

Under the testbed policy (the A2000 is a correctness-only testbed; every timing, ratio or band basis comes from rented compute on the target card; grouped-nf4-gemm#475, `docs/audits/a2000-timing-2026-10-05.md` §2), an A2000 timing may not seed a band or a prediction, whatever its label.

What it changes:
- **The registration stands as stamped.**
- **The verdict does not rest on the A2000.** TF32_PATH comes from the rule's thresholds (`tree32 / served` ≤ 0.60 and
  `err(tree32) / err(served)` ≤ 1.10 in both families), read on the 5090: 0.448 / 0.502, error ratio 1.000 / 1.000.
- **The paragraph under "Against the predictions" sets the A2000 rehearsal's ratios beside the 5090's** and calls the
  scaled prediction right in direction and conservative. That comparison is not speed evidence, and scaling an A2000
  ratio is not a prediction method; the 5090 rows are this lane's only speed reading.
- **The error band** (`err(tree32) / err(served)` 0.95–1.05) rests on the A2000's numerics, which the policy allows; it
  held at 1.000 on the 5090. **The `tree16 / served` band** came from K26's 5090 read and is unaffected.
- **For later lanes:** bands come from a 5090 or H100 reading, or from a rented microbench of the component on the
  target card.
