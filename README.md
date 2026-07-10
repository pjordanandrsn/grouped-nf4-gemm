# grouped-nf4-gemm — W4A16 GEMM over fused NF4 expert stacks

**Private until Gate 2** (registered thresholds met at bs1 decode on sm_86, or a
roofline-backed narrowing recorded). Then public **with receipts** — this repo's commit
history is the provenance record; it flips visibility in place. Commons layer: destined for
the bitsandbytes ecosystem as the expert-grouped extension of the #1949 `gemm_4bit` family.

Objective: dequant-inside-the-kernel for the fused 3D expert case, so 4-bit stops being
storage-only and the honest caveat ("NF4 costs 1.2–2.3× energy; the GEMM still runs in
bf16") is deleted from the product page **because it is no longer true**.

| phase | status |
|---|---|
| 0 — contract + census + roofline + tolerance + pre-registration | **Gate 0 complete** (this commit) |
| 1 — baseline harness (A2000/3090) | staged next; runs behind Sessions 1–4 |
| 2 — Triton prototype (go/no-go gate) | — |
| 3 — integration behind Experts4bit (`E4B_GEMM_BACKEND`) | — |
| 4 — sm_120 + energy writeup (one pod session) | — |
| 5 — ecosystem landing | calendar-gated on bnb v0.50.0 release (#1949 merged, milestone set) |

Regenerate the machine-generated artifacts:

```
python3 census/make_census.py     # census/shape_census.json
python3 roofline/roofline.py      # roofline/ceilings.json
```

Docs: `docs/KERNEL_CONTRACT.md` (op contract, #1949 convention adoption, deviations),
`docs/TOLERANCE_CONTRACT.md` (fidelity ordering + comparative bounds + property-test spec),
`gemm_predictions.json` (registered thresholds and ceilings).
