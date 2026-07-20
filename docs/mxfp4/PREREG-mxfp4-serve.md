# PREREG — native-MXFP4 serve: tax deletion + provenance (PRIVATE, pre-data)

**Tier: CONFIRMATORY of the Phase-1..4 local gates; the ppl/throughput/energy
numbers are pre-data. Status: DRAFT → operator review + OTS stamp. NO POD
BEFORE THE STAMP.** Code under test (private fork): `mxfp4_pack_ref.py`
(`2123da9`), `mxfp4_grouped.py` (`90d1b44`), `mxfp4_loader.py` (`90660f3`),
`mxfp4_pipelined.py` (`9ee48ff`). Standing oracle: the A4 dequant path
(`transformers _convert_moe_packed_tensors`), sha-pinned; disagreement = STOP.

## What is already proven (local, no pod — the anchors)

- Reference `dequant_mxfp4` == the A4 oracle **bit-exact** (rtol=0 atol=0),
  incl. the nibble order (2j=LOW) and the 0xFF/ldexp edge (Phase 1, 7/7).
- The fused kernel `gemm_mxfp4_grouped` == that reference within bf16 tol,
  decode + prefill + device-ids (Phase 2, 12 gates, A2000).
- Loader is byte-preserving; `sha256(arena)==sha256(file data-section)`; a
  single flipped bit breaks the receipt (Phase 3, 4/4).
- The pipelined engine reproduces the reference at **every K** (0..E), eager
  AND CUDA-graph, K = table rebuild not code path (Phase 4, 7/7).

So the pod is not asked to establish correctness — it is asked to measure the
**consequence** at 120b scale on the A4 text.

## Receipts the bands derive from (nothing else admissible; R6)

- A4 (public `bench/homelab`, S4 completion): reference-standard exact-chunk
  ppl **26.75**; NF4-conversion path **29.27** (**+9.4%**, KL 0.0657, top-1
  88.15%); ref-logits sha256[:16] `a7ca117747f8657f`; matched text 1825 tok /
  3×512 chunks; tokenizer-identity gate held.
- Native size (Phase-0 metadata): **12.61 MiB/expert** (blocks+e8m0), vs NF4
  13.1 MiB — 3.7% smaller weight traffic per expert.
- e8m0 scale is `exp2(e-127)` (one FMA-class op) vs NF4 per-64 fp32 absmax
  multiply — decode arithmetic is at-or-below NF4 cost (Phase-0 analysis).

## Predictions (falsifiers two-sided; R7 lock on every comparative sentence)

- **P1 — correctness at scale (void gate, never scores):** fused-mxfp4
  next-token logits vs the A4 dequant reference on the 120b model,
  exact-chunk, **b_rel < 2e-2 at every measured K**, eager and graphs-on.
  *Falsify:* ≥ 2e-2 ⇒ a scale/layout/epilogue bug the small-model gates
  missed; run is void, not a loss. (Local gates put this at < 3e-2 synthetic;
  120b native weights are the real test.)
- **P2 — THE TAX DELETION (headline measurement):** fused-mxfp4 exact-chunk
  ppl on the A4 text ∈ **[26.55, 27.05]** — i.e. the reference 26.75 ±
  accumulation-order noise (±0.3, ~1%; basis: fp32-accum/bf16-epilogue order
  differs from the reference's dtype path, the only source of non-identity
  once the bytes are identical). *Falsify high (> 27.05):* residual gap ⇒
  the fused path is NOT computing shipped precision; **pre-committed
  localization:** (a) decode/scale bug → the Phase-2 GPU exact-decode gate
  re-run at 120b shapes, (b) epilogue-order effect → bf16-vs-fp32 GLU
  ablation, (c) a real accumulation-order penalty larger than estimated →
  named and sized, not hidden. *Falsify low (< 26.55):* below the reference
  ⇒ a scoring artifact (like llama's sub-reference number), investigated not
  celebrated. The deletion claim requires ppl within the band AND b_rel < 2e-2.
- **P3 — throughput vs the NF4 engine (matched K, same box):** fused-mxfp4
  decode tok/s ∈ **[0.95, 1.15]×** the NF4 pipelined engine at K∈{0,16,128}
  (basis: identical mainloop/tiling; e8m0 exp2 ≤ fp32-absmax cost; 3.7%
  smaller cold traffic). *Falsify below 0.95×:* the format swap cost
  throughput — decode-cost analysis was wrong, localize to the exp2 path or
  the 32-vs-64 block loop. *Falsify above 1.15×:* the smaller traffic +
  cheaper scale helped more than modeled — quantify.
- **P4 — provenance (binary, not banded):** the 120b hash table
  (`provenance_table` over all expert tensors) generates clean, and
  `verify_arena_matches` passes for every loaded expert tensor
  (`sha256(arena)==sha256(file)`); the table is stamped alongside the run.
  *Falsify:* any mismatch ⇒ the loader is transforming bytes; STOP.

## The regime map is unchanged (R7 — the public form)

This slip measures a QUALITY axis (fidelity to native) and a byte-provenance
artifact. It does NOT reopen inference-superiority: the S3/S4 host-CPU regime
map stands verbatim, mxfp4 numbers are context rows within it, and no
sentence reads "fastest gpt-oss" or "beats llama.cpp." The demo sentence is
*"fine-tuned/served on the native bytes — expert bytes bit-identical to the
release (hashes), and fidelity is the shipped model's, not a re-quant's."*

## Protocol (Phase 6, fires only after the stamp)

Same pod class as the A4 completion (H200, for continuity with 26.75/29.27).
Load gpt-oss-120b native mxfp4 via the loader → **generate the provenance
hash table (P4)** → build the pipelined engine at K∈{0,16,128} → **P1** b_rel
vs the A4 dequant reference (reuse the sha-pinned harness) at each K, eager +
graphs → **P2** exact-chunk ppl on the identical 1825-tok text, same 3×512
chunking → **P3** decode tok/s vs the NF4 engine at matched K (both on-box).
Reuse the A4 reference-logits artifact (`a7ca117747f8657f`) as the b_rel
anchor so P1/P2 share one ground truth. Dev-box gates are already green;
budget ≤ ~2.5 h H200 (≈ $9) hardcapped; one session; destroy+verify;
artifacts land private (no public strings; PyPI still HELD).

## Pre-committed reading, plainly

If P1 green + P2 in-band + P4 clean: the native-byte path **deletes the
measured +9.4% NF4 tax** (ppl returns to the shipped model's) **and** proves
byte-identity to the release — the gate package upgrades to *native-byte
serving, tax deleted (measured), provenance stamped*. Any red amends the
private RESULTS before it informs any gate decision; nothing here touches a
public artifact regardless of outcome.
