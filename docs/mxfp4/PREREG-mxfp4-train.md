# PREREG — mxfp4 training lane (Phase 7 / plan §Phase 2.5): 20b counted run + 120b pod run

**Pre-data for the counted runs.** Bars calibrated on the 3-step smoke
(artifact `smoke20b/run_artifact.json`, dev box, 2026-07-19) per the spin-5
lesson: calibrate on dev evidence, then freeze. Stamped before either counted
run; the 120b pod fires only after the 20b counted run grades GREEN
(operator GO 2026-07-19, conditional on that green). PRIVATE fork only.

## Claim under test

LoRA adapters train over the FROZEN native MXFP4 expert bytes of gpt-oss —
decode recomputed in backward, nothing dense retained — and the expert bytes
are **bit-identical to OpenAI's release before, during, and after training**
(sha256 file-range == loaded == post-training, per tensor). B1 language lock:
20b is dev/proof-of-method (never a headline — Unsloth serves the 16 GB user
outcome via MXFP4→NF4 conversion, which destroys byte identity);
**gpt-oss-120b QLoRA ≤ 16 GB is the outcome headline** once measured.

Anchors: Phase-1 oracle decode (bit-exact); Phase-7 module gates 12/12
(`d70a89f`); smoke-5 full-path evidence: 96/96 load provenance, pre==post
hashes, canary Δloss 0.0228 / KL 0.0015 / top1 7/8, eval 4.948→2.473 in 3
steps, peak 6.18 GB, ~23 s/step.

## Run A — gpt-oss-20b counted (A2000 dev box, free)

Recipe frozen at stamp: `run_mxfp4_20b_qlora.py` @ the stamped commit, loop
path, dequant arm via GPU layer-shuttle canary (CANARY_TAIL=32 positions),
storage CPU-pinned, LoRA r=8 α=16 fp32 experts-only, AdamW 2e-4, wikitext-2
seq-512, **30 steps**, 8 held-out eval chunks (evals at 0/10/20/30), seed 41,
`--canary-tokens 128`.

| id | gate | bar (calibrated) | kind |
|---|---|---|---|
| A-T1 | provenance, load | 96/96 sha256(loaded) == sha256(file range) | HARD (mismatch = STOP) |
| A-T2 | provenance, post | module hash table pre == post (blocks+scales+biases) | HARD |
| A-T3 | trains | eval@30 ≤ eval@0 − 1.0 nat (smoke: −2.47 in 3 steps); all step losses finite | HARD |
| A-T4 | canary parity | \|our − ref\| loss ≤ 0.10 (smoke 0.023); top1 ≥ 25/32 (smoke 7/8 at n=8); KL recorded (smoke 0.0015) | catastrophic guard; magnitudes informational (P1 lesson) |
| A-T5 | memory | peak CUDA ≤ 9.0 GB (smoke 6.18; shared-GPU neighbor budget) | HARD ceiling |

## Run B — gpt-oss-120b pod (fires only on Run A GREEN)

Recipe frozen at stamp: same runner `--native-load` (meta-init + shard load —
no dequant materialization; the dequant path measured 41.6 GB RSS at 20b and
~4× that at 120b does not fit a pod), `--r 4` (E=128 quadruples adapter
count; r=8 fp32+Adam ≈ 8.5 GB VRAM alone), 30 steps, seq-512, evals 0/10/20/30,
`--ppl-text fixtures/prompts_ppl.txt` (sha256[:16] `a6b7535d4427607b`, the
Phase-6 fixture) with `--ppl-band 26.55,27.05`.

| id | gate | bar | kind |
|---|---|---|---|
| B-T1 | provenance, load | 144/144 (36 layers × 4) sha256 == file ranges | HARD |
| B-T2 | provenance, post | pre == post | HARD |
| B-T3 | trains | eval@30 ≤ eval@0 − 0.5 nat; finite | HARD |
| B-P2' | step-0 fidelity | exact-chunk ppl ∈ **[26.55, 27.05]** (the stamped Phase-6 band; the serve run measured 26.72 fused / 26.75 dequant-ref) | HARD (outside band = STOP — the training path disagrees with the stamped serve fidelity) |
| B-T5 | memory | **peak CUDA ≤ 16.0 GB** — THE headline bar (projected ≈ 11 GB: ~5 non-expert bf16 + 4.2 r=4 adapters+Adam + ~1.5 activations/transients) | HARD; two-sided reading pre-committed below |
| B-OPS | discipline | KEEP-flag at create; watcher teardown on evidence-complete; hard deadline 2.6 h < sweeper 3 h cap; 404-verified; cost ≤ $5 hard cap | HARD |

Pre-committed readings (both directions): B-T5 over 16 GB ⇒ adapter/optimizer
arithmetic wrong or activation checkpointing not engaging — localize before
any re-run; B-T5 far under (≤ 8 GB) ⇒ the ≤16 GB headline understates and the
claim may name the measured number. B-P2' below band ⇒ impossible-good,
suspect harness (chunking/tokenizer drift); above band ⇒ decode or epilogue
divergence in the training path — STOP, A4 oracle arbitration.

Falsification (both runs): T1/T2 failure means the training path writes or
re-packs storage — the provenance thesis dies. T3 failure at bar means
adapters do not learn through recompute — the training thesis dies.

## Deliverables

Per run: `steps.jsonl`, `run_artifact.json` (canary/ppl, evals, provenance
tables incl. per-tensor hashes, config, VRAM), RESULTS-mxfp4-train.md grading
every bar, receipts + OTS stamps alongside this slip.
