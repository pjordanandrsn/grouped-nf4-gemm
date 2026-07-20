# RESULTS — mxfp4 training lane (Phase 7 / plan §Phase 2.5)

**Grades `PREREG-mxfp4-train.md` (OTS-stamped `c3a9f4e43cd12e36`, commit
`d65ee7c`). Run A complete; Run B section filled from the pod artifact.**

## Run A — gpt-oss-20b counted (A2000 dev box, free) — **ALL GATES GREEN**

Artifact `run20b/run_artifact.json` + `run20b/steps.jsonl`; code `8175e2d`
(+ CANARY_TAIL/mid-evals, same tree as stamped).

- QLoRA over the frozen native MXFP4 experts of gpt-oss-20b trains on the
  shared 12 GB A2000; held-out eval **5.0124 → 2.3342** (evals 0/10/20/30 =
  5.0124 / 2.4339 / 2.3649 / 2.3342 — fast domain-shift drop then plateau,
  as expected for a harmony-tuned model on raw wikitext); **every native
  expert tensor bit-identical to OpenAI's release before, during, and after
  training** (96/96 sha256: file data-section range == loaded bytes ==
  post-training bytes; e.g. `model.layers.0.mlp.experts.down_proj_blocks`
  = `934ef8a259b64bf3…`).
- Peak CUDA **6.18 GB** (loaded 3.98 GB) — under the 9.0 GB shared-GPU bar
  with the resident voice-tts neighbor untouched. 88.47 M adapter params
  (r=8 fp32, experts-only). ~23 s/step at seq-512 (storage CPU-pinned,
  packed bytes streamed per expert visit, decode recomputed in backward).
- Step-0 canary vs the transformers dequant arm (A4-oracle path, GPU
  layer-shuttle): loss **4.4139 vs 4.4033** (|Δ| 0.0106, bar 0.10),
  KL **0.0092**, top-1 **31/32** (bar 25/32).

| id | bar | got | verdict |
|---|---|---|---|
| A-T1 load provenance | 96/96 == file ranges | 96/96 | **GREEN** |
| A-T2 post provenance | pre == post | identical | **GREEN** |
| A-T3 eval descends | ≥ 1.0 nat | **2.678 nats** | **GREEN** |
| A-T4 canary | \|Δ\| ≤ 0.10; top1 ≥ 25/32 | 0.0106; 31/32; KL 0.0092 | **GREEN** |
| A-T5 memory | peak ≤ 9.0 GB | 6.18 GB | **GREEN** |

B1 language lock holds: 20b is proof-of-method only. The artifact of record
is the hash table (96 tensors) + the loss curve, not a VRAM number.

## Method

Phase-6 shard-read adapted to training (run_mxfp4_20b_qlora.py @ `TBD`):
dequant reference arm on CPU (canary) → per-layer native read + T1 verify →
ExpertsMxfp4LoRA patch (module gates 12/12, `d70a89f`) → CUDA move with
storage stubbed (native bytes stay CPU-pinned, stream per expert visit,
decode on device, recompute in backward) → LoRA r=8 α=16 experts-only,
AdamW 2e-4, wikitext-2 seq-512, 30 steps.

## Run B — gpt-oss-120b pod (RunPod L40S/SECURE `66b4gl1mfb3343`) — **ALL GATES GREEN**

Artifact `receipts-train/run120b/` (run_artifact.json + steps.jsonl +
run120b.log + setup.log); recipe as stamped: `--native-load` meta-init (NO
dequant materialization anywhere — native build 100 s at **61.8 GB host
RSS**), r=4 α=16 experts-only (265.4M adapter params), AdamW 2e-4,
wikitext-2 seq-512, 30 steps, seed 41.

- **The outcome headline, measured: gpt-oss-120b QLoRA at peak 9.82 GB VRAM**
  (loaded 5.44 GB) — under the ≤16 GB bar with a third to spare, on native
  MXFP4 expert bytes, ~30 s/step. Two-sided reading: 9.82 sits between the
  pre-committed suspects (not >16, not ≤8) — the projection arithmetic
  (~11 GB) was honest.
- **Step-0 fidelity: exact-chunk ppl 26.697 ∈ [26.55, 27.05]** — the stamped
  Phase-6 band, between the fused-serve 26.72 and the dequant-ref 26.75: the
  TRAINING path reproduces the stamped serve fidelity on the same fixture
  (sha `a6b7535d4427607b`).
- Held-out eval **5.1331 → 2.1152** (0/10/20/30 = 5.1331 / 2.1906 / 2.1267 /
  2.1152); **144/144 native tensors sha256 == file ranges at load AND pre ==
  post through training.**

| id | bar | got | verdict |
|---|---|---|---|
| B-T1 load provenance | 144/144 == file ranges | 144/144 | **GREEN** |
| B-T2 post provenance | pre == post | identical | **GREEN** |
| B-T3 trains | ≥ 0.5 nat | **3.018 nats** | **GREEN** |
| B-P2' step-0 ppl | ∈ [26.55, 27.05] | **26.697** | **GREEN** |
| B-T5 memory | peak ≤ 16.0 GB | **9.82 GB** | **GREEN** |
| B-OPS discipline | KEEP + evidence-teardown + 404 + ≤$5 | all held; ≈$2.7 total incl. 5 voided draws | **GREEN** |

Ops trail (all voided pods 404-verified): the fat-RAM 3090/SECURE pool was
one broken DC (CUDA-init dead ×2 — the second caught in 30 s by the new
setup CUDA gate — plus a wedge and a mapping-vanish); two thin-RAM draws
voided by the create-time RAM gate; one healthy L40S lost to watcher
friendly-fire on stale cross-pod evidence (fixed: per-pod evidence dirs).
The run landed on the 6th draw via the automated hunt ladder.

Incumbent (R9): Unsloth's 120b path ≈ 65 GB via MXFP4→NF4 conversion —
which destroys the byte identity this run certifies. The demo sentence is
earned at both scales: **"Fine-tuned; expert bytes bit-identical to
OpenAI's release — here are the hashes."**
