# Docs index — what each document is, and whether it is current

This index says what each document is *for*, whether it is
the thing to read, and whether it is OpenTimestamps-anchored
(**anchored** documents are never edited in place).

Start here: [`STATUS.md`](STATUS.md) → [`claims.json`](claims.json) →
the README's confirmatory section for how the kernel numbers were made.

## Current — read these

- **[`SOLUTIONS.md`](SOLUTIONS.md)** and **[`solutions/`](solutions/)** —
  routing/usage: one page per ordinary problem (symptoms, cause, install,
  smallest example, verification, limits, evidence by claim ID). Current.
- **[`capabilities.json`](capabilities.json)** (+ [`capabilities.schema.json`](capabilities.schema.json)) —
  the machine-readable capability contract; validated in CI by
  `scripts/check_capabilities.py` against pyproject, source and the claims
  register. One status per entry, so the fp8 paged attention is two
  entries by compute path (fp8 supported; f32 open under #319). Current.
- **[`discovery-queries.json`](discovery-queries.json)** — the
  discoverability regression corpus (`scripts/check_discovery_contract.py`).
  Current.
- **[`RELEASE_NOTES_GUIDE.md`](RELEASE_NOTES_GUIDE.md)** — how the first
  paragraph of a release note is written. Current.

| doc | what it is |
|---|---|
| [`STATUS.md`](STATUS.md) | what the kernel does today, its three limits, what was retired, what is open |
| [`claims.json`](claims.json) / [`claims-schema.md`](claims-schema.md) | machine-readable register of every claim, with status and evidence path |
| [`KERNEL_CONTRACT.md`](KERNEL_CONTRACT.md) | the op signature and layout conventions (note: it schedules sm_120 as "Phase 4"; sm_120 shipped in 0.15.0 and is the primary serving target; its "storage-only asterisk" is the Gate-0 framing — the version-aware bitsandbytes boundary, 0.50.0 packed 2-D inference upstream with the grouped routed-MoE GEMM a separate contract, is on [`solutions/nf4-grouped-gemm-without-bf16-materialization.md`](solutions/nf4-grouped-gemm-without-bf16-materialization.md)) |
| [`TOLERANCE_CONTRACT.md`](TOLERANCE_CONTRACT.md) | the registered fidelity bound and test spec |
| [`PORTABILITY.md`](PORTABILITY.md) | pre-port hazard register (captured 2026-07-15; verify before a port session). No claim in it is a port result. |
| [`nvme-ceilings.md`](nvme-ceilings.md) | the per-box NVMe constant `S ≈ 3.45 GB/s` and what it implies |
| [`context-budgets.md`](context-budgets.md) | KV-cache KB/token table — **rung one (A2000) only**; full-depth confirmation pending; K3 row a declared gap. Its own text forbids promoting pending rows to the README. |
| [`K3-PROVENANCE-CHAIN.md`](K3-PROVENANCE-CHAIN.md) · anchored | chain of custody from publication hash to the multiply, `measured` tier |

## Results and pre-registrations (dated; the anchored ones are immutable)

| doc | what it graded |
|---|---|
| [`mxfp4/PREREG-mxfp4-serve.md`](mxfp4/PREREG-mxfp4-serve.md) · anchored → [`RESULTS-mxfp4-serve.md`](mxfp4/RESULTS-mxfp4-serve.md) · anchored | native-MXFP4 serving of gpt-oss-120b: ppl 26.72 vs 26.75; P1 missed as stamped (a calibration error in the stamp, per its own receipt) |
| [`mxfp4/PREREG-mxfp4-train.md`](mxfp4/PREREG-mxfp4-train.md) · anchored → [`RESULTS-mxfp4-train.md`](mxfp4/RESULTS-mxfp4-train.md) · anchored | 120b QLoRA at 9.82 GB peak, 144/144 hashes identical |
| [`mxfp4/PHASE0-seam-map.md`](mxfp4/PHASE0-seam-map.md) | the MXFP4 seam map; three STOP items carried into Phase 1 |
| [`RESULTS-k3-phase1-oracle.md`](RESULTS-k3-phase1-oracle.md) · anchored, [`RESULTS-k3-slice-roundtrip.md`](RESULTS-k3-slice-roundtrip.md) · anchored | Kimi K3 decode exact against its own reference; 48/48 arena segments identical. Stamps applied after the runs: `measured`, not `confirmed`. |
| [`PREREG-nvme-bake-determinism.md`](PREREG-nvme-bake-determinism.md) → [`RESULTS-nvme-determinism.md`](RESULTS-nvme-determinism.md) | bake determinism across two hosts, 20,480/20,480 (neither anchored, despite the prereg/results shape) |
| [`RESULTS-ikllama-ab.md`](RESULTS-ikllama-ab.md) · anchored | the ik_llama same-box A/B, in band |
| [`provenance/gptoss20b_expert_bytes.md`](provenance/gptoss20b_expert_bytes.md) · anchored | the "before" column: 144 expert-tensor hashes of gpt-oss-20b |
| [`artifacts/pypi-family-20260718/NOTES.md`](artifacts/pypi-family-20260718/NOTES.md) | the packaging gate log for that day's wheels |

## Research record — the cold-engine programme (plans and syntheses)

Working documents from an exploration whose main findings were
refutations. None is a shipped position; `STATUS.md` says what survived.

| doc | note |
|---|---|
| [`cold-engine/PHASE0-premise.md`](cold-engine/PHASE0-premise.md) | the premise test — **refuted** on the target box; self-declares incomplete (pipe term pending) |
| [`cold-engine/ARCHITECTURE-NOTES.md`](cold-engine/ARCHITECTURE-NOTES.md) | the pre-work seam map; one of its conclusions is superseded by the hybrid directive (it says so) |
| [`cold-engine/TRIBRID-ARCHITECTURE.md`](cold-engine/TRIBRID-ARCHITECTURE.md) | the Stage 3 plan; four open decisions |
| [`cold-engine/STAGE3-SYNTHESIS.md`](cold-engine/STAGE3-SYNTHESIS.md) | the verdict table across R1–R10 and gates 1–3; heavily self-corrected; **one correction still outstanding** — no read count in it should be quoted until re-run |

## What is not under `docs/` and where it lives

The serving-side kernels shipped in 0.14–0.24 — fp8 paged decode
attention and its compute modes, the int4-b32 GEMV/GEMM and calibrated
packer, the decode glue kernels, the sm_120 census, the decode anchor —
are documented in `CHANGELOG.md` and receipted in `kernel/RESULTS-*.md`
and `bench/sm120-census/`. `STATUS.md` and `claims.json` cover them; a
dedicated reference page is the next documentation gap to close.

The blind confirmatories v1–v6 and the Unsloth head-to-head are under
`kernel/RESULTS-*.md` with their `prereg_*.json` protocols. The 235B
flagship is under `bench/phase3/flagship/`. Cross-vendor projections are
top-level (`PROJECTIONS-multiarch.md`, `PROTOCOL-multiarch.md`, both
anchored) with the MI300X follow-ups beside them.
