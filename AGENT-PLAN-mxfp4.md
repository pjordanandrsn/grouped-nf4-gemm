# AGENT-PLAN-mxfp4 — flattened + corrected (2026-07-16)

**Status: DRAFT for operator review. NOT OTS-stamped** (stamps are immutable;
stamp after review per the addendum-hold rule). **PRIVATE-fork only** — this
document contains competitive strategy and must not reach the public remote.

**Provenance.** The base `AGENT-PLAN-mxfp4.md` was never committed to any repo
— it existed only by reference. Addendum 1 (2026-07-15) and Addendum 2
(2026-07-15, adversarial viability audit) were delivered in chat and held
uncommitted at operator instruction because the forward-only filing rule
("all hashes in the commit") was unsatisfiable without a committed base. This
file **flattens all three into the single canonical plan** at operator
instruction (2026-07-16: "flatten the mxfp4 plan into one corrected doc").
Corrections applied during flattening are marked **[CORRECTED]**; base-plan
text that existed only by reference is marked **[RECONSTRUCTED]** or **[LOST]**.
The addenda originals are preserved verbatim in Appendices A/B. Forward-only
discipline (R2) resumes from this file once stamped.

---

## 1. Objective and headline (A1 superseded by B1)

- **The demo centerpiece is the provenance receipt, not a VRAM number.**
  SHA-256 every frozen expert tensor before training and after N steps,
  publish the hash table, stamp it (OTS) alongside the run ledger. The demo
  sentence: *"Fine-tuned; expert bytes bit-identical to OpenAI's release —
  here are the hashes."* No competing framework currently produces this
  artifact (Unsloth's path converts MXFP4→NF4; transformers dequantizes to
  bf16 — both destroy byte identity).
- **gpt-oss-20b is a development target only.** PROHIBITED as a headline or
  launch claim in any form resembling "fine-tune gpt-oss on a 16 GB card" —
  Unsloth ships that user outcome at 12.8–14 GB. 20b appears publicly only as
  the provenance-receipt proof-of-method.
- **gpt-oss-120b QLoRA on ≤16 GB is the outcome headline**, projected tier
  until measured (Unsloth's 120b path needs ~65 GB). If the 120b run doesn't
  exist by the gate, the launch leads with provenance + the Qwen/NF4 story,
  and 120b stays a stamped projection.
- The training claim is the differentiator, not inference: for pure inference
  gpt-oss (~5.1B active/tok) is llama.cpp's best case and its GGUFs preserve
  the native MXFP4 blocks. Inference numbers are measured and published as
  context rows, never as the headline (A1/R7).

## 2. Standing rules (consolidated)

- **R1 — anchor before extend.** Every new surface reproduces a known-good
  anchor before extending it. A structural harness change to accommodate a
  new format is a red flag to report, not a workaround to apply.
- **R2 — forward-only; stamped docs are immutable.** This file is the new
  base. Corrections to it after stamping arrive as addenda, never edits.
- **R3 — tier language.** "projected / untested / port target" until a run
  exists; confirmed numbers cite their receipts.
- **R4 — escalation boundary.** Purchases, signups, spend above the approved
  budget, and public posts require the operator's word.
- **R5 — sprint mode and the gate (A3).** See §4.
- **R6 — sources or silence.** No constants from memory; every external number
  carries a citation or is marked unverified.
- **R7 — no inference-superiority claims for gpt-oss-class sparsity** (A1):
  never "fastest gpt-oss inference," never "beats llama.cpp"; any public
  gpt-oss comparison acknowledges the llama.cpp baseline explicitly.
- **R7-ext / B4 — kernel phrasing lock.** Sanctioned kernel description:
  *"fused grouped GEMM on bitsandbytes NF4-packed weights with host streaming,
  on consumer GPUs."* Bare "fused 4-bit MoE GEMM" is prohibited as a novelty
  claim — Marlin MoE exists and a skeptical reviewer will cite it.
  Marlin/Machete are acknowledged by name wherever the kernel's novelty is
  asserted; the gpt-oss K=N=2880 shape incompatibility may be cited (from
  COMPETITIVE.md) as a concrete coverage difference, not as a dunk.
- **R8 — [LOST with the uncommitted base.** Both addenda list R8 as
  "unchanged" but its text never existed in a committed file. Operator:
  restate or retire at review.]
- **R9 — name the incumbent (B2).** Any claim adjacent to an incumbent
  capability names the incumbent in the same breath. Enforced via
  `docs/COMPETITIVE.md` (§5): every public comparative or "first/only" claim
  must cite its COMPETITIVE.md entry. **No entry, no claim.**
- **R10 — the watchlist (B6).** `docs/WATCHLIST.md` checked at the start of
  every agent session on this lane; any trigger = stop work on the affected
  phase and report before proceeding. See §6. **[Recorded, NOT armed — no
  autonomous session-start checks exist yet, per operator 2026-07-15.]**

## 3. Phases

- **Phase 0 — stamped projections.** QLoRA peak-VRAM rows for gpt-oss-20b and
  -120b (derived from the existing per-layer offload arithmetic), each with a
  falsification criterion; gpt-oss rows carry a **competitive-baseline
  column** (llama.cpp CPU-MoE for inference; transformers bf16-dequant for
  training; Unsloth NF4-conversion for QLoRA) so every projected number sits
  next to the incumbent it must be read against (A4/B1, R6 citations).
  Stamped before any run. *Status: not started (the multiarch Phase 0 docs
  in-repo are a separate, already-stamped set).*
- **Phase 1 — transformers-dequant anchor.** The ground-truth gate: reproduce
  the transformers MXFP4→bf16 dequant path and validate our MXFP4 decode
  against it bit-for-bit before any training or kernel work. [Per A1
  "unchanged"; full base text **[RECONSTRUCTED]** from references — re-derive
  the exact gate criteria at activation.]
- **Phase 2 — native MXFP4 load/decode path.** **[RECONSTRUCTED]** Bring
  packed MXFP4 expert tensors into the e4b stack natively (storage + decode),
  validated against the Phase-1 anchor. **Honest scoping flag (2026-07-15
  hold-note finding): `ExpertsMxfp4` does not exist** — `experts4bit-qlora`
  today has the NF4 path (`ExpertsNbit`) only, so A2's "existing path gains
  the MXFP4 decode" is **net-new work**, not an extension of shipped code.
- **Phase 2.5 — native-weight QLoRA demo (the gate deliverable, A2).**
  1. Training path over native packed MXFP4 experts: recompute-in-backward
     autograd gains the MXFP4 decode; LoRA adapters train over frozen native
     experts. The NF4 seed-matched A/B methodology is reused **verbatim** —
     same harness, same reducers, same `ab-telemetry/` layout (R1).
  2. Dev target: gpt-oss-20b on the A2000. Deliverables: loss curve
     (held-out eval), loaded/peak VRAM, per-step JSONL, golden-canary pin,
     and the **identity gate elevated to artifact** (B1): the SHA-256
     pre/post hash table of frozen expert bytes vs the shipped checkpoint.
  3. Stretch: gpt-oss-120b expert-offload fine-tune, disk and time
     permitting; its row stays projected until the run exists.
- **Phase 3 — kernel MXFP4 decode path. POST-GATE LOCKED:** no Phase-3
  session may be scheduled before 2026-08-01 (A3). The training headline does
  not require the fused kernel.

## 4. Priorities, dates, and the gate (A3)

**All dates recorded, NOT enforced** (operator 2026-07-15: "record but don't
enforce" — no autonomous reminders armed).

- **Still outranks this plan:** issue #22 follow-through; the private
  application track (EV, NVIDIA Inception, Modal, SBIR Project Pitch) — these
  are invisible to competitors and run on reviewer calendars. Recorded
  blocking-reminder date if untouched: **2026-07-18**.
- **No longer blocks this plan:** HN post, Daniel note, GPU MODE surfacing —
  deliberately deferred to the gate.
- **The gate: 2026-08-01, hard.** Whatever exists ships. Minimum launch
  package = the existing Qwen/NF4 story; target package adds Phases 0–2 and
  the 2.5 training demo. The date may not slip for readiness reasons;
  past-operator has pre-decided the exit. Near the gate the agent's job is
  freeze-and-polish, not feature push.
- **[CORRECTED 2026-07-16]** Any "Daniel opener" or outreach must **NOT**
  anchor on the bnb-4bit-MoE-unsupported wedge — that wedge has collapsed
  (§6, trigger 2). Anchor on the provenance niche instead.

## 5. `docs/COMPETITIVE.md` (R9 deliverable)

Dated, cited entries the agent **verifies (R6) rather than trusting the
audit**. Claimed by the audit, **UNVERIFIED as of 2026-07-16** except where
noted — these become entries with citations when the lane goes live, and are
not treated as fact until then:

- **Unsloth**: gpt-oss-20b QLoRA 12.8–14 GB via MXFP4→NF4 conversion; 120b at
  ~65 GB; MoE 4-bit QLoRA publicly conceded unsupported **[stale — see §6
  trigger 2 correction]**; MXFP4 backward W_TRANSPOSE unimplemented.
- **Marlin/Machete + vLLM fused Marlin MoE**: fused 4-bit MoE GEMM for
  GPTQ/AWQ/MXFP4/NVFP4; Ampere/Hopper lean; known K=N=2880 alignment failure
  on gpt-oss shapes.
- **ik_llama.cpp / ktransformers / llama.cpp**: Qwen3-235B ~7.4 tok/s
  community benchmark (quant-precision caveat: IQ3_K vs NF4); gpt-oss-120b
  CPU-MoE offload ~28–30 tok/s on high-end desktops.
- *Already verified in-repo (multiarch lane):* bnb ROCm = preview, source
  build required (PyPI wheel CUDA-only); bnb XPU preview-grade; Triton on
  RDNA4 ~30–50% behind hand-HIP; Strix Halo ~212–215 GB/s measured (citations
  in `docs/PORTABILITY.md` + `PROJECTIONS-multiarch.md`).

**No throughput-leadership claims for Qwen3-235B** (B3): the sanctioned
differentiators are exact-checkpoint-bytes fidelity (4-bit NF4 vs ~3-bit IQ
quants — a quality-per-bit claim with the precision caveat stated),
energy-per-token where measured, and the receipts methodology itself — never
raw tok/s supremacy. Two further B3 provisions: (a) the incumbent-baseline
column extends to **all** projection rows (multiarch P0 and mxfp4 P0), each
with a quant-precision field so numbers are never compared across unequal
bit-rates without saying so; (b) optional gate-package stretch item, tightly
scoped: one same-box incumbent A/B (ik_llama.cpp on the operator's own
hardware, one model, one config, quant difference documented) to pre-empt the
first hostile launch-thread question — drops if it threatens the gate date.

## 6. `docs/WATCHLIST.md` (R10 — recorded, not armed)

1. **Unsloth ships the MXFP4 backward pass** (W_TRANSPOSE or equivalent) →
   byte-provenance novelty gone; the B1 headline dies; escalate for re-plan.
2. **unslothai/unsloth #4032.** **[CORRECTED 2026-07-16 — the addendum's
   premise is refuted by the authoritative API.]** B6.2 asserted #4032 is
   "OPEN with a WIP label" and claimed the earlier "Closed" reading was a
   render artifact. Live `gh issue view 4032` (checked 2026-07-15 and re-
   checked 2026-07-16): **state=CLOSED, stateReason=COMPLETED, closed
   2026-06-18**, label WIP, title "Bnb4bit support for MoE models on
   transformers v5." Per the addendum's own branch logic, closed-by-shipping
   ⇒ **the bnb-4bit-MoE wedge has collapsed; the pivot to B1's provenance
   niche is SELECTED** (this plan's headline already reflects it). Nuance
   from the close context: what shipped is **dequant-path MoE support**
   (capacity/correctness, three architectures: OLMoE / Qwen3-MoE / Gemma-4
   text; tracks bnb#1849) — **not** a fused grouped kernel — so trigger 4
   below has NOT fired and the kernel's niche is untouched.
3. **bitsandbytes v0.50.0 releases** → immediately audit its contents for
   grouped/MoE 4-bit kernels; also fires the PR #1965 "congrats + ready"
   comment per the existing Monday-gate rule.
4. **Marlin/vLLM or bnb adds bnb-NF4 grouped-MoE GEMM** → the kernel's format
   niche closes; host-streaming becomes the sole differentiator; re-plan
   kernel positioning. *[Status 2026-07-16: not fired — see trigger 2.]*

## 7. Multiarch corrections (B5) — DISCHARGED

All three B5 items are already applied in-repo as of 2026-07-16, so no
erratum is needed:

- **Intel below AMD in Phase-2 ordering** + bnb ROCm/XPU status + the
  Triton-maturity band note: in `docs/PORTABILITY.md` (commit `d5fc4c7`).
- **Strix Halo constant**: the stamped `PROJECTIONS-multiarch.md` already
  carries "256 GB/s theo, ~215 meas" with citations and a reality-check
  section — it was written measured-first per R6.
- **PCIe 5.0 ×16 on RDNA4**: confirmed; row stands.

*(Multiarch execution state, for context: Phase 0 stamped + public; Phase 1
scaffolding + SYCL M1/M2 cross-vendor on the private fork; MI300X confirmatory
armed on stock; Arc capacity-rejected. See memory + `docs/PORTABILITY.md`.)*

## 8. Out of scope (unchanged from the base, as referenced)

- PR #1965 scope freeze (parked until bnb v0.50.0 per its own gate).
- Blackwell FP4 paths.
- **[LOST:** the remainder of the base out-of-scope list was never committed;
  restate at review if anything else belonged here.]

---

## Appendix A — Addendum 1 verbatim (2026-07-15, superseded by this file)

> # Addendum 1 — AGENT-PLAN-mxfp4.md
>
> **Date:** 2026-07-15 · **Applies to:** AGENT-PLAN-mxfp4.md as committed · **Method:** forward-only (R2); the base plan is not edited. Where this addendum conflicts with the base plan, the addendum governs.
>
> ## A1 — Headline redefinition: training, not inference
>
> The base plan's "Why" names decoding gpt-oss-120b as the headline. **Superseded.** For pure inference, gpt-oss (~5.1B active/token) is llama.cpp's best case — its CPU-MoE offload path serves it respectably, and their GGUFs preserve the native MXFP4 blocks. The differentiated claim is **fine-tuning**: transformers' path dequantizes MXFP4 to bf16 to train, blowing the memory budget; recompute-in-backward over the *native packed experts* does not.
>
> **New headline sentence (projected tier until measured):** *"QLoRA fine-tune of gpt-oss on a 12–16 GB card, over the exact released MXFP4 bytes."*
>
> **R7 extension:** the agent may not write "fastest gpt-oss inference," "beats llama.cpp," or any inference-superiority claim for gpt-oss-class sparsity. Any public-facing comparison for gpt-oss must acknowledge the llama.cpp inference baseline explicitly. Inference numbers are still measured and published (ledger rows, projection confirmatories) — they are context, not headline.
>
> ## A2 — New Phase 2.5: native-weight QLoRA demo (the gate deliverable)
>
> Insert between Phase 2 and Phase 3:
>
> 1. **Training path over `ExpertsMxfp4`:** the existing recompute-in-backward autograd path gains the MXFP4 decode; LoRA adapters train over frozen native experts. Anchor (R1 style): the NF4 seed-matched A/B methodology is reused verbatim — same harness, same reducers, same `ab-telemetry/` layout. A structural change to the harness to accommodate MXFP4 is a red flag, not a workaround (report first).
> 2. **Dev target:** gpt-oss-20b on the A2000. Deliverables: loss curve (held-out eval), loaded/peak VRAM, per-step JSONL, golden-canary pin. **Identity gate:** the frozen expert bytes after N steps are bit-identical to the shipped checkpoint bytes (the "exact released bytes" claim, in test form).
> 3. **Stretch:** gpt-oss-120b expert-offload fine-tune, disk and time permitting. Its row stays projected until the run exists.
> 4. Phase 0 gains matching projection rows: predicted peak VRAM for 20b/120b QLoRA (derived from the existing per-layer offload arithmetic), each with a falsification criterion, stamped with the rest.
>
> ## A3 — R5 revision: sprint mode and the August 1 gate
>
> The base plan queued this work behind the HN post and the Daniel note. **Superseded by the operator's sprint decision:**
>
> - **Still outranks this plan (unchanged):** issue #22 follow-through; the private application track (EV, NVIDIA Inception, Modal, SBIR Project Pitch) — these are invisible to competitors and run on reviewer calendars; the agent surfaces them as blocking reminders if untouched by 2026-07-18.
> - **No longer blocks this plan:** HN post, Daniel note, GPU MODE surfacing — deliberately deferred to the gate.
> - **The gate:** 2026-08-01, hard. Whatever exists ships: minimum launch package is the existing Qwen/NF4 story; target package adds Phases 0–2 and the 2.5 training demo. The gate date may not slip for readiness reasons; past-operator has pre-decided the exit. The agent's job near the gate is freeze-and-polish, not feature push.
> - **Phase 3 (kernel decode path) is explicitly post-gate.** The training headline does not require the fused kernel; no Phase 3 session may be scheduled before 2026-08-01.
>
> ## A4 — Projection-table amendments (Phase 0)
>
> - gpt-oss rows gain a **competitive-baseline column** (llama.cpp CPU-MoE for inference; transformers bf16-dequant path for training) so every projected number sits next to the incumbent it must be read against (R6 citations required).
> - The corrected inference framing: gpt-oss streaming rows are published as context with the A1 caveat inline, not as headline rows.
>
> ## Unchanged
>
> R1–R4, R6, R8; Phase 1 in full (the transformers-dequant anchor remains the ground-truth gate); the out-of-scope list, including #1965 scope freeze and Blackwell FP4 paths; all deliverable formats.
>
> *Addendum ends. Stamp alongside the base plan; both hashes in the commit message.*

## Appendix B — Addendum 2 verbatim (2026-07-15, superseded by this file)

> # Addendum 2 — Audit Corrections (applies to AGENT-PLAN-mxfp4.md, Addendum 1, and AGENT-PLAN-multiarch.md)
>
> **Date:** 2026-07-15 · **Source:** adversarial viability audit, 2026-07-15 · **Method:** forward-only (R2). Where this conflicts with the base plans or Addendum 1, this addendum governs.
>
> The audit's verdicts: the byte-provenance training claim is genuinely novel (no framework trains over native MXFP4; Unsloth's backward pass is an unimplemented WIP), but the *user outcome* "gpt-oss-20b fine-tuned on 12–16 GB" is already served by Unsloth at 12.8–14 GB via NF4 conversion, and the Qwen3-235B 4.3 tok/s figure is no longer a throughput lead (ik_llama.cpp community result ~7.4 tok/s on a 3090 + 128 GB at IQ3_K). The plan survives; its language doesn't. The following amendments make the agent enforce the surviving version.
>
> ## B1 — Headline resharpened (supersedes A1's headline)
>
> - **The demo centerpiece is the provenance receipt, not a VRAM number.** Elevate A2's identity gate from test to artifact: SHA-256 every frozen expert tensor before training and after N steps, publish the hash table, stamp it (OTS) alongside the run ledger. The demo sentence: *"Fine-tuned; expert bytes bit-identical to OpenAI's release — here are the hashes."* No competing framework can currently produce this artifact.
> - **gpt-oss-20b is a development target only.** It is PROHIBITED as a headline or launch claim in any form resembling "fine-tune gpt-oss on a 16 GB card" — Unsloth ships that outcome at 12.8–14 GB. 20b appears publicly only as the provenance-receipt proof-of-method.
> - **gpt-oss-120b on ≤16 GB is the outcome headline**, projected tier until measured (Unsloth's 120b path needs ~65 GB). If the 120b run doesn't exist by the gate, the launch leads with provenance + the Qwen/NF4 story, and 120b stays a stamped projection.
>
> ## B2 — New standing rule R9: name the incumbent
>
> Any claim adjacent to an incumbent capability must name the incumbent in the same breath. To make this enforceable:
>
> - New deliverable: `docs/COMPETITIVE.md` — dated, cited entries the agent verifies (R6) rather than trusting this addendum: **Unsloth** (gpt-oss-20b QLoRA 12.8–14 GB via MXFP4→NF4 conversion; 120b at ~65 GB; MoE 4-bit QLoRA publicly conceded unsupported; MXFP4 backward W_TRANSPOSE unimplemented), **Marlin/Machete + vLLM fused Marlin MoE** (fused 4-bit MoE GEMM for GPTQ/AWQ/MXFP4/NVFP4; Ampere/Hopper lean; known K=N=2880 alignment failure on gpt-oss shapes), **ik_llama.cpp / ktransformers / llama.cpp** (Qwen3-235B ~7.4 tok/s community benchmark, quant-precision caveat IQ3_K vs NF4; gpt-oss-120b CPU-MoE offload ~28–30 tok/s on high-end desktops).
> - Every public sentence (README, writeups, launch posts) making a comparative or "first/only" claim must cite its COMPETITIVE.md entry. No entry, no claim.
>
> ## B3 — Inference claims requalified (extends A1/A4)
>
> - **No throughput-leadership claims for Qwen3-235B.** The sanctioned differentiators: exact-checkpoint-bytes (4-bit NF4 vs ~3-bit IQ quants — a quality-per-bit claim, stated with the precision caveat), energy-per-token where measured, and the receipts methodology itself.
> - Projection tables (multiarch P0 and mxfp4 P0): the incumbent-baseline column extends to **all** model rows, not just gpt-oss, each with a quant-precision field so numbers are never compared across unequal bit-rates without saying so.
> - **Gate-package stretch item (optional, tightly scoped):** one same-box incumbent A/B — ik_llama.cpp on the operator's own hardware, one model, one config, quant difference documented — pre-empting the first hostile launch-thread question. If it threatens the gate date, it drops (R5/A3 priority order unchanged).
>
> ## B4 — Kernel phrasing lock (extends R7)
>
> Sanctioned kernel description: *"fused grouped GEMM on bitsandbytes NF4-packed weights with host streaming, on consumer GPUs."* Bare "fused 4-bit MoE GEMM" is prohibited as a novelty claim — Marlin MoE exists and a skeptical reviewer will cite it. Marlin/Machete are acknowledged by name wherever the kernel's novelty is asserted; the gpt-oss shape incompatibility may be cited (from COMPETITIVE.md) as a concrete coverage difference, not as a dunk.
>
> ## B5 — Multiarch plan corrections (AGENT-PLAN-multiarch.md)
>
> - **P1.1/P1.3 pre-seeded findings to verify, not assume:** ROCm 7.2 gives official RDNA4 (gfx1201) support, but Triton GEMM on RDNA4 runs materially behind hand-written HIP (competition findings: 30–50%); bnb ROCm backend is functional but the stable PyPI wheel is CUDA-only (source build or fork required); bnb Intel XPU is preview-grade. Consequence: **Intel drops below AMD in Phase 2 ordering**, and AMD-consumer projection rows carry a "Triton maturity" note widening the band.
> - **Strix Halo constant correction (R6):** use measured bandwidth (~212–215 GB/s) with citation, not the ~256 GB/s theoretical; note Vulkan-vs-ROCm variance and that training tooling lags inference on that platform.
> - PCIe 5.0 ×16 on RDNA4: confirmed; row stands.
>
> ## B6 — New standing rule R10: the watchlist
>
> `docs/WATCHLIST.md`, checked at the start of every agent session; any trigger = stop work on the affected phase and report before proceeding:
>
> 1. **Unsloth ships the MXFP4 backward pass** (W_TRANSPOSE or equivalent) → byte-provenance novelty gone; B1 headline dies; escalate for re-plan.
> 2. **unslothai/unsloth #4032 closes.** Correction to the conversation record: #4032 is **OPEN with a WIP label** (the earlier "Closed" reading came from a cross-reference render, not the issue). Open = the bnb-4bit-MoE wedge stands and the Daniel opener re-anchors on it. Closed-by-shipping = wedge collapses; pivot to B1's provenance niche.
> 3. **bitsandbytes v0.50.0 releases** → immediately audit its contents for grouped/MoE 4-bit kernels; also fires the PR #1965 "congrats + ready" comment per the existing Monday-gate rule.
> 4. **Marlin/vLLM or bnb adds bnb-NF4 grouped-MoE GEMM** → kernel's format niche closes; host-streaming becomes the sole differentiator; re-plan kernel positioning.
>
> ## Unchanged
>
> R1–R6, R8; A2's harness-reuse and anchor structure; A3's gate date (2026-08-01), private-track priority, and Phase-3 lockout; Phase 1 in full; the out-of-scope list.
>
> *Addendum ends. Stamp with the set; all three hashes in the commit message. The audit was run under the house standard — failures at full volume — and the plan is better for the dents.*

*Note: B6.2's "#4032 is OPEN" assertion above is preserved verbatim as
delivered but is factually wrong — see §6 trigger 2 for the correction and
its evidence.*
