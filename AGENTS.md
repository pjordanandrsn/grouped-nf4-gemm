# AGENTS.md — working in grouped-nf4-gemm

## 1. Router

Starting from a model, training, serving, or fit problem? → experts4bit-qlora.
Starting from a packed layout, kernel, GEMM/GEMV, attention, or arena primitive? → grouped-nf4-gemm (this repository).
Need a current number? → docs/claims.json.
Need current support/status? → docs/STATUS.md + docs/capabilities.json.
Changing public behaviour? → AGENTS.md: 'When you change something'.

## 2. Purpose and ownership boundary

`grouped-nf4-gemm` is the kernel package of a two-package system: the grouped
expert GEMMs directly on NF4-packed (bitsandbytes `gemm_4bit` layout) and native
MXFP4 (e2m1 + e8m0) stacks, the int4-b32 decode GEMV and its calibrated packer,
the fp8 paged decode attention, the decode glue, the pure-torch references every
kernel is asserted against, and the host/NVMe arena primitives. The consumer,
[`experts4bit-qlora`](https://github.com/pjordanandrsn/experts4bit-qlora), loads
and quantises models, trains adapters, places bytes across tiers, serves, owns
model-level measurement and its own claims register, and installs this package
through its `[fast]` extra; this package never depends on it. A kernel lands and
is released here first; the consumer floors on the release that ships it. The
machine-readable boundary — ownership, capability IDs per package, the `compatibility`
records (which consumer versions require which kernel floor), the evidence vocabulary,
the invariants — is [`docs/system-manifest.json`](docs/system-manifest.json), byte-identical
in both repositories and validated by `scripts/check_system_manifest.py`.

## 3. Repository map

| path | what |
|---|---|
| `kernel/` | the shipped flat modules (`pyproject.toml` `py-modules`) with their tests (`kernel/test_*.py`) beside them; `_triton_shim` binds triton or a stand-in |
| `gnf4_native/` | the compile-at-first-use CPU kernels (the hybrid CPU path) |
| `examples/dequant_tax.py` | the one-minute on-your-own-GPU demonstration; CI runs it on CPU and asserts the CPU note |
| `docs/` | `STATUS.md`, `claims.json` + `claims-schema.md`, `capabilities.json`, `system-manifest.json`, `change-impact.json`, `KERNEL_CONTRACT.md`, `TOLERANCE_CONTRACT.md`, `PORTABILITY.md`, `SOLUTIONS.md` + `solutions/`, `INDEX.md` (which documents are current, which are anchored), results and pre-registrations |
| `bench/`, `projections/`, `router_probe/` | receipts, the projection model and its anchor gate |
| `scripts/` | the CPU-only contract checks of section 6, the README link checker, the wheel smoke |

## 4. Sources of truth

- **claims.json wins for numbers.** Whether a number is current is
  [`docs/claims.json`](docs/claims.json), not CHANGELOG prose or a README
  sentence; prose quotes the claim ID and its status.
- **STATUS.md wins for position** ([`docs/STATUS.md`](docs/STATUS.md)),
  including the three limits where the fused path loses.
- **Historical and anchored records are never rewritten.** A document with a
  sibling `.ots` file (`find . -name '*.ots'`) or an `ots-attestation-footer` marker
  is never edited in place; corrections go in a sibling file ([`docs/INDEX.md`](docs/INDEX.md)
  says which documents are current).
- **A green skipped test is not evidence** that a path was exercised: a green
  CI run whose GPU suites skipped proves nothing about the GPU path.
- **A private measurement is not publicly reproducible**: `measured-private`
  is a real run receipted outside this repository, and is always labelled.
- **A noise floor is not a budget**: the registered gate is applied in its
  own units, with the floor quoted beside the verdict.
- **Failed gates stay failed**; a gate is never retuned to fit.

These are the `invariants` and the `evidence_vocabulary` (confirmed, verified,
measured, measured-private, projected, open, superseded, retired — every tier
distinct) of `docs/system-manifest.json`; `docs/claims.json` `status_vocabulary`
defines the tiers this register uses. Dependency floors live in `pyproject.toml`
(validated by `scripts/check_system_manifest.py` and `scripts/check_dependency_floor.py`):
never copy a floor into prose. Kernel-level numbers are registered here;
model-level numbers (tok/s for a named model, perplexity gates) are the consumer's.

## 5. Public API and capability map

[`docs/capabilities.json`](docs/capabilities.json) is the list: one entry per
capability ID — `grouped-nf4-gemm`, `native-mxfp4-moe-inference`, `int4-decode-gemv`,
`fp8-paged-attention-fp8-compute`, `fp8-paged-attention-f32-compute`,
`decode-glue-kernels`, `stream-experts-from-host-or-nvme`, `verify-checkpoint-provenance`
— with its `entrypoints` (`module:Symbol`, `cli:…`, `flag:…`), environment,
limitations, `status` and claim IDs. `scripts/check_capabilities.py` resolves every
entry point against the shipped modules; the README's "Which entry point?" table is
the human form. Shipped modules outside that list (`cold_*`, `reuse_profile`, `vram_slots`,
`segmented_pool`, `dev_row_cache`, the 20b drivers, `_triton_shim`) are internal;
shipping is not publishing.

## 6. Build and test

```bash
pip install -e . pytest numpy                      # CPU torch + triton (Linux) run the interpreter suites
cd kernel && TRITON_INTERPRET=1 python -m pytest test_interp_contract.py test_mxfp4_interp.py test_mxfp4_gemv_b32.py -q   # the _INTERP_FILES members, alone
cd kernel && TRITON_INTERPRET=1 python -m pytest test_int4_b32.py -q             # a compiled-path file: its own process (why: below)
cd kernel && python -m pytest test_packaging_covers_kernel.py test_check_readme_links.py -q
python examples/dequant_tax.py                     # prints the CPU note without a GPU; ~1 min on one GPU
python -m build && python -m twine check dist/* && python scripts/wheel_smoke.py   # the smoke runs from outside the tree, against the wheel
python scripts/check_readme_links.py               # README links are absolute; self-refs = v<version> or main (network)
python scripts/check_system_manifest.py            # docs/system-manifest.json vs pyproject, claims, capabilities; --sibling PATH for the pair
python scripts/check_dependency_floor.py           # version statements in current docs vs pyproject and the manifest
python scripts/check_change_impact.py --base origin/main   # docs/change-impact.json: the companions a diff is missing
python scripts/check_capabilities.py               # docs/capabilities.json vs schema, pyproject, source, claims
python scripts/check_claims_register.py            # docs/claims.json bookkeeping: evidence resolves at HEAD, ISO measured_on, superseded/retired fields, quoted_in, no placeholder notes
python scripts/check_readme_claims.py              # README, STATUS and solution pages quote the register: ids exist, inactive ids say so, README/STATUS tables hold current values
python scripts/check_discovery_contract.py --bm25 --bm25-min-top1 33   # docs/discovery-queries.json vs the pages; BM25 floor = baseline 35/44 minus two
python scripts/check_docs_examples.py --root .     # doc code blocks parse, links resolve; --run-cpu-blocks kernel executes the CPU-only ones
python scripts/build_llms_bundle.py --check        # llms-full.txt is current
```

The two interpreter commands stay separate on purpose: `kernel/conftest.py` raises
`pytest.UsageError` when a CUDA device is present and an `_INTERP_FILES` member
(`test_interp_contract.py`, `test_mxfp4_interp.py`, `test_mxfp4_gemv_b32.py`) is
collected in the same process as a compiled-path file such as `test_int4_b32.py`, because
`TRITON_INTERPRET` latches process-wide at triton's first import; CI's CPU-only runner
never trips that guard, a GPU box does. An interpreter-mode file
must be registered in `_INTERP_FILES` and named in the CI step
(`test_packaging_covers_kernel.py` enforces both). CI (`.github/workflows/ci.yml`)
runs the anchor gate, the CPU example, the interpreter suites, the wheel smoke and
the `discoverability` job (every `scripts/check_*.py` above), asserting triton is importable so a skip is never silent.

## 7. Rules that have bitten

- Every kernel has a pure-torch reference and is asserted against it;
  parity with the reference outranks speed in review.
- Claims carry receipts, a self-pair beside every ratio, and the cells that
  lose; a pre-registered protocol is stamped before the data exists.
- Nothing falls back silently: the NF4 grouped GEMM refuses CPU tensors with an
  error that names `dequant_ref` (the MXFP4 and int4_b32 kernels fail inside the
  Triton launch); `examples/dequant_tax.py` prints a CPU note rather than pretending.
- Layouts are contracts ("Layouts at a glance" in [`docs/KERNEL_CONTRACT.md`](docs/KERNEL_CONTRACT.md));
  the consumer's stores are built to them.
- Name the comparator: every "dequant path" ratio is against this repository's
  own per-expert loop as its receipt ran it; bitsandbytes 0.50.0 has a packed 2-D
  inference path and no grouped routed-MoE contract, so never write "stock 4-bit
  always dequantises" (the version-aware statement is on the NF4 solution page).
- The BM25 floor in `scripts/check_discovery_contract.py` is a local proxy over
  the query corpus: a BM25 regression is never evidence of LLM discoverability,
  and a BM25 pass is never a ranking claim.

## 8. When you change something (change-impact rules)

The contract is [`docs/change-impact.json`](docs/change-impact.json);
`python scripts/check_change_impact.py --base origin/main` names the companions
a diff is missing, and CI runs it on every pull request. Classes:

- **new-kernel-capability** (a module added to `py-modules`, or a new
  `@triton.jit` kernel): the ten-step flow in the contract — kernel and reference,
  layout docs, capability entry, tests wired into CI, consumer integration,
  consumer floor if required (only after the release is tagged), model-level
  measurement, quality gate, claim registration, release notes. `docs/capabilities.json`
  and `CHANGELOG.md` in the same diff (FAIL); `docs/KERNEL_CONTRACT.md` when a
  layout constant moved (WARN).
- **public-api-change** (signature, layout, return, or the entry-point set):
  docstring, contract, README table, capabilities entry points, the affected
  solution page, `CHANGELOG.md` under `## Unreleased` (FAIL), the consumer-side change.
- **measured-result**: receipt → claim entry → `docs/STATUS.md` (FAIL without
  it; `--allow-claims-only` downgrades) → prose quoting the ID.
- **dependency-floor**: a version bump needs `CHANGELOG.md` (FAIL); a
  torch/triton/numpy floor needs the pyproject comment, README install note and
  capabilities environments; a consumer floor is a new `compatibility` record
  in `docs/system-manifest.json` in both repositories, only after the kernel
  release exists.

Then regenerate `llms-full.txt` (`--check` is a CI gate). Release notes follow
[`docs/RELEASE_NOTES_GUIDE.md`](docs/RELEASE_NOTES_GUIDE.md); releases are `v<version>`
tags cut by the maintainer (`publish.yml` refuses a tag that disagrees with `pyproject.toml`), never from a branch.

## 9. Platform caveats

Kernels: Linux + NVIDIA sm_80+ with Triton ≥ 3.4 (sm_120 is the primary serving
target). CI tests Python 3.11 only, on Linux. On macOS/Windows the wheel installs
(triton is a Linux-only marker): the pure-torch surface (pack references, dequant,
provenance, arena bake/verify) imports and runs without triton via `_triton_shim`;
the Triton kernels need a CUDA GPU; macOS and Windows are not exercised by CI, so
say "not exercised by CI" rather than "supported". `int4_b32` imports triton
directly and is not importable without it. ROCm/XPU are port targets
(`docs/PORTABILITY.md`), not supported. The fp8 paged kernel's f32 compute modes
(the sm_80–sm_88 default and every explicit f32 request) miss their reference on
triton 3.4 (#319, `gnf4.open.f32-compute-modes-triton34`); `docs/capabilities.json` carries them as
`fp8-paged-attention-f32-compute` (`unsupported`) beside `fp8-paged-attention-fp8-compute` (supported: sm_89+, measured on the RTX 5090 only).

## 10. Working with other agents

Six agents touch this repository, coordinated through GitHub (the record) and
Slack channel `#ml-packages` (live handoffs). The live board — role chart,
blocking issues, handover protocol — is the Slack canvas "ML packages — live
board" in #ml-packages; `@Name` mentions in threads are answered with
`ACK <repo>#<issue>`, `DONE <repo>#<issue> -> <link>`, or `BLOCKED <repo>#<issue> - <reason>`. Seats without Slack (desktop agents) post `ACK <repo>#<issue>` / `DONE … -> <PR url>` / `BLOCKED … - <why>` as comments on the GitHub issue; review requests and fix lists for their PRs are posted on the issue as well as the PR; the CEO mirrors to #ml-packages.

**Jordan — Chair / Owner**: the maintainer (final decisions, account access,
rulings).

**Claude Code — CEO**: orchestration, releases, the merge gate.

**Cursor — CTO**: a bounded code task that starts from an issue and ends as a
pull request (substantive implementation).

**ChatGPT — Chief Science Officer**: review, research, repo-state reports;
read-only.

**Chief Data Officer**: Claude Code desktop on the mini; owns the organisational
corpus `pjordanandrsn/org-corpus`, the ingestion jobs, receipts + ledger
ingestion, scaffold ops; first successor to the CTO.

**Chief Experiments Officer**: Kimi K3 desktop agent on the MBP;
pre-registrations and experiment design, `PREDICT:` before any run, next-lane
harnesses under new `bench/<lane>/` trees, compute-economics analysis; never
touches a committed receipt tree.

**Scout — Chief Evidence Officer**: read-only verification (claims vs receipts,
prior art, upstream behaviour at a pinned revision); cites `file:line` / URLs;
never edits.

**Forge — COO**: mechanical repository work from issues on branches like
`forge/<issue>-<slug>`; single PR per issue; no merges; restricted from
touching gates, thresholds, floors, registered claims, or documents with an
OpenTimestamps footer.

**Warden — Chief Risk Officer**: read-only review: correctness, silent
fallbacks, gate / threshold drift, claims outrunning receipts, security;
findings as `file:line — claim — why — severity` + what was not checked; hands
fixes to Forge.

### Succession & dynamic reassignment

Any role can go dark; any role that notices reassigns. A teammate is unavailable
when it posts an explicit failure line (unpaid invoice, usage limit, context
limit, "something broke"), gives no ACK within 15 minutes of a mention, shows no
DONE/BLOCKED/progress within 60 minutes of an ACK, or says so.

**Successors:**

| Role | First successor | Second successor | Third successor | Fourth successor |
|------|----------------|------------------|-----------------|------------------|
| CEO | CTO | CDO | CSO coordinates; Jordan | — |
| CTO | CDO | COO (docs/register/small code only, Warden review) | CEO | — |
| Chief Science Officer | Scout | CDO | CEO | — |
| CDO | CEO | COO | — | — |
| COO | CDO | CTO | CEO | — |
| Chief Evidence Officer | Warden | Chief Science Officer | — | — |
| Chief Risk Officer | Scout | Chief Science Officer | CEO | — |
| Chief Experiments Officer | Chief Science Officer | Warden | CEO | — |
| Independent review | Warden (source read) | CXO / CTO / CSO / CDO (not the author) | CEO | — |
| Chair / Owner | none (decisions wait, `blocked:user`) | — | — | — |

**Procedure:**

1. Post in the issue's thread: `REASSIGN <repo>#<issue>: <role> -> <successor role> -- reason: billing_limit|usage_limit|context_limit|availability|crash|manual`
2. Include a fenced code block labelled `handoff` with: task, current_state, known_facts, open_questions, artifacts, next_action
3. Add a real mention of the successor
4. The successor answers `ACK <repo>#<issue> (acting <role>)`
5. The successor works under ITS OWN charter limits (a read-only role stays read-only, the COO never touches gates)
6. In-flight work stays with the acting owner until DONE; the returning primary posts HANDBACK only for items not yet started

Every rule survives reassignment.

Every handoff posted in Slack is also written on the issue.

- A task is an issue opened from the **Agent task** template (goal, acceptance
  criteria, evidence required, constraints) carrying exactly one of
  `agent:cursor` / `agent:chatgpt` / `agent:claude`; `handoff` marks it ready
  for the agent named in the last comment; `blocked:user` means the maintainer
  must act (decision, credential, account link).
- The taking agent acknowledges on the issue, works on a branch, opens a pull
  request that cites the issue, and stops. Every PR gets one independent review
  before merge — by a seat that did not author it: Warden's source read
  (`file:line — claim — why — severity`), or a review by the CXO, CTO, CSO, CDO
  or CEO; Cursor Bugbot and automated Claude Code review are retired (Jordan,
  2026-09-06). Claude Code holds the merge gate: checks green on the head that
  carries the `ready-to-merge` label + that review + zero unresolved threads,
  with section 8's companions in the same diff. CI runs on `main` and on a pull
  request only when the gate holder applies `ready-to-merge` (or the PR leaves
  draft), never on every push (Jordan, 2026-09-06); a head with no CI run is
  not green — remove and re-apply the label after a new push. Nobody merges
  their own PR. The gate holder reads the PR's base branch before merging and confirms the squash commit is on `main` afterwards; a stacked PR is retargeted with `gh pr edit --base main` (or its base branch deleted) before merge. Receipts
  and code ship in separate pull requests; a pull request too large to review
  in one read is a finding, not a pass.
- A task pull request never moves a gate, a threshold, a compiled-kernel
  default or a registered claim; if the task needs one, the agent stops and
  says so on the issue. Kernel changes need the GPU tests run on the named
  hardware, with the receipt in the pull request.
- Sections 2–9 bind every agent equally; documents that carry an
  OpenTimestamps footer are never edited in place (a sibling file with errata,
  never the original).
- Nothing is filed upstream without the maintainer's explicit say-so: no issues,
  pull requests, discussions or bug reports on any repository outside
  `pjordanandrsn/*` (Triton, PyTorch, transformers, bitsandbytes, vLLM, Unsloth,
  ...). Findings about other projects stay as notes in this repository's receipts
  and issues until the maintainer decides.
- Never invent an identifier, URL, hash, timestamp, title or number. If you do not have it, write UNKNOWN and say what would produce it. A placeholder is a fabrication.

