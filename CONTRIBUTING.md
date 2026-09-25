# Contributing

Thanks for looking. This kernel is small on purpose; the bar is receipts, not
volume.

## The one rule: claims carry receipts

Every performance or comparative claim in a PR must cite a committed receipt
(a results doc + its evidence JSONs). A number in prose quotes its claim ID in
`docs/claims.json`: the register, not the PR text, decides whether it is
current, and `python scripts/check_readme_claims.py` holds the README, STATUS
and the solution pages to it in CI. A new number follows the measured-result
flow in `docs/change-impact.json` — receipt, claim entry (status from the
register's `status_vocabulary`), `docs/STATUS.md`, then prose quoting the ID.
`AGENTS.md` sections 2–9 are the full rule set and bind every contributor. The
README examples are CI-executed (`test_readme_cpu_block.py`) so they cannot
drift from the API; keep that invariant — a new documented call gets a
runnable block.

## Running the checks

```bash
cd kernel
pip install torch pytest numpy             # torch CPU wheel is fine; add triton on Linux (no macOS/Windows wheel)
python -m pytest test_readme_cpu_block.py test_cpu_refusal.py -q   # CPU-only, no GPU; runs without triton via _triton_shim
TRITON_INTERPRET=1 python -m pytest test_interp_contract.py -q      # device-free semantics; needs real triton (Linux)
```

The private-marker guard CI runs as a required check (`guard`) is also a
pre-push hook, scanning with the same `.github/private-markers.txt`:
`ln -sf ../../scripts/hooks/pre-push .git/hooks/pre-push`.

The pure-torch `dequant_ref` is the CPU-checkable oracle; the fused
`gemm_4bit_grouped` requires CUDA (it says so, loudly, if called on CPU).

## Hardware we'd love help measuring

The cross-vendor projections (`PROJECTIONS-multiarch.md`) are stamped
arithmetic. One non-NVIDIA confirmatory exists: AMD MI300X (CDNA3), run
2026-07-22 — correctness confirmed and a resident compute census measured, no
streaming row confirmed (`RESULTS-multiarch-mi300x.md` and its follow-up).
On-silicon confirmatory runs are the most valuable contribution:

- **AMD (ROCm)** — MI2xx or Radeon, or a streaming (host-tier) run on MI300X,
  which the 2026-07-22 run did not cover. AMD's Developer Cloud offers free
  credits that fit a confirmatory run; a `bench/hw_contract.py` pass + a
  census sweep is the ask.
- **Intel (XPU/SYCL)** — Arc / Max; the SYCL port (`sycl/`, results in
  `sycl/M2-RESULTS.md`) is cross-vendor and wants absolute-magnitude numbers
  on real Arc silicon.

Open a "hardware-wanted" issue (template provided) with your device + the
receipt, and we'll fold it into the projections table with credit.

## Scope

Kernel-math changes need a fidelity receipt (the property suite must stay green
and the fused error must stay at/below the dequant baseline). Docs/test PRs are
welcome and low-ceremony.

## Issue to PR: how something gets from reported to merged

**Labels are a state machine, not decoration.** Every issue lands
`needs-triage`, and each issue form also applies its *kind* label (`bug`,
`measurement` or `hardware-wanted`); triage removes `needs-triage` and leaves
exactly one *kind* label.

| label | means |
|---|---|
| `needs-triage` | not looked at yet — the default entry state |
| `bug` | it raises, hangs, or returns wrong values |
| `measurement` | a published number does not reproduce (see that issue form) |
| `hardware-wanted` | we cannot measure this ourselves; a run on your silicon is the contribution |
| `prereg-required` | cannot be answered without a stamped, pre-data protocol |
| `wontfix` | closed with a reason (out of scope included), never silently |

`prereg-required` is the one that surprises people. If settling an issue means
producing a *number* — is X faster, does Y use less memory — the protocol gets
written and OTS-stamped **before** the data exists, and the PR links it. This is
not ceremony: it is why claims in this repo can be retracted cleanly when they
are wrong, and several have been.

**Branch names** say what the change is: `fix/`, `feat/`, `docs/`, `chore/`,
`bench/` for a measurement lane. Nothing is pushed to `main` directly — every
change lands through a PR and is squash-merged, so `main` is one commit per PR
titled `… (#N)` and the discussion stays on the PR. The maintainer reviews and
merges every PR, his own included.

**What review actually blocks on**, in order:

1. **Does it compute the right function?** Anything touching expert math needs
   parity against the pure-torch oracle, not a loss curve. This outranks speed.
2. **Do the claims carry receipts?** A number without a committed receipt cannot
   be reviewed, only believed.
3. **Does CI pass?** Including the private-marker guard, which is not advisory
   and runs on every push. The `ci` workflow itself runs only once the
   maintainer applies the `ready-to-merge` label (or the PR leaves draft), never
   on every push: after a new push the label is removed and re-applied, and a
   head with no CI run is not green.
4. Style, naming, and everything else — last, and rarely.

**If you disagree with a review, say so with evidence.** A measurement that
contradicts a maintainer is the most useful thing this project receives, and it
has changed conclusions more than once.
