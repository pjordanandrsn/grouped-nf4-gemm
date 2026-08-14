# Contributing

Thanks for looking. This kernel is small on purpose; the bar is receipts, not
volume.

## The one rule: claims carry receipts

Every performance or comparative claim in a PR must cite a committed receipt
(a results doc + its evidence JSONs), or be marked "measuring now." The README
examples are CI-executed (`test_readme_cpu_block.py`) so they cannot drift from
the API; keep that invariant — a new documented call gets a runnable block.

## Running the checks

```bash
cd kernel
pip install torch triton pytest numpy      # torch CPU wheel is fine for the CPU suite
python -m pytest test_readme_cpu_block.py test_cpu_refusal.py -q   # CPU-only, no GPU
TRITON_INTERPRET=1 python -m pytest test_interp_contract.py -q      # device-free semantics
```

The pure-torch `dequant_ref` is the CPU-checkable oracle; the fused
`gemm_4bit_grouped` requires CUDA (it says so, loudly, if called on CPU).

## Hardware we'd love help measuring

The cross-vendor projections (`PROJECTIONS-multiarch.md`) are stamped but
pre-silicon on non-NVIDIA parts. On-silicon confirmatory runs are the most
valuable contribution:

- **AMD (ROCm)** — MI2xx/MI3xx or Radeon. AMD's Developer Cloud offers free
  credits that fit a confirmatory run; a `hw_contract.py` pass + a census
  sweep is the ask.
- **Intel (XPU/SYCL)** — Arc / Max; the SYCL port (`sycl-m2`) is cross-vendor
  and wants absolute-magnitude numbers on real Arc silicon.
- **NVIDIA sm_120 (RTX 5090)** — the one gap in our own fleet (cloud stock,
  not code).

Open a "hardware-wanted" issue (template provided) with your device + the
receipt, and we'll fold it into the projections table with credit.

## Scope

Kernel-math changes need a fidelity receipt (the property suite must stay green
and the fused error must stay at/below the dequant baseline). Docs/test PRs are
welcome and low-ceremony.

## Issue to PR: how something gets from reported to merged

**Labels are a state machine, not decoration.** Every issue lands
`needs-triage`; triage removes it and applies exactly one *kind* label.

| label | means |
|---|---|
| `needs-triage` | not looked at yet — the only default |
| `bug` | it raises, hangs, or returns wrong values |
| `measurement` | a published number does not reproduce (see that issue form) |
| `hardware-wanted` | we cannot measure this ourselves; a run on your silicon is the contribution |
| `prereg-required` | cannot be answered without a stamped, pre-data protocol |
| `wontfix` / `out-of-scope` | closed with a reason, never silently |

`prereg-required` is the one that surprises people. If settling an issue means
producing a *number* — is X faster, does Y use less memory — the protocol gets
written and OTS-stamped **before** the data exists, and the PR links it. This is
not ceremony: it is why claims in this repo can be retracted cleanly when they
are wrong, and several have been.

**Branch names** say what the change is: `fix/`, `feat/`, `docs/`, `chore/`,
`bench/` for a measurement lane. Nothing is pushed to `main` directly — every
change lands through a PR, merged with a merge commit so the discussion stays
attached to the history.

**What review actually blocks on**, in order:

1. **Does it compute the right function?** Anything touching expert math needs
   parity against the pure-torch oracle, not a loss curve. This outranks speed.
2. **Do the claims carry receipts?** A number without a committed receipt cannot
   be reviewed, only believed.
3. **Does CI pass?** Including the private-marker guard, which is not advisory.
4. Style, naming, and everything else — last, and rarely.

**If you disagree with a review, say so with evidence.** A measurement that
contradicts a maintainer is the most useful thing this project receives, and it
has changed conclusions more than once.
