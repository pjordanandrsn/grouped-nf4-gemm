<!-- Keep what applies, delete what doesn't. A one-line typo fix does not need
     the measurement section — say so and move on. -->

## What this changes

<!-- One or two sentences. If it fixes an issue, link it so the issue closes on
     merge: "Closes #123". -->

## Why

<!-- The reasoning, not the diff. What was wrong, or what could not be done
     before. -->

---

### If this PR makes a performance or comparative claim

The one rule in [CONTRIBUTING.md](../CONTRIBUTING.md) is that claims carry
receipts. A number in a PR description with no committed evidence behind it
cannot be reviewed, only believed.

- [ ] The claim cites a **committed receipt** (results doc + its evidence JSONs), or is marked "measuring now"
- [ ] A **self-pair** ships with it — the same arm timed against itself. A ratio inside the instrument's own spread is not a measurement
- [ ] **Two devices**, or the claim names the single architecture it holds for. Results in this repo have reversed between devices before
- [ ] The **cells that lose** are reported, not only the ones that win
- [ ] If a pre-registered protocol governs it, the prereg was **stamped before the data** and is linked

### If this PR touches the README

- [ ] Links are **absolute** (`https://github.com/...`), never relative — the README renders on PyPI, where relative paths are dead
- [ ] Self-repo `blob`/`tree` refs pin the **current `project.version` tag** or `main`
- [ ] Any new documented call has a **runnable CPU block**, so `test_readme_cpu_block.py` keeps the docs from drifting from the API

### Always

- [ ] `python -m pytest test_readme_cpu_block.py test_cpu_refusal.py -q` passes (CPU-only)
- [ ] No private-lane paths or markers (the pre-push guard and `private-marker-guard.yml` scan for these; do not `guard-allow` your way past one without saying why)
- [ ] Commit messages say **why**, and correct anything they supersede rather than quietly dropping it
- [ ] **`Cursor Bugbot` reads `pass`, not `skipping`** — on this repo `skipping` means Bugbot found something and is withholding the green, not that it declined to run. Neither state shows as a failure in `gh pr checks`, so "nothing is red" is not the same as clean; read the Bugbot line itself before merging

<!-- If work was drafted with an AI assistant, keep the Co-Authored-By and
     AI-disclosure trailers. That is this project's default, not an apology. -->
