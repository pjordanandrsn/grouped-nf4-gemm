### Changelog entries are one file each in `changelog.d/`; `## Unreleased` is no longer edited by hand (tooling only)

- **The rule** (experts4bit-qlora since #1149): a change adds `changelog.d/<pr-or-slug>.md`, exactly the `### Title` and body
  it would have put under `## Unreleased`. That section's body is now one pointer paragraph. New files never conflict, so
  concurrent pull requests stop going DIRTY on the one hunk they all rewrote.
- **`scripts/changelog_fragments.py` is shared tooling** (added to `SHARED`; this repository is upstream, experts4bit-qlora
  carries the identical file). `--release` writes the fragments into the new version's section, newest first by the
  commit that added each, and deletes them. `--render` previews the section. `--check` runs in CI. It refuses entries
  under `## Unreleased` and malformed fragments. Released history is append-only: the merge base's released sections must
  survive byte for byte as the file's tail, so only a release adds to them, and an unreleased fragment leaves only by
  being released. A deliberate edit to a released section passes with the `changelog-history-edit` label.
- **Migration.** The five sections under `## Unreleased` moved, unchanged, into `changelog.d/<original PR>-<slug>.md`.
  Joined in their old order they reproduce the old section byte for byte. The released sections are byte-identical.
- `kernel/test_changelog_fragments.py` (the same file as experts4bit-qlora's `tests/test_changelog_fragments.py`) is
  wired into CI. CONTRIBUTING.md, AGENTS.md section 8, `docs/RELEASE_NOTES_GUIDE.md` and `docs/change-impact.json`
  state the rule. No package code changes.
