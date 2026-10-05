### Shared tooling: `check_change_impact.py` accepts a `changelog.d/` fragment as the CHANGELOG companion where a repository keeps one (tooling only)

- experts4bit-qlora moves its `## Unreleased` entries to one file per change in `changelog.d/`, because every merge left
  every other open pull request DIRTY on the one hunk they all rewrote. There, a fragment the diff adds or edits satisfies
  a `CHANGELOG.md` companion (public-api-change, new-kernel-capability). A version bump still needs `CHANGELOG.md` itself,
  because the release writes it.
- **Inert here:** this repository has no `changelog.d/`, so every rule reads as before. Shared tooling lands here first,
  then the identical file goes to experts4bit-qlora. Moving this repository's own `## Unreleased` to fragments is a
  separate change. No package code changes.
