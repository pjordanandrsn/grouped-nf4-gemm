### Docs: the required guard checks by their new names

AGENTS.md and CONTRIBUTING.md now name the two required guard checks, `private-marker-guard` and
`conflict-marker-guard`, in place of `guard`. Each runs on every push and pull-request event, not on a label, and
push-scan runs ("(push scan)") are not required. They also say that a label event on a pull request that does not carry
`ready-to-merge` now fails the required `ci` jobs instead of skipping them, while on one that does, any label re-runs
CI on its head.
