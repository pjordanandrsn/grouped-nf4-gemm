# Security policy

## Reporting a vulnerability

**Do not open a public issue.** Two private channels, either is fine:

- **GitHub private advisory** — [report it here](https://github.com/pjordanandrsn/grouped-nf4-gemm/security/advisories/new). Best for anything specific to this repository.
- **Email** — `security@cerinamroth.com`. Encrypt sensitive reports to the
  [published PGP key](https://cerinamroth.com/.well-known/cerinamroth-pubkey.asc),
  also fetchable by WKD (`gpg --locate-keys security@cerinamroth.com`).

The canonical contact details are in
[`security.txt`](https://cerinamroth.com/.well-known/security.txt), and the
[disclosure policy](https://cerinamroth.com/policy/) that governs them applies
here: good-faith research, coordinated disclosure, no disclosure of an
unresolved issue before a reasonable remediation window has passed.

## What is in scope

This package is a compute kernel and the machinery that feeds it, so the
interesting surface is **what it parses and what it trusts**, not a network
boundary — it has none.

- **Arena and checkpoint readers.** `nvme_arena`, `nvme_residency`,
  `mxfp4_residency` and the loaders parse on-disk headers, offsets and lengths.
  A crafted arena or checkpoint that causes an out-of-bounds read, a wild
  offset, or memory disclosure is in scope.
- **Provenance verification.** `verify_provenance` and the hash paths exist so a
  user can assert their weights are the released bytes. A way to make
  verification report success on bytes that do not match is in scope, and is
  the highest-severity class here — it defeats the guarantee rather than
  crashing.
- **Deserialization.** Anything that turns a file into Python objects or into
  tensor metadata, including manifest and index handling.
- **Dependency handling** in what this project ships (pinning, install-time
  behaviour), not vulnerabilities in the dependencies themselves — report those
  upstream, though we appreciate a heads-up.

## What is not in scope

- Model *outputs*. Quantization changes numbers; a low-precision result is a
  documented tradeoff, not a vulnerability. Fidelity claims that do not
  reproduce belong in an issue, and there is a template for exactly that.
- Running untrusted code you supplied to your own process. Passing a malicious
  callable to a Python API is not a boundary crossing.
- Resource exhaustion from parameters you chose (an arena larger than your disk,
  a `hot_rows` larger than your RAM). These raise deliberately.
- Vulnerabilities in NVIDIA drivers, Triton, PyTorch or bitsandbytes. Report
  those to their maintainers.

## What to include

Whatever you have. A crafted file that reproduces it is worth more than a
paragraph of prose. If you can, say what the impact actually is rather than
scoring it — **honest severity is the standard applied in both directions here**,
and a well-described medium is more useful than an inflated critical.

## What to expect

An acknowledgement as soon as it is seen — this is a small project run by one
maintainer, so that is best-effort and not an SLA. From there: confirmation or a
reasoned disagreement, a fix, and a release. Credit in the advisory and the
changelog unless you would rather not be named.

If you disagree with a severity assessment, say so. That conversation is
welcome and has changed conclusions before.
