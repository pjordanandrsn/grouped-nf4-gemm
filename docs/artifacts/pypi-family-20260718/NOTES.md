# PyPI family prep — ops notes (2026-07-18)

Template = live e4b/experts4bit metadata (e4b.pypi.json / experts4bit.pypi.json,
fetched 2026-07-18): version 0.1.0, MIT, requires-python >=3.9, summary
"Alias for <target> — <target tagline>.", requires_dist "<target>>=<current>"
+ per-extra passthrough, project_urls {Homepage: https://cerinamroth.com/ml/,
"Real package (<target>)": pypi page, Source: canonical repo}, NO module.

Interpretation recorded: live target experts4bit-qlora is 0.3.0 with extras
train/serve/TEST (test added after e4b published train/serve-only). The
directive's bolded "mirroring the TARGET's extras exactly" governs ->
expertsnbit passes through all three at >=0.3.0.

Canonical grouped-nf4-gemm 0.1.0: description = the B4-sanctioned kernel
phrasing; ships kernel/{nf4_grouped,nf4_pack_ref,host_gather}.py as top-level
modules (package-dir mapping, zero file moves — matches the interp suite's
bare `import nf4_grouped`); pins torch>=2.8/triton>=3.4 = the validated
floors (12-card sweep + CI), KERNEL_CONTRACT.md pins op conventions/silicon
tiers only.

Gates (2026-07-18, gnf4-v6 container): twine check PASSED x4 (canonical
sdist+wheel + 3 alias wheels); canonical wheel venv-install -> 3 imports OK;
interp contract suite 18/18 FROM THE WHEEL; nf4gemm+gnf4 --no-index
--find-links resolve -> import OK; expertsnbit -> live 0.3.0 from PyPI ->
import OK. Artifact hashes in SHA256SUMS.
