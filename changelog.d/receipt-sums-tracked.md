### Tooling: every SHA256SUMS line is tracked or named at the same sha by STORE-ONLY.sha256

`kernel/test_receipt_sums.py` reads every tracked `SHA256SUMS` in the repository: each line must be a file git tracks
(the index, so a log ignored by `*.log` but present on disk fails) or a line of the `STORE-ONLY.sha256` beside it at
the same sha256. One directory is excluded by name, `docs/artifacts/pypi-family-20260718/`: an ops prep manifest of
pre-upload builds, which a new `README.md` there describes. No existing file changes.
