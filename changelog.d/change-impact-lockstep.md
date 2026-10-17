### Shared tooling: check_change_impact accepts a manifest record that landed first (the kernel-first lockstep)

- In the runtime role's dependency-floor class, `docs/system-manifest.json` now counts as satisfied when the diff does
  not touch it but the BASE's manifest already carries the head's record. That record is a compatibility entry for this
  consumer, `grouped-nf4-gemm` and the `fast` extra whose `consumer_versions` contain the head's new version and whose
  `floor` equals its new floor (`lockstep_record`). The report prints it as `landed first`.
- This is the normal shape of a floor move: the record is synced into both repositories before the consumer's
  pyproject moves (experts4bit-qlora#1692, then its 0.55.0 release). Any other missing manifest still FAILs.
- `kernel/test_check_scripts.py` covers the accepted shape and each refusal: a floor mismatch, a range that misses the
  version, no record, no base manifest, another consumer or extra, and an unsupported range.
