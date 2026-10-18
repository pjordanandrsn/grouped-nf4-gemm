### Changed

- `docs/system-manifest.json`: both experts4bit-qlora compatibility records now say the consumer's CI installs the
  kernel from the v0.47.1 release commit (they said v0.45.0). experts4bit-qlora#1768 moved its CI pin to v0.47.1
  (`decode_split_plan` and `split_cap`, #558). Prose only: no floor, record or kernel changes. The same bytes land in
  experts4bit-qlora's copy once a release carries this one (its CI compares against this repository's latest tag).
