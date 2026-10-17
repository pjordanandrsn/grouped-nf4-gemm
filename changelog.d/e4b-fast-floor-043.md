### Changed

- `docs/system-manifest.json`: a compatibility record for experts4bit-qlora 0.55.0 and later, flooring its `fast`
  extra at grouped-nf4-gemm 0.43.0, the release that makes the bandwidth-targeted NF4 decode GEMV the default
  (`GNF4_GEMV_BW=auto`, #509). It is experts4bit-qlora P116's registered consequence. The previous record (floor 0.30.0)
  now covers experts4bit-qlora 0.35.0 to 0.54.x, so it still describes every released consumer. Documentation only:
  no kernel or default changes.
