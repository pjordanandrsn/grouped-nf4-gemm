# Errata (stamped documents are never edited; corrections live here)

- **RESULTS-v3-confirmatory.md, "Cumulative claim" paragraph**: the phrase
  "more energy-efficient ... in every measured cell with top_k ≥ 2" is
  overbroad relative to the same document's own criteria table, which
  records two parity-margin energy readings outside the top_k=1/tiny class
  (Arctic down 1.010, gpt-oss down 1.005, one A5000 instance, v3 run). The
  precise cumulative count is 104 of 112 confirmatory-grade cells below the
  baseline across v1–v3. The README carries the corrected statement.

- **docs/mxfp4/RESULTS-mxfp4-train.md, Method line "run_mxfp4_20b_qlora.py @ `TBD`"**:
  the runner shipped in the 0.2.x sdist at commit **`260b4ff`**
  (`git log -1 -- kernel/run_mxfp4_20b_qlora.py`). The stamped doc kept the
  `TBD` placeholder frozen (its bytes are immutable under the `.ots`); this is
  the pinned value. No result changed.
