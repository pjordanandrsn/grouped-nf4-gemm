### Docs: consumer defaults and the bandwidth GEMV's scope, current at 0.50.0

- **K25, K21 and K19:** `docs/capabilities.json` no longer calls them opt-in in experts4bit-qlora. Each is its
  default for decode-batch rows: K25 since its P96, K21 since its P90 and K19 since its P88, each cited by claim id.
- **K39 (`lut="prmt"`):** it stays opt-in in this package. The README, STATUS, PORTABILITY and capabilities now add that
  experts4bit-qlora serves it by default on `qwen3_moe` from 0.56.0 (its P158).
- **The bandwidth GEMV's `auto` scope** is stated as shipped: Qwen3-30B-A3B, Granite-3.1-3b-a800m, OLMoE-1B-7B and
  Mixtral-8x7B on >= 160-SM parts, and Qwen3-30B-A3B on the RTX 4090. Dot-pad runs only with `GNF4_GEMV_BW=0`.
- **P133's and P134's figures** cite their claims (`e4b.serve.p133.gemv-bw.qwen3.4090.2026-10-18`,
  `e4b.serve.p134.gemv-bw.{granite,olmoe}.5090.2026-10-18`). P141 cites its claim id. P127's read is cited, and the line
  saying it was unread is gone.
- **Two dot-pad claims:** their text now says dot-pad was the default when measured. Their values are unchanged.
