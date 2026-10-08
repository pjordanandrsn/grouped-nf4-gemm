### `GNF4_GEMV_BW=auto` is the default: the bandwidth-targeted NF4 decode GEMV at Qwen3-30B-A3B's two shapes on ≥ 160-SM parts (experts4bit-qlora P116)

- **Why.** experts4bit-qlora's served lane P116 read DEFAULT_ON on an RTX 5090 (experts4bit-qlora#1336). On the default
  `serve_paged` server, `GNF4_GEMV_BW=1` at K33's plans decoded one request 1.2417× as fast (step 10.22 → 8.22 ms) and 16
  requests were unchanged. It stayed within P110's teacher-forced quality bar on wikitext and c4val1. K33 had read the
  kernel at 0.341× dot-pad (`gnf4.kernel.k33-nf4-decode-gemv-bw.5090.2026-10-07`).
- **What.**
  - Unset or empty `GNF4_GEMV_BW` now reads `auto`.
  - `_BW_SHAPES` names Qwen3-30B-A3B's gate_up (1536 × 2048) and down (2048 × 768).
  - `auto` engages there on ≥ 160-SM parts only. Every other shape and part keeps dot-pad or the scalar GEMV.
  - `_BW_PLANS` carries K33's selected plan per family shape, so `GNF4_GEMV_BW_PLAN` is no longer needed for them.
  - `1` still forces the route everywhere, and `0` turns it off.
- **Tests.**
  - `test_m3_defaults.py` has the defaults trio: unset engages where it was read, keeps the old route elsewhere, and an
    explicit `1` or `0` is never downgraded.
  - `test_dispatch_counts.py` asserts the new default at the dispatch layer: `bw_prmt32` on a 200-SM part, the scalar
    GEMV on a 26-SM one.
  - `test_nf4_gemv_bw_interp.py` pins the switch's parsing and K33's plan table against `receipts-k33/5090/k33.json`.
  - The dot-pad and scalar tests at Qwen3's shapes now pin `GNF4_GEMV_BW=0`, so they keep reading those routes.
- **Not measured.** Served reads on other families (Granite and OLMoE run the scalar GEMV at B=1; K33 read 0.22–0.27×
  there), on other cards, and with P115's fused stack at the same time.
