### `GNF4_GEMV_BW=auto` also engages at Granite-3.1-3b-a800m's and OLMoE-1B-7B's expert shapes on ≥ 160-SM parts (experts4bit-qlora P134)

- experts4bit-qlora's served lane P134 read each family LICENSED on an RTX 5090 at the shipped plans (experts4bit-qlora#1697,
  run `p134-5090-1`). On the default `serve_paged` server, one request decoded 1.3979× (Granite) and 1.5011× (OLMoE) as fast.
  Sixteen requests were unchanged (0.9992, 0.9954), within P110's quality bar.
- `_BW_SHAPES` gains Granite's (1024, 1536) / (1536, 512) and OLMoE's (2048, 2048) / (2048, 1024), under auto's ≥ 160-SM
  gate. Their plans were already in `_BW_PLANS`.
- The 128-SM sm_89 path (the RTX 4090) keeps Qwen3-30B-A3B's two shapes only (`_BW_SM89_SHAPES`): P134 read the RTX 5090
  alone.
- Dot-pad, every other shape and every other sub-160 part are unchanged. `GNF4_GEMV_BW=0` turns it off.
- P134 licenses shapes, not families: Granite's pair also covers the Granite-4.0-h fixture.
- Tests:
  - `kernel/test_m3_defaults.py`: P134's four shapes engage at 160, 170 and 188 SMs; they keep their route on sub-160 parts
    and on the RTX 4090; Mixtral's expert shapes stand for an unlicensed shape at ≥ 160 SMs; `_BW_SM89_SHAPES` is a
    subset.
  - `kernel/test_dispatch_counts.py`: the dispatch layer at two of P134's shapes, mocked, and unmocked on a real
    ≥ 160-SM card (skipped elsewhere).
