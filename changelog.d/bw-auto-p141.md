### `GNF4_GEMV_BW=auto` also engages at Mixtral-8x7B's expert shapes on ≥ 160-SM parts (experts4bit-qlora P141)

- experts4bit-qlora's served lane P141 read LICENSED on an RTX 5090 (experts4bit-qlora#1754, run `p141-5090-2`). Its
  phase 1 chose each shape's plan on the card by K33's rule: every one of the twelve plans meets the contract first, then
  the lowest median over 40 rounds wins. At those plans, on the default `serve_paged` server, one request decoded
  2.4075× as fast (19.53 → 8.11 ms a step). Sixteen requests were unchanged (0.9964), within P110's quality bar.
- `_BW_SHAPES` gains Mixtral's (28672, 4096) and (4096, 14336), under auto's ≥ 160-SM gate.
- `_BW_PLANS` gains their plans, (16, 1024, 4, 1) and (16, 512, 4, 2): the plans P141's served read ran.
- The 128-SM sm_89 path (the RTX 4090) keeps Qwen3-30B-A3B's two shapes only. Dot-pad, every other shape and every other
  sub-160 part are unchanged. `GNF4_GEMV_BW=0` turns it off.
- P141 licenses shapes, not families: any model with these expert shapes takes them.
- Tests:
  - `kernel/test_m3_defaults.py`: P141's two shapes engage at 160, 170 and 188 SMs, take exactly the plans the read ran,
    and keep their route on sub-160 parts and on the RTX 4090; two unread shapes now stand for "every other shape" at
    ≥ 160 SMs, in place of Mixtral's.
  - `kernel/test_nf4_gemv_bw_interp.py`: the shape set and the plan table.
  - `kernel/test_dispatch_counts.py`: the dispatch layer at both shapes, mocked.
