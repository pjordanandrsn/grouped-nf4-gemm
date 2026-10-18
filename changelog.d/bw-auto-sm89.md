### `GNF4_GEMV_BW=auto` also engages on 128-SM sm_89 parts (the RTX 4090), at Qwen3-30B-A3B's two shapes (experts4bit-qlora P133)

- experts4bit-qlora's served lane P133 read LICENSED on an RTX 4090 at the shipped plans: on the default `serve_paged` server,
  one request decoded 1.5145× as fast and 16 requests were unchanged (0.9998), within P110's quality bar
  (experts4bit-qlora#1695, run `p133-4090-2`). Until now auto left the 4090 on the scalar GEMV.
- `_bw_engages` now also engages where the device's compute capability is 8.9 and its SM count is exactly
  `_BW_SM89_SMS` (128). `_capability` reads the capability: None off CUDA and on ROCm.
- Unchanged: other sub-160 parts, other shapes, and dot-pad's gate. `GNF4_GEMV_BW=0` turns it off.
- Tests:
  - `kernel/test_m3_defaults.py`: the licensed part engages at both shapes; seven other sub-160 parts or shapes keep their
    route; `0` stays off; the capability helper ignores ROCm.
  - `kernel/test_dispatch_counts.py`: the dispatch layer on a mocked 128-SM sm_89 part, and unmocked on a real RTX 4090
    (skipped on any other card).
