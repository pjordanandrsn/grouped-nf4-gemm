# Lane B393 receipts — RTX 5090 (the lane's reading)

Run `b393-5090-1`, 2026-09-23, one RTX 5090 on Vast verified/secure (instance 52285900), launched by
`adertha/tools/pod-launch.sh` under the relayed owner authorization at
https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/393#issuecomment-5801391293. Actual cost $0.0299.

- `census.json` is every case: 144 `combine_rows` and 270 `reduce_partials`, three readings each, plus the bound ratios.
- `census.txt` / `summary.txt` hold the summary lines the runner printed. `summary.txt` adds the case-count assertion.
- `versions.txt` records where `int4_b32` resolved (site-packages) and the installed cut (0.33.2 @ `69bb93f`).
- `forensics.txt` is the GPU, memory, driver, power limit and clock.
- `teardown-proof.json` is the launcher's destroy record: HTTP 200, instance absent, empty list after.

The launcher's full receipt (`receipt.json`) and ledger row live in the private audit tree. This directory is the
public copy of the lane's own outputs. Read: [`../../RESULTS-b393-combine-reduce-bitwise.md`](../../RESULTS-b393-combine-reduce-bitwise.md).
