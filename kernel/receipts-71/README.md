# gnf4#71 receipts: a pinned request is charged at the next power of two

`pinned_charge_probe.py` reads the container cgroup's own memory charge (v2 `memory.current`, v1
`memory.usage_in_bytes`) before and after allocating N MB in a fresh process. It allocates pinned memory with
`torch.empty(N, uint8, pin_memory=True)` and pageable memory with the same call without pinning, and it touches every
byte.

Both runs: RTX A2000 12GB, cgroup v1, driver 575.64.05, torch 2.8.0+cu128 (the stack `PINNED_ROW_FACTOR`'s docstring
names), in a throwaway container from `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` on the QNAP. 2026-10-04.

| file | sizes (MB) | fits |
|---|---|---|
| `a2000-v1-pow2-sizes.json` | 0, 512, 1024, 2048, 4096 (two reps) | pinned slope **1.0043**, pageable **1.002**, R² 1.0. This is #71's cap ladder (1.004) by another instrument, so the instrument agrees with that gold |
| `a2000-v1-non-pow2-sizes.json` | 0, 340, 700, 1359, 2100, 3000 | pageable slope 1.002 (R² 1.0). Pinned charges are **514.5, 1028.7, 2057.1, 4113.8, 4113.8 MB**: each request's next power of two |

The flat `PINNED_ROW_FACTOR = 1.9` was a size-dependent rounding ratio (1.0 to 2.0) read at two sizes. The 2026-08-13
ladder's 128-row (340 MB) and 512-row (1359 MB) tiers round to 512 and 2048 MB, which with the process baseline sit
inside its recorded brackets. `capacity_for_bytes` now models the rounding (`pinned_request_cost`).
`kernel/test_nvme_residency.py::test_pinned_landing_charge_matches_the_model_on_this_box` checks it on any box with
CUDA and a readable cgroup charge. It passed on this A2000.

**cgroup v2 (lane K29, 2026-10-04):** a rented RTX 5090 container (kernel 6.8.0-138, driver 590.48.01) read r = 1.0048–1.0103 over 18 pinned rows, with a pageable slope of 1.0021. That is CONFIRMED by the registered rule. See `../RESULTS-k29-pinned-charge-cgroup-v2.md`.
