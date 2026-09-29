# MXFP4 prefill combine on the NAS RTX A2000: repeat, order and cost (#408, #410)

Kernel-level receipts for #410, which replaces `Mxfp4PipelinedGptOss._forward_prefill`'s
`out.index_add_(0, rows, ...)` with `_index_add_ordered_`. Claim
`gnf4.kernel.mxfp4-prefill-combine-ordered.a2000.2026-09-28`. The model-level A/B (Kimi-K3, nine
processes) is experts4bit-qlora's, in `bench/kimi-k3-a2000/` (RESULTS §5, receipts `2026-09-28-det-ab/`).

**Where.** The NAS RTX A2000 12 GB (sm_86, driver 575.64.05) inside the `gpu-dev` container, 2026-09-28.
torch 2.8.0+cu128 and triton 3.4.0 in both venvs used; pytest 9.1.1.

| file | what |
|---|---|
| `combine_repeat.py`, `combine_repeat.log` | The first read (log written 21:38:12Z), on the shipped 0.33.4 stack. The combine loop is replayed at Kimi-K3 geometry (topk 16 of 896, 16 slots per chunk, width 7168) on identical inputs, 50 calls per arm. Arms: the shipped `index_add_`, the same under `torch.use_deterministic_algorithms(True)`, and a first ordered-add draft that uses `cummax` where #410 uses `searchsorted`. T = 6 and 90. |
| `combine_pr.py`, `combine_pr.log` | The same replay with #410's own `_index_add_ordered_`, imported from the PR tree at `f180045`, at T = 6, 90 and 512, plus the combine's time (CUDA events, median of 20 after 3 warm-ups). |
| `gnf4_pr_tests.log` | `cd kernel && python -m pytest test_mxfp4_prefill_combine.py test_mxfp4_pipelined.py -v`, from the PR tree at `f180045`, 22:17Z: 28 passed. |
| `mustfail.py`, `gnf4_pr_mustfail.log` | The new tests with `_index_add_ordered_` swapped for the plain `index_add_`. 4 of 5 CUDA cases fail. The `(1, 7)` sequential-loop case passes: all seven terms on one row happened to land in index order on this card. That case is there to exercise seven passes, not to catch the race. |

**Read with care.**

- **`combine_pr.log` is a rerun.** Its first run, inside the K3 window at 22:17Z, died at once on an
  argument bug in the script (`chunks` was not passed). That runner's status line still says `rc=0`,
  because it logged `$?` after a `$(date …)` in the same string. The script was fixed and rerun after the
  window closed (log written 22:36:28Z). The sdxl sidecar was resident on the card then, at 0 % GPU
  utilization when the run started and 9 % when it ended, so the times are indicative. Load can move
  the times and the shipped arm's spread, not the ordered arm's bits: no two of its threads share an address.
- In `combine_repeat.log` the ordered arm is the draft, not #410's code. `combine_pr.log` is the
  reading of #410's code. Both agree.
