### `NF4_QLORA_COMPACT_DELTA=1`: the compact delta's backward frees each intermediate at its last use (opt-in; default unchanged)

- **Why.** experts4bit-qlora's TC1 amendment 36 measured the flag on an RTX 5090 (box `tc1-5090-80`, whole-layer gradient
  checkpointing). Both arms stepped faster, but the matched arm's peak (fp32 adapters) rose 27.49 → 27.72 GB, **+0.23 GB**; the
  shipped arm's (bf16 adapters) was unchanged. The registration had predicted a lower peak.
  - `_CompactPaddedDelta.backward` held every intermediate until it returned. The padded output grad `gd` `[G, W, N]` was still
    live while the block `x` and its grad `gx` (both `[G, W, K]`) were rebuilt and computed. The autograd path frees its `gd`
    before `gx` exists.
- **What.** The backward drops (`del`) each intermediate after its last use:
  - the gathered `B` after `gh`, and `gd` after `gBt`;
  - the gathered `A` after `gx`, and `x` and `gh` after `gAt`;
  - `gx` once `grad_a` is gathered (before its dtype cast), and `gAt` and `gBt` once the adapter grads are scattered.

  Every op, operand layout and dtype, and the order of the calls, are unchanged. The incoming grad stays held by autograd until
  the node returns, so it is not dropped.
- **Values.** Forward and every gradient stay `torch.equal` to the autograd path, in the existing grid and at every cell below.
- **Allocator peaks, read on an RTX A2000** (torch 2.8.0+cu128; the A2000 is a correctness testbed, and allocator bytes do not
  depend on contention). One `lora_delta_grouped` call per cell: 128 experts, top-8, r=16, gate_up K=2048 → N=1536 and down
  K=768 → N=2048, bf16 activations, `GNF4_HOST_REUSE` on.
  The router has the skew of `bench/host-reuse/` (logits `randn + linspace(1.5, -1.5)` over the experts). Peaks are
  `max_memory_allocated` above what was allocated before the forward, in MiB, and were identical in two runs.

  | projection, tokens, adapters | G × widest | backward peak: autograd / compact before / compact after | forward + backward peak |
  |---|---|---|---|
  | gate_up, 380, fp32 | 106 × 112 | 264.2 / 360.6 / **227.9** | 264.2 / 360.6 / **227.9** |
  | down, 380, fp32 | 106 × 112 | 193.1 / 250.4 / **136.2** | 193.1 / 250.4 / **170.2** |
  | gate_up, 380, bf16 | 106 × 112 | 120.2 / 186.2 / **114.0** | 120.2 / 186.2 / **114.0** |
  | down, 380, bf16 | 106 × 112 | 92.1 / 127.4 / **68.1** | 92.1 / 127.4 / **85.1** |
  | gate_up, 1100, fp32 | 114 × 312 | 717.4 / 973.1 / **636.6** | 717.4 / 973.1 / **636.6** |
  | down, 1100, fp32 | 114 × 312 | 514.8 / 637.2 / **370.6** | 514.8 / 637.2 / **472.6** |
  | gate_up, 1100, bf16 | 114 × 312 | 324.3 / 483.3 / **318.3** | 324.3 / 483.3 / **318.3** |
  | down, 1100, bf16 | 114 × 312 | 244.5 / 323.2 / **185.3** | 244.5 / 323.2 / **236.3** |

  - **Before**, the compact backward peaked 24–55 % above the autograd path's, with every intermediate live at once at the return.
    **After**, it is 2–30 % below in every cell.
  - **Where the peak now sits.** For gate_up it is `x` and `gx` together, the same two blocks the autograd path holds in that
    bmm's backward (read with `torch.cuda.memory._record_memory_history`). For down it is `gd` with the gathered adapters, and the
    forward's peak is the higher one.
  - **A harsher router.** With one expert taking every token (G = 128, widest = tokens), compact after is also at or below the
    autograd path in all eight cells. The closest is gate_up, bf16, 1,100 tokens: 1,152.1 / 1,623.2 / 1,148.4 MiB.
  - **Time is not read here.** The A2000 is a correctness testbed only, so this entry carries no timing. The RTX 5090 reads
    are experts4bit-qlora's TC1 amendments 36 and 37: the full Qwen3-30B-A3B training step with the flag on runs 0.969 / 0.967
    of the flag-off step on the matched arm and 0.970 / 0.948 on the shipped arm, before and after this patch.
- **The training peak, measured since:** experts4bit-qlora's TC1 amendment 37 (`tc1-5090-83`, a second host) read the matched
  arm's peak with the flag on at 27.189 GB against 27.477 GB off, **−0.288 GB**, where amendment 36 had read +0.229 GB without
  this patch. The flag stays opt-in; the default decision is proposed as TC1 amendment 38
  (pjordanandrsn/experts4bit-qlora#1125).
- **Not in this change.** Computing `gAt` before `gx` would free `x` before `gx` exists. In the same cells, gate_up's backward peak
  would fall a further 30–33 % below this patch (227.9 → 158.9 MiB at 380 tokens, fp32), with values bit-identical. The forward
  peak then becomes the binding one (204.0 MiB in that cell). It reorders two calls, so it is left to a separate change.
- **Tests** (`kernel/test_compact_delta.py`). `test_compact_backward_peak_at_most_autograd` is CUDA-only. It runs one forward +
  backward at the hot-expert case, with gate_up-like and down-like widths, bf16 and fp32 adapters, and `GNF4_HOST_REUSE` off and on.
  It asserts `torch.equal` on the output and all three gradients, and that the compact forward, backward and combined peaks are at
  most the autograd path's.
  - Against the unpatched backward it fails 7 of 8 cases under torch 2.8 and all 8 under torch 2.11.
  - On the A2000, with the patch: `test_compact_delta.py` gives 71 passed and 20 skipped (host reuse on CPU), and
    `test_lora_delta_lean.py` 104 passed. With `test_host_reuse.py`, `test_nf4_qlora_grad.py` and `test_eids_forms.py` added, the
    run gives 220 passed and 21 skipped, both with `NF4_QLORA_COMPACT_DELTA` unset and with it set to 1.
