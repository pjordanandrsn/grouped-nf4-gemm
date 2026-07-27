# PREREG — single-pass NF4 FlashAttention, with a stopping rule

**Tier: CONFIRMATORY. Status: STAMPED before the kernel was written.**
gnf4 @ `ae62eca`, e4b @ `3510248`. Both local, unpushed.

## The gap this closes, and the one it must not

`attend_nf4_kv_gqa` attends on packed NF4 directly — the thing this project is
premised on — and loses to bf16 SDPA by **12.7×** (D1b). The profile says why,
and it is not a tuning constant:

```
1. _kv_scores_gqa   read K packed        9.44 MB   write scores fp32  8.39 MB
2. torch.softmax    read+write scores   16.78 MB   (a separate kernel)
3. _kv_wsum_gqa     read probs 8.39 + V 9.44 MB    write partials     0.26 MB
   total 61.08 MB, of which the KV itself is 18.87 MB = 31%
```

**69% of its traffic is a materialized `[H_q, T]` fp32 score matrix** — allocated
and zero-filled per call, written once, read twice. FlashAttention never writes
it, and the bf16 baseline *is* FlashAttention. At 61.08 MB in 3.760 ms the kernel
runs at **16.2 GB/s = 6% of an A2000's ~288 GB/s**, while moving **3.2× more
bytes than the algorithm requires**.

A single-pass online-softmax kernel with the NF4 decode in the inner loop removes
the score matrix entirely, leaving only the 18.87 MB of KV — **3.56× less than
bf16 reads**. That byte advantage is why this is worth attempting at all.

## The stopping rule, fixed in advance

This is a kernel rewrite competing against a vendor implementation that is
already memory-bound at 72% of peak. It could easily become an open-ended
project, so:

- **N1a — the bar.** The new kernel reaches **≥ 40%** of achievable bandwidth on
  its own traffic (KV bytes / elapsed). *Falsified below 25%*, at which point the
  packed-KV attention path is **abandoned** and NF4 KV is documented as a pure
  memory play, not a speed one. **No further kernel work is done on it.**
- **N1b — the point of the exercise.** New / bf16 SDPA (`enable_gqa=True`) ≤
  **1.0**, i.e. it actually wins, since it reads 3.56× fewer bytes. *Falsified
  above 1.5.* Between 1.0 and 1.5 it is recorded as "close but losing" and does
  not get wired in.
- **N1c — GATE, correctness.** Relative error against `dequant→SDPA` on the same
  packed inputs < **2e-3**. This is a different arithmetic path (online softmax,
  fp32 accumulation), so a wrong answer here would be silent. *Any failure voids
  N1a/N1b.*
- **N1d — traffic, not just time.** Measured bytes moved ≈ KV bytes to within
  **1.3×**, confirming the score matrix is actually gone rather than merely
  smaller. *Falsified above 2.0×.*

## Pre-committed decisions

- **N1a and N1b confirmed** → wire it into the attention path (patching the
  attention module, since the transformers Cache protocol cannot express this),
  and #37's "bf16 unless VRAM binds" is revisited with data.
- **N1a confirmed, N1b in [1.0, 1.5]** → keep the kernel, do **not** wire it in;
  NF4 KV remains a memory play and the kernel is a foundation for a later attempt.
- **N1a falsified** → stop. The path is abandoned, and that is recorded as a
  negative result rather than left as an open TODO.

## Confounds

1. Written against one fixture (T=32768, GQA 16:1, D=128) on one card. A decode
   kernel that wins only at one T is not a win.
2. bf16 SDPA dispatches to FlashAttention via `enable_gqa=True`; that is the
   correct baseline (#12's original comparison used `repeat_interleave` and was
   10.7× flattering — the error this prereg is explicitly avoiding).
