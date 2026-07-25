# PREREG — do streamed weights and a streamed KV cache share a link, or fight over it?

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `a558594`, e4b @ `1ad6604`.
Both local, unpushed.

## The gap this closes

The flagship streams **weights** over PCIe. Findings #13–#18 measured streaming
the **KV cache** — and did so deliberately with weights **resident**, because
E1's design note said streamed weights "would contend for the same link and make
the KV term unattributable". That was right for isolating KV. It also means the
composed case — both streaming, one link — is the one thing never measured, and
it is exactly what a "235B at 32K on 16 GB" claim would rest on.

**The arithmetic that makes it worth testing.** At 32K the flagship's weight
stream is ~11.7 GB/token and NF4 KV adds ~1.65 GB/token. If they simply add,
that is ~2.0 tok/s against the flagship's 2.29 at seq-512 — **~13% for 64× the
context**, with the KV held off-device entirely so the working set stays ~15.2 GB.

**Why this is a scout and not the flagship run.** Qwen3-30B-A3B is the same
composition at 1/8 the download: 48 layers, 96.0 KB/token, GQA 8:1, and ~17 GB
of NF4 weights — large enough that weights *must* stream on a 16 GB budget. If
additivity holds here it will hold at 235B; if the streams interfere, that costs
a dollar to learn instead of an afternoon.

## Arms — a 2×2, which is what makes the decomposition possible

| arm | weights | KV |
|---|---|---|
| `neither` | resident | resident | floor
| `W-only` | **streamed** | resident |
| `KV-only` | resident | **streamed** |
| `both` | **streamed** | **streamed** |

All four use NF4 KV, so quantization is held constant and only *residence*
varies.

## Predictions

- **M1a — the whole question.** Additivity:
  `(both − neither) / [(W-only − neither) + (KV-only − neither)]` ∈
  **[0.85, 1.20]**. *Falsified outside [0.70, 1.60].* Below 0.85 means the two
  transfers overlap and the link is not the shared bottleneck it appears to be;
  above 1.20 means they interfere and a composed claim costs more than its parts.
- **M1b — gate.** Greedy ids identical between `neither`/`KV-only` and between
  `W-only`/`both`. Residence must not change values; only quantization does, and
  quantization is constant across all four. *Any mismatch voids M1a.*
- **M1c.** The `both` arm's peak VRAM is **< 10 GB** — i.e. the composed config
  is a 16 GB-card configuration with headroom, not a 24 GB one.
  *Falsified above 14 GB.*
- **M1d.** Measured at ctx 4096 and 32768, additivity does not degrade with
  context: `M1a(32768) ≥ M1a(4096) − 0.20`. *Falsified below.* Both transfer
  terms grow with context, so if they contend, contention should worsen.

## Pre-committed decisions

- **M1a confirmed** → the transfer law composes, the flagship-with-context claim
  is arithmetic rather than hope, and the 235B run is worth its download.
- **M1a falsified high** (> 1.20) → the streams interfere; the composed claim is
  *not* the sum of its parts and must be measured per configuration rather than
  derived. The flagship run still happens, but no number is projected first.
- **M1a falsified low** (< 0.70) → better than predicted and more interesting
  than a confirmation; something overlaps that the single-link model says cannot,
  and that gets its own investigation before anything is claimed.
- **M1c falsified** → the composition is a 24 GB story, not a 16 GB one, and the
  headline changes accordingly.

## Confounds, stated in advance

1. **One device, and it is a datacenter card.** The A100's host link measured
   18.89 GB/s; a consumer gen4 ×16 board is nearer 25 GB/s theoretical and often
   less in practice. Additivity is a *shape* claim and should transfer; the
   tok/s figures will not, and are reported as the scout's numbers rather than a
   consumer projection.
2. Qwen3-30B is GQA 8:1 against the 235B's 16:1, so its KV per token is
   relatively larger. That makes this scout **harder** on the KV side than the
   flagship, not easier.
3. This measures decode only. Prefill streams the same weights but pays the KV
   write rather than the read, and is not covered.
