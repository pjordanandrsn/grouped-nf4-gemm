# Giant-MoE shapes exploratory (Qwen3-235B / Qwen3-Coder-480B / Llama-4-Maverick)

**2026-07-13 · EXPLORATORY** (shapes chosen after four confirmatories — not
blind, no registered criteria; n=3 fresh-process reps on a fresh SECURE
A5000, decode bs1, frozen v4 kernel `80bda12`, energy window 5 s; the A2000
side-leg was skipped — Maverick's 5.4 GB packed stack doesn't fit 12 GB and
the host was having an ssh evening).

Question under test: does the fused kernel hold (or improve) on the
largest-expert MoEs — the 200B+ class — on a small card? Per-layer expert
stacks all fit trivially (0.4–5.4 GB packed):

| cell (per-layer stack) | traffic/call | vs dequant med / worst | energy (fused/deq) | fused more accurate |
|---|---|---|---|---|
| Qwen3-235B-A22B gate_up (3072×4096, E128 k8) | 57 MB | **1.45× / 1.38×** | 0.69 | yes (3/3) |
| Qwen3-235B-A22B down (4096×1536) | 28 MB | **1.62× / 1.57×** | 0.59 | yes |
| Qwen3-Coder-480B gate_up (5120×6144, E160 k8) | 142 MB | **1.31× / 1.30×** | 0.78 | yes |
| Qwen3-Coder-480B down (6144×2560) | 71 MB | **1.19× / 1.14×** | 0.77 | yes |
| Llama-4-Maverick gate_up (16384×5120, E128 k1) | 47 MB | 1.05× / 0.97× | 0.83 | yes |
| Llama-4-Maverick down (5120×8192, k1; plan splits sk4) | 24 MB | 1.01× / 1.01× | 0.78 | yes |

Readings:

1. **The 200B+ k=8 class is a clean win**: Qwen3-235B at 1.45–1.62× median
   with energy at 0.59–0.69× the dequant path — consistent with the
   bandwidth-bound-regime claim, on the exact model class asked about.
2. **"Bigger is better" saturates rather than grows**: the 480B cells (the
   largest per-call traffic ever measured here, 2.6× the census max) win at
   1.19–1.31×, *below* the mid-size census band (1.16–2.73×). At extreme
   traffic both paths are pure-bandwidth and the ratio compresses toward the
   bounded advantage of skipping the bf16 materialization round-trip; the
   fused speed multiple peaks in the mid-band. Energy stays strictly below
   throughout — that part does not compress away.
3. **Maverick behaves exactly as the k=1 law predicts**: parity speed on
   this instance (its class is instance-unstable, documented since v3),
   split-K delivering its paired gain on `down` (nosplit/fused = 1.56×),
   and — notably — **energy still 0.78–0.83× the dequant path even at speed
   parity** (fewer bytes moved is fewer joules regardless).
4. Fidelity ordering held on all 6 cells × 3 reps.

Sub-100 GB context (the question behind the question): these are PER-LAYER
kernel results. Whole-model NF4 residency for Qwen3-235B is ~128 GB
(experts + absmax) — it does not fit a sub-100 GB card resident; fitting it
on a 24 GB card is the expert-offload product this kernel is the compute
half of (active working set per token ≈ the ~22B active params).
