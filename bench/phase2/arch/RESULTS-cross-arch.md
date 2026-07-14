# Cross-architecture exploratory: RTX 4090 (sm_89) + H100 PCIe (sm_90)

**2026-07-13/14 · EXPLORATORY** (no registered criteria; suite + n=2 decode
reps on census + giant shapes + full 360-cell config grid + batched-decode
M-sweep per arch; fresh SECURE pods, torn down + 404-verified). The sm_86
kernel ran UNMODIFIED — same universal 64/2 constant, same dispatch plan.

## Headlines

1. **The 4090 is the kernel's best card so far.** Census decode:
   **1.18–3.72× median** (OLMoE 3.6–3.7×), energy 0.25–0.55× the dequant
   path, fidelity better in every cell. Suite 44/44. The universal 64/2
   constant is oracle-grade on the sm_89 grid (median regret 1.000,
   p95 1.053) — **no retune needed**.
2. **H100 census holds** (1.13–3.67× median, energy 0.21–0.53, suite 44/44,
   grid p95 regret 1.080 — constant fine), **but the giant/k≤2 classes
   compress harder on HBM3**: Qwen3-235B still wins (1.15–1.48×), 480B slips
   to 0.76–0.93×, and Maverick (k=1) drops to 0.44–0.49× with the first H100
   energy miss (1.40). Mechanism: the fused win is the eliminated
   materialization round-trip; at 2 TB/s that round-trip costs the baseline
   little, and 132 SMs deepen k=1 starvation. The saturation law from the
   giant exploratory, amplified.
3. **Batched decode (M=2–32 per group): the fused M-tile path holds the
   whole continuous-batching band on narrow/medium-K experts** — 4090:
   1.3–2.4× flat across M; H100: ~1.0–2.1 decaying gently. The gpt-oss-wide
   class (K=2880) loses at every M on both (0.4–0.76) — the same mainloop
   weakness the prefill work identified, now mapped across the batch axis.
4. RTX 5090 was attempted twice; both SECURE pods wedged in provisioning
   (created RUNNING, never got a public IP) — deleted, unread, deferred.

## Files

`arch4090/`, `archh100/`: `arch_dec_r{1,2}.json` (census+giant decode),
`arch_grid.json` (360-cell config grid), `arch_msweep.json` (M∈{2..32}),
`arch_suite.log`. Sweep tool: `bench/phase2/decode_config_sweep.py`;
M-regime: `decode_m<N>` in the harness.

## Implication for the claim

The bandwidth-bound census claim **travels across three architectures
without retuning** (sm_86 → sm_89 → sm_90), strongest on consumer cards —
which is the product's home turf. The known-loser classes shrink or grow
with memory bandwidth exactly as the saturation model predicts; on
datacenter HBM parts the honest positioning is census/mid-size shapes and
the energy win, not the giants.
