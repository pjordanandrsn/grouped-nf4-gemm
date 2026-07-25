# PREREG — closing the line on a card whose noise is known

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `15b1f6c`, e4b @ `5594538`.
Both local, unpushed.

## Why this run, having refused twice and accepted once

Three questions remain, and all three are now **unanswerable on the A2000** for a
reason that is measured rather than suspected: that card produced a ±12% band, a
physically impossible G1 row (wrapper faster than the reference on identical
data), a physically impossible K2 sweep (prefetch faster than a fully resident
cache; resident time falling as context rose), and two runs of identical code
disagreeing about the **sign** of prefetch's effect. The A100 measured **MAD
0.28%** on the same work.

This is the last thing money can buy on this line.

## L1 — prefetch's crossover, on an instrument that can see it

K2 was void, not falsified, so the question is untouched. Same four contexts,
quiet card.

- **L1a — gate.** The arms must be **physical**: resident step time strictly
  increasing in context, and prefetched ≥ resident at every context. *Any
  violation and L1b/L1c are void*, exactly as K2 was — this is the check K2
  failed, promoted to a gate so it cannot be discovered afterwards.
- **L1b.** Fitting `prefetch_cost = c + m·ctx` over {2048, 4096, 8192, 16384}
  gives a **fixed term c > 0**. *Falsified at c ≤ 0*, which kills the
  fixed-cost explanation and leaves prefetch with no model at all.
- **L1c.** The crossover implied by the fit falls **between 4096 and 16384**,
  where the A100 was observed to flip (lost 1.047 at 4096, won 0.889 at 16384).
  *Falsified outside.*

## L2 — R1a, with the stopwatch K1 exposed

R1a measured 276.5 GB/s on the A100, synced, at T=4096 — the shape K1 showed is
~50% launch overhead, and the smallest problem is where a big card is furthest
from saturation. Both biases ran against the A100. Amortized, the A2000 reaches
**86% of its memory bandwidth**.

- **L2a.** Amortized peak dequant bandwidth on the A100 ≥ **800 GB/s** (~39% of
  its ~2039). *Falsified below 400* — which would mean the kernel genuinely does
  not scale past the A2000's absolute figure and something portable limits it.
- **L2b — gate.** Bit-identical to `dequant_kv_ref` on this device.

## L3 — does NF4's cost actually rise with context?

#17 claimed it does and #18 softened it. The two cards disagree about the slope:
A2000 said 1.133 → 1.244 across 4K→16K; the A100 said 1.144 → 1.158. Given what
is now known about A2000 noise, the A100's shallow slope is the credible one and
the steep one was probably never real.

- **L3a.** NF4/bf16 rises by **≤ 0.10** from ctx 4096 to ctx 32768.
  *Falsified above 0.20.* Confirmation would mean the "cost grows with context"
  warning in #17 is real but mild; falsification would mean it is steep and the
  warning stands as written.

## Cost and teardown

A100-class, **hard cap 45 minutes**, delete-only backstop armed at creation and
independent of this session. Expected ~$0.80. Teardown verified by API query
returning zero pods, not by assumption. Substitutions disclosed if the
registered part is unavailable, as in the previous run.

## Pre-committed decisions

- **L1a fails** → the line closes with prefetch guidance resting on exactly two
  measured points and no model, and that is stated as the final word rather than
  chased onto a third card.
- **L2a confirmed** → R1a is resolved: the kernel is bandwidth-bound on both
  devices and the earlier cross-device puzzle was entirely instrumentation.
- **L3a falsified** → #17's context warning is restored to its original strength
  wherever #18 softened it.
