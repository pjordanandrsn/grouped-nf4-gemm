# PREREG — does any of today's KV characterization survive a different card?

**Tier: CONFIRMATORY. Status: STAMPED before any pod was created.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `0cc54fe`,
e4b `claude/e4b-gemma-inflight-d41f93` @ `5594538`. Both local, unpushed.

## Why rent now, having declined twice today

Earlier refusals were correct: D1's rental was conditioned on D1 confirming (it
did not), and rung two was low-value once rung 1.5 tested the mechanism it
guarded. **This is a different question**, and it exists because of what the day
produced rather than in spite of it.

Every number in findings #13–#18 came off one A2000 that is **shared with a
home-lab service, memory-constrained (~8.6 GB free), and carries ±12%
run-to-run variance** — measured, not assumed: F1a re-ran 1.887 → 1.679, and the
same dequant sweep put its argmax at three different configs. And one result is
openly unexplained: the fused dequant sits at **113 GB/s against a ~287 GB/s**
byte-count ceiling, after two diagnoses (strided stores: +13%; codebook gather:
**−18%**, wrong).

A different card is the cleanest instrument for both.

## Phase 1 — kernel only, no model, no download

Synthetic tensors. This is the whole reason the run is cheap, and it carries the
prediction that matters.

- **R1a — the diagnostic.** Fused dequant bandwidth on an A100-80G-PCIe
  (~1935 GB/s HBM2e vs the A2000's ~288). If the kernel is **bandwidth-bound**,
  it scales with the card and lands **≥ 400 GB/s**. If it is **latency- or
  ALU-bound**, it stays near the A2000's 113 regardless of the memory system.
  *Confirmed ≥ 400; falsified below 250.* Either verdict resolves the
  unexplained headroom: confirmation says the A2000 was the limit, falsification
  says the kernel is and names the next thing to fix.
- **R1b — gate.** Still bit-identical to `dequant_kv_ref` on the rented device,
  same shapes as H1b. A kernel correct on one GPU and not another is a worse
  problem than a slow one. *Any mismatch voids R1a.*
- **R1c.** The link measures **≥ 20 GB/s** pinned H2D (`pcie_probe.py`), i.e.
  the box really is gen4 ×16 class and not a downgraded slot. *Falsified below
  15* — and if it fails, every Phase-2 streamed number is void, because the
  transfer term would not be the one being tested.
- **R1d.** Variance: median absolute deviation over 7 repeats < **4%**, against
  this A2000's ±12%. *Falsified above 8%.* If a dedicated card is not quieter,
  the noise was never the sharing and every band today was mis-attributed.

## Phase 2 — end-to-end, only if Phase 1's gate passes

OLMoE-1B-7B (13 GB, the same model as #17/#18, for direct comparability).

- **R2a.** NF4 resident / bf16 `DynamicCache` at ctx 4096 ∈ **[1.05, 1.30]**,
  bracketing the A2000's **1.133**. *Falsified outside [1.00, 1.60].* This asks
  whether #18's headline is a property of the cache or of the A2000.
- **R2b.** Prefetch still loses at 4096 (transfer share stays low on a fast
  link — it should get *worse* for prefetch, not better, since the link is 4×
  quicker): prefetched/streamed **> 1.00**. *Falsified at ≤ 0.95* — which would
  mean the crossover is not a transfer-share law but something device-specific.

## Cost, and the discipline around it

A100-80G-PCIe at **$1.19/hr**. Phase 1 needs no download and should take ~10
minutes; Phase 2 adds a 13 GB fetch. **Hard cap: 1 hour**, enforced by a
delete-only backstop armed at creation and independent of this session — the
failure mode the notes record is a session dying with a pod still billing.
Teardown is verified by querying the API for zero pods, not by assuming.

## Pre-committed decisions

- **R1a confirmed** → the 113 GB/s was the A2000's memory system and the kernel
  needs no further work; #18's "unexplained headroom" is closed as explained.
- **R1a falsified** → the kernel is the limit, it is portable, and the next
  optimization target is named on evidence instead of guessed at for a third
  time.
- **R2a outside its interval** → #18's 1.13× is device-specific and gets that
  qualifier wherever it appears, exactly as #17's number did.
- Phase 2 does **not** run if R1b fails or R1c falsifies.

## Outcome — the characterization is device-independent; the kernel diagnosis is not resolved

**Substitution disclosed:** no A100-80G-PCIe was available. Ran on an
**A100-SXM4-80GB** (~2039 GB/s HBM vs the registered PCIe part's ~1935), taken
first from an ordered candidate list. Faster memory makes R1a's ≥400 threshold
*easier*, not harder, so the substitution cannot have caused its outcome.

| prediction | predicted | measured | verdict |
|---|---|---|---|
| R1b **gate** bit-identical on new silicon | exact | exact, 4 shapes × 2 dtypes | **CONFIRMED** |
| R1a fused dequant bandwidth | ≥ 400 GB/s | **276.5** | **outside interval** (falsify was < 250) |
| R1c link, pinned H2D | ≥ 20 GB/s | **18.89** | **outside interval** (falsify was < 15) |
| R1d variance, MAD over 7 | < 4% | **0.28%** | **CONFIRMED** |
| R2a NF4/bf16 @4096 | [1.05, 1.30] | **1.144** | **CONFIRMED** |
| R2b prefetch still loses @4096 | > 1.00 | **1.047** | **CONFIRMED** |

**R2a is the result worth having.** The A2000 measured **1.133**, this A100
measures **1.144** — a 1% difference across a 7× gap in memory bandwidth, a
different link generation and a different vendor SKU. And the decomposition
travels too: wrapper **1.002 / 0.987**, dequant **1.142 / 1.148**. #18's headline
is a property of the cache, not of the A2000, and needs no device qualifier.

**R1d settles where today's noise came from.** MAD **0.28%** against the A2000's
±12%. Every band stated across #13–#18 is a property of a shared,
memory-constrained card, not of the measurements. Numbers taken here are worth
three digits; numbers taken there are worth two.

**R1a and R1c both landed between their thresholds, and neither pre-committed
decision fires.** Recorded as such rather than resolved by picking the nearer
edge after the fact. What the number *shows*: the kernel went 113 → 276.5 GB/s
on a memory system 7.1× faster — it captured **~35% of the improvement**, so it
is partly bandwidth-bound and mostly not. The headroom is still unexplained,
but it is now bounded on both sides: neither "the A2000 was the limit" (it would
have scaled) nor "the kernel is entirely the limit" (it would not have moved).
The reference dequant, by contrast, scaled 6.7× — consistent with it being purely
throughput-bound, which is what a seven-intermediate implementation should be.

**And prefetch's "transfer share" framing is wrong as stated.** P1 bracketed the
crossover between 17.8% and 46.7% share. This run breaks that:

| device | ctx | transfer share | prefetch |
|---|---:|---:|---|
| A2000 | 4096 | 17.8% | **loses** (1.096) |
| A100 | 16384 | **17.3%** | **wins** (0.889, 64.1% hidden) |

Nearly the same share, opposite outcomes. Share alone does not predict it. The
plausible reason is that prefetch's *cost* (an extra allocation and a full-size
concatenation) scales with **HBM** while its *benefit* scales with **PCIe**, and
the A100's PCIe:HBM ratio is 2.3× more favourable — but that is a mechanism
argument, and this document set has falsified every mechanism argument it has
tested today. It is offered as a hypothesis, not a finding, and P1's bracket is
withdrawn rather than replaced.

**Cost:** one pod, ~30 minutes, **$0.70**. Terminated; zero pods verified by API
query, not assumed.
