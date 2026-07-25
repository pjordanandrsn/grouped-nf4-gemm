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
