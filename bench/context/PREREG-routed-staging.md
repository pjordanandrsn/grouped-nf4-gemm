# PREREG — does staging only the routed experts actually pay?

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
Code: gnf4 `kernel/nf4-kv-cache` @ `1f89187`, e4b @ `79411db` (routed staging
verified 5/5, full suite 34 passed). Both local, unpushed.

## What #21 established, and what it did not

#21 located the 235B decode cost: the offload pre-hook stages a layer's **entire**
expert stack (128) while the step routes to **8** — 16× the bytes routing needs.
Neither prefetch (1.11×) nor the grouped kernel (1.6%) could touch it, because
both optimize *around* a transfer that is 16× too large.

`enable_routed_staging` makes the copy follow the router. It is **verified
correct** — bit-identical to bulk — and **entirely unmeasured for speed**. The
arithmetic says 0.37 s/token where bulk measured 5.14. That arithmetic is the
same class of claim that was **10.7× wrong** in `PREREG-planner-validation.md`
three hours ago, which is the reason this document exists rather than a README
edit.

## Fixture

Qwen3-235B-A22B, 2×A100-SXM-80GB (for host RAM), NF4 experts pinned, KV NF4
host-resident, greedy, 12 new tokens, median of 2, **one process** so both arms
share a load. `prefetch=False` throughout — routed staging refuses prefetch-linked
handles, and that refusal is structural, not a tuning choice.

- `bulk` — today's path. Reproduced at **5.782 s/token** on two independent pods.
- `routed` — `enable_routed_staging(handles)`, same process, same weights.
- `routed` at ctx 32768.

## Predictions

Grounded in the byte model **as corrected by #21**, where full-stack bytes/link
predicted 5.849 s against 5.144 measured (ratio 1.14). Routed bytes are
7.98 GB; at a measured ~21.8 GB/s that is 0.37 s, plus `c_box`.

- **R1a — the win.** `routed` / `bulk` s/token ∈ **[0.08, 0.30]**. *Falsified
  above 0.50* (less than a 2× win would mean the byte accounting is still wrong)
  *or below 0.05* (faster than the bytes allow ⇒ something is not being copied).
- **R1b — GATE, correctness at scale.** Greedy token ids from `routed` are
  **identical** to `bulk`. The unit suite proves bit-identity on a 16-expert toy;
  this proves it on 94 layers × 128 experts where an uncopied row would actually
  be reachable. *Falsified by any divergence, which voids R1a* — a faster wrong
  answer is not a result.
- **R1c — memory is not the trade.** `routed` peak ≤ **1.10×** `bulk` peak. The
  destination keeps the full `[E, …]` shape, so allocation should be unchanged;
  a rise would mean the arena path is leaking a second copy.
- **R1d — the planner's model becomes applicable.** Measured `routed` s/token
  within **±40%** of `c_box + routed_bytes/link`. The planner's time model was
  suppressed because it assumed routed-only bytes on a bulk path. If the path is
  now genuinely routed, the model should finally describe the system.
  *Falsified outside ±60%.*

## Pre-committed decisions

- **R1a confirmed and R1b holds** → routed staging becomes the documented default
  for streamed inference, and #21's "the fastest configuration is unreachable by
  construction" is **retired**: it is reachable, and it does not need the grouped
  kernel or hot residency to get there.
- **R1d confirmed** → `plan.py` un-suppresses throughput **for routed configs
  only**, still as a band, and the suppression stays for bulk paths. Un-suppressing
  everything on one arm's evidence is the mistake that produced the 10.7×.
- **R1a falsified above 0.50** → the 16× byte reduction did not convert into time,
  which would mean the step is not link-bound after all and #21's attribution —
  though arithmetically exact to 1.14 — identified a correlate rather than a cause.
- **R1b falsified** → routed staging is withdrawn, not patched, until the unit
  suite is extended to whatever the real model exposed that the toy did not.

## Confounds, stated in advance

1. Routed staging adds a **per-layer host sync** (`torch.unique(...).tolist()`) and
   issues ~32 small copies per layer instead of 4 large ones. Both are unmeasured
   overheads that the byte model does not contain, and both push the same way
   (against R1a). If R1a lands near its upper bound, these are the first suspects.
2. Prefetch is **off in both arms**, so `bulk` here is the 5.782 s/token path, not
   the 5.144 s prefetch-on one. Comparing routed against the *faster* bulk arm
   would be the fairer headline and is computed too, but R1a is scored on the
   same-process same-config pair.
3. One model, one box, one link speed.

## Outcome — 5.95×, correctness gate held, and the model still does not describe it

Qwen3-235B-A22B, 2×A100-SXM-80GB, link **22.21 GB/s**, `prefetch=False` both
arms, one process, 94 offload handles, greedy, 12 new tokens, median of 2.

| arm | ctx | s/token | tok/s | peak |
|---|---:|---:|---:|---:|
| `bulk` | 512 | 5.570 | 0.180 | 18.62 GiB |
| **`routed`** | 512 | **0.936** | **1.068** | 18.62 GiB |
| `routed` | 32768 | 0.986 | 1.015 | 26.81 GiB |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| R1b **GATE** greedy ids identical | identical | **identical** | **CONFIRMED** |
| R1a routed/bulk | [0.08, 0.30] | **0.1681** (**5.95×**) | **CONFIRMED** |
| R1c peak ratio | ≤ 1.10 | **1.000** | **CONFIRMED** |
| R1d measured / byte model | ±40% | **2.106** | **FALSIFIED** |

**The gate is the important one.** Greedy ids are identical to bulk across 94
layers × 128 experts, where an uncopied row is genuinely reachable — the unit
suite only ever proved that on a 16-expert toy. 16× fewer bytes, same answer.

**And memory was not the trade:** peak is *identical* (1.000), because the
destination keeps the full `[E, …]` shape and only the copied rows differ.

### R1d falsified, exactly where confound #1 said to look

The byte model predicts 0.445 s; the measurement is 0.936. The residual is
**0.491 s = 5.2 ms per layer** — the per-layer host sync (`torch.unique(...)
.tolist()`) plus ~32 small copies where bulk issued 4 large ones. Both were named
in advance as unmeasured and as pushing this way. They are now the dominant term:
**more of the routed step is overhead than is bytes.**

**So the pre-committed un-suppression does NOT fire.** `plan.py` keeps throughput
suppressed. The model was wrong by 10.7× on the bulk path and is wrong by 2.1× on
the routed one; 2.1× is a large improvement and still not something to quote.

### What fires

- **R1a + R1b → routed staging is the documented default for streamed
  inference**, and #21's "the fastest configuration is unreachable by
  construction" is **retired**. It is reachable, and it needed neither the grouped
  kernel nor hot residency to get there.
- Still **4× short of the stamped 4.3–4.4 tok/s.** Routed staging closes 5.95× of
  the 27×; the remainder is now visibly the 5.2 ms/layer overhead, which is a
  concrete, attackable target rather than an open question.

### A trade that only became visible now

At 512 the routed step is 0.936 s; at 32768 it is 0.986 — **context costs 5%**,
where #19 measured it at 0.4%. Nothing about the KV tier changed. The weight term
shrank 6×, so the same context cost is now a visible share of a smaller step.
#19's "context is free" was true *of a step dominated by 16× surplus weight
traffic*, and that qualifier belongs on it.
