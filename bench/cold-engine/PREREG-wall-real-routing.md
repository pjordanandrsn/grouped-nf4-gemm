# Preregistration — does the R2 wall null survive REAL routing?

Registered before the box is rented and before any gpt-oss routing is
captured.

## Why this and not another model capture

Three predictions from [`PREREG-tribrid-stage3.md`](PREREG-tribrid-stage3.md)
and its successors were scored offline on captured traces, and
[`structural_check.py`](routing-trace/structural_check.py) now shows all three
are **forced by the allocator, not by routing**:

| claim | synthetic conditions swept | verdict moved in |
|---|---|---|
| cache loses below one step, wins at it | 24 | **0** |
| LRU/FIFO zero-hit below one step, random not | 24 | **0** |
| `headroom ≤ 1` ⇒ demand beats static, no false positives | 27 | **0** |

Stickiness from 0 (independent draws, no temporal structure) to 0.95, expert
popularity flat to Zipf-3. Nothing moves them. **No captured model can test a
claim in that table**, which is why
[`PREREG-fourth-model.md`](routing-trace/PREREG-fourth-model.md) was withdrawn.

What is **not** forced is wall time. Transfer counts are arithmetic; whether
avoided transfers become saved milliseconds depends on PCIe, NVMe, and kernel
overlap, and it is measurable only on hardware.

## The gap this closes

[`RESULTS-r2-wall.md`](RESULTS-r2-wall.md) refuted R2 — resurrections reached
5.37–15.72% of routed work and moved wall by nothing. That measurement drives
the engine with `routes()`, which draws **fresh `torch.randn` logits every
step**. Routing is therefore independent across steps and step-to-step reuse
sits at chance, `k/E` = 4/32 = **12.5%**.

Every captured real trace has far more:

| trace | lag-1 overlap | ÷ chance |
|---|---|---|
| OLMoE | 41.3% | 3.30× |
| Granite | 43.6% | 2.18× |
| Qwen1.5-MoE | 13.4% | 2.01× |

So R2's null was measured on the routing **least** favourable to the mechanism
it was testing. That is not a reason to doubt the null; it is a reason it has
not yet been tested where the mechanism has something to work with.

## Setup

Same arena and engine as `RESULTS-r2-wall.md`: **openai/gpt-oss-20b** baked
with `nvme_arena.bake`, 24 layers × 32 experts, 13.22 MB per expert row,
10.15 GB. One layer, decode-shaped (T=1).

The routing sequence is the only thing that changes, and it comes from
**gpt-oss-20b itself** — the same model the arena was baked from, so E = 32,
k = 4, no id remapping and no cross-model substitution.
`capture_routing.py`, four prompts, 512 steps, tokens recorded.

**Paired**: every `(rows, protected)` point is run with the synthetic sequence
and the captured one, same arena, same process, alternating, **5 repeats**
each (the published run used 2, which is too few to call a null).

* `rows` ∈ {12, 16, 24, 32, 48}, `protected` ∈ {rows//4, rows-k} — the
  extremes of the published sweep. `rows-k` is now known to be the only
  correct setting: the margin `rows - protected` must be at least k to serve
  an all-miss step and at most k or the cache thrashes
  ([`RESULTS-third-model.md`](routing-trace/RESULTS-third-model.md)).
  `rows//4` is the deliberately-thrashing arm, which is what makes the pair a
  resurrection on/off contrast at fixed capacity.

## W1 — real routing produces materially more resurrection

> At matched `(rows, protected)`, the captured sequence's resurrections per
> routed invocation are **at least 1.5×** the synthetic sequence's.

Confirmed if the ratio is ≥ 1.5 at every one of the five `rows` values with
`protected = rows-k`. Refuted otherwise. This is the premise check: if real
routing does not actually give the mechanism more to work with here, W2 is
uninformative and will be reported as such rather than as a null.

## W2 — the wall null survives

> With real routing, the wall-time difference between `protected = rows-k`
> (resurrections on) and `protected = rows//4` (suppressed) at fixed `rows`
> is **under 2%** of the median, and **within run-to-run spread**.

* **Confirmed** if at every `rows` the paired difference is < 2% *and* smaller
  than the max−min across that arm's 5 repeats.
* **Refuted** if any `rows` shows a ≥ 2% separation that exceeds repeat
  spread — which would mean R2's refutation was an artifact of testing the
  mechanism on routing that gave it nothing to do, and `RESULTS-r2-wall.md`
  gets a banner.
* A separation in the *wrong* direction (resurrections on being slower by
  ≥2%) is also a refutation of the null and will be reported as one, not
  filed as noise.

## What would count as a miss

* W1 fails ⇒ W2 is reported as uninformative, not as a confirmation.
* W2 fails ⇒ `RESULTS-r2-wall.md` gets a banner and R2 is reopened.
* A failed bake, a failed capture, or a router the probe cannot read is **not
  a result either way** and will be reported as a failed run.
* The box is destroyed when the run ends, and the receipts are committed
  before it is.
