# PREREG — the flagship, with context: 235B at 32K on a consumer-sized working set

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `3496676`, e4b @ `1ad6604`.
Both local, unpushed.

## The claim under test

The flagship is **Qwen3-235B-A22B decoding on ≤16 GB of VRAM** — weights
NF4-packed in pinned host RAM, streamed per token over PCIe — measured at
**4.3–4.4 tok/s at seq-512**, and re-labelled today as *"at ~5K context"*
because KV was never in the figure.

Findings #13–#18 made KV a controllable term, and the composition scout measured
weights and KV **adding** on one link (additivity 0.962 at 32K, weights
dominating 6×). This run asks whether the composed configuration is real at
flagship scale:

> **235B, 32K of context, KV entirely off-device, working set ≤ 16 GB.**

## What this run can and cannot do — stated before it runs

**The 2×2 the scout used is impossible here.** Its `neither` and `KV-only` arms
need weights *resident*, and 235B in NF4 is ~123 GiB — larger than any single
GPU on offer. **Additivity therefore cannot be decomposed at 235B**, and this
run does not attempt it. That is precisely why the scout existed, and it is the
reason its result carries the mechanism while this one carries the claim.

Feasible arms, all weights-streamed:

| arm | ctx | KV |
|---|---:|---|
| `short` | 512 | streamed | reproduces the flagship's measured point
| `long-resident-KV` | 32768 | resident |
| `long` | 32768 | **streamed** | the claim

## Predictions

Grounded in the scout's measurements, not in a mechanism — which is the one
pattern that has held all day.

- **N1a — the headline.** `long` arm peak VRAM ≤ **16.0 GB**. *Falsified above
  18.0.* The stamped flagship working set is 15.2 GB at seq-512; streaming the
  KV should hold that roughly flat while resident NF4 KV at 32K would add
  1.65 GB and break it.
- **N1b — the cost of context.** `long` tok/s / `short` tok/s ≥ **0.85**, i.e.
  64× the context for ≤15%. The scout measured **11%** on a model with a
  *worse* KV ratio (GQA 8:1 against 235B's 16:1). *Falsified below 0.75.*
- **N1c — gate.** `long` reports **0 bytes** of KV on device, and greedy decode
  is deterministic across repeats. *Either failing voids N1a/N1b* — a
  configuration that quietly keeps KV resident would satisfy N1b for the wrong
  reason.
- **N1d.** KV's share of the composed step ∈ **[6%, 25%]**, bracketing the
  scout's 11% and the byte-model's ~12%. *Falsified outside.*

## Cost, risk, and the stop rule

2×A100-80GB (**for host RAM, not GPUs** — ~123 GiB of pinned NF4 experts needs
headroom a 1× slice may not have), ~$2.78/hr, 438 GB checkpoint download, hard
cap **2 hours** on a delete-only backstop. Expected **$4–6**, an order of
magnitude above anything else today.

**Staged stop rule, to avoid paying for a run that cannot finish:** host RAM is
checked **before** the download starts, and the pod is terminated immediately if
it cannot hold the expert stack. That costs ~$0.15 to learn instead of $5.

## Pre-committed decisions

- **N1a and N1c hold** → the flagship heading changes from *"≤16 GB (at ~5K
  context)"* to a figure that carries 32K, and the README says so with the
  tok/s cost beside it.
- **N1a falsified** → the composed configuration is a 24 GB story; the heading
  keeps its ~5K qualifier and the 32K claim moves to a larger card.
- **N1b falsified** → context is expensive at flagship scale in a way the scout
  did not predict, and the scout's additivity result gets a scale caveat it
  currently does not have.

## Outcome — the gate holds, context is free, and the working-set claim falls

Qwen3-235B-A22B-Instruct-2507, NF4 experts streamed from pinned host RAM,
2×A100-SXM-80GB (used for the **2 TB of host RAM**, not the GPUs), greedy,
12 new tokens, median of 2. Load: 1186 s, 122 GB pinned.

| arm | ctx | ms/step | tok/s | peak | KV on device |
|---|---:|---:|---:|---:|---:|
| `short` | 512 | 6266.5 | 0.160 | 18.62 GiB | 0 |
| `long-resident-KV` | 32768 | 6095.7 | 0.164 | 28.46 GiB | 1774.8 MB |
| `long` | 32768 | 6124.1 | 0.163 | **26.81 GiB** | **0** |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| N1a working set | ≤ 16.0 GB | **28.79 GB** | **FALSIFIED** |
| N1b cost of 64× context | ≥ 0.85 | **1.023** | **CONFIRMED — but see below** |
| N1c gate: KV off device, deterministic | both | both | **CONFIRMED** |
| N1d KV share of step | [6%, 25%] | **0.5%** | **FALSIFIED** |

### N1b and N1d are the same quantity, and only one interval caught the error

Both describe what context costs. N1b asked for ≥0.85 and got 1.023; N1d asked
for 6–25% and got 0.5%. **I was wrong about the magnitude by 24× in both cases**
— N1b's interval was one-sided and absorbed the error, N1d's was two-sided and
caught it. So N1b is recorded as CONFIRMED and simultaneously as **weak
evidence**: a one-sided interval that passes on a 24×-wrong prediction has not
tested much. The finding is N1d's, not N1b's.

**Why context is free here.** The step is ~6.1 s at *both* 512 and 32768 tokens
— flat. Whatever dominates it does not scale with context, so KV's bytes vanish
inside it: streaming 1.77 GB per step costs **28 ms**. That is the sixth
mechanism-derived prediction falsified today, and it fell the same way the others
did — the byte model was right about direction and wrong about what dominates.

**Correction, made against my own first write-up.** I initially explained the
6.1 s as "94 layers × ~65 ms of `c_box`". That is retracted. `c_box` is a
**whole-box** constant — 53.5–114 ms across seven receipted boxes for the entire
per-token forward — not per-layer. I divided the measurement by the layer count,
got 65 ms, and presented the quotient as though the law had predicted it. That is
the exact failure mode this document set exists to catch, committed while writing
up a finding about that failure mode. The dominant term is **unidentified.**

### The working-set claim falls, and the short arm is why

N1a failed at **28.79 GB against a ≤16 GB target** — but the more serious number
is that the **`short` arm alone measured 18.62 GiB and 0.160 tok/s**, where the
stamped flagship records **15.2 GB and 4.3–4.4 tok/s at seq-512**. That is
**27× slower and 1.2× larger at the flagship's own operating point.**

**This does not show the stamped flagship number is wrong, and it is not reported
as showing that.** This harness ran `load_moe_4bit_streaming` at its defaults; the
stamped figure came from a tuned configuration (hot residency, E-pinning) that
was not reproduced here. What it does establish is narrower and sufficient:
**nothing measured on this box supports extending the flagship claim to 32K**,
and the discrepancy at the shared operating point has to be resolved before any
235B number is quoted with more confidence than it has earned.

**Pre-committed decision fires (N1a falsified).** The README heading keeps its
`(at ~5K context)` qualifier. The 32K claim does **not** go on the flagship line.
The transferable result — *when weights stream, 64× the context is free because
the step is fixed-cost bound* — is a finding about **streaming**, not a headline
about 235B, and it is filed that way.

**CLOSED by finding #29** (no new measurement needed). The flagship stages
*one layer's active experts*, double-buffered, on a 44.3 GB/s link; this harness
staged the whole stack synchronously on a 22.5 GB/s link. Both figures match the
additive law on their own mechanism to within 6%. The 27x was 16x of surplus
bytes times ~2x of link, not an error in either number.
