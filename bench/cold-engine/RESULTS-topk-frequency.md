# Top-k does not explain when frequency beats recency — both predictions refuted

Receipts: [`topk-frequency-2026-08-22/`](topk-frequency-2026-08-22/) — 32
captured traces, [`policy-topk.json`](topk-frequency-2026-08-22/policy-topk.json)
(96 cells). Preregistration:
[`PREREG-topk-frequency.md`](PREREG-topk-frequency.md), merged before capture.
RTX 5090, **$2.14**.

| prediction | outcome |
|---|---|
| **K1** — `LFU ÷ LRU` is monotone in k within each model | **REFUTED** |
| **K2** — at matched k the two models differ by < 0.20 | **REFUTED** |
| the registered native-k gate | **FAILED** — for a reason that turned out not to be about this experiment at all |

The hypothesis was that top-k explains why a frequency-aware victim rule
closes half the LRU-to-optimal gap on OLMoE and Granite and nothing on
gpt-oss. It was registered precisely because a binary variable splits four
points 2–2 about a third of the time by chance. **It was the coincidence.**

## K1 — refuted, but only on one of the two models

`LFU ÷ LRU`, median over 4 prompts × 3 capacities, k varied while the model,
its weights and its decode trajectory are held fixed:

| k | OLMoE | gpt-oss |
|---|---|---|
| 2 | 0.841 | 1.012 |
| 4 | 0.814 | 0.992 |
| 8 | 0.804 | 0.919 |
| 16 | **0.729** | 0.984 |
| range | 0.112 | 0.092 |
| monotone ↓ | **yes** | **no** (inverts at 16) |

Registered as "refuted by any inversion in either model", so **K1 is
refuted**. What survives is narrower and real: **within OLMoE, k does move the
ratio monotonically** across a factor of 8 in k. Within gpt-oss it does not
move it anywhere useful — LFU stays between 0.92 and 1.01 at every k, so on
that model frequency simply does not help regardless of how many experts a
step samples.

## K2 — refuted, and this is the substantive result

| k | OLMoE | gpt-oss | gap |
|---|---|---|---|
| 2 | 0.841 | 1.012 | 0.171 |
| 4 | 0.814 | 0.992 | 0.178 |
| 8 | 0.804 | 0.919 | 0.116 |
| 16 | 0.729 | 0.984 | **0.256** |

Registered as confirmed only if the gap is under 0.20 at **every** k. It is
0.256 at k = 16, so **K2 is refuted** — and the shape matters more than the
verdict. **The gap never closes.** Equalising the variable that perfectly
separated the four models leaves a 0.12–0.26 difference between them at every
matched k. Top-k is not what distinguishes these models.

That is the useful outcome: it removes the last cheap explanation. Skew,
working-set pressure and reuse distance were refuted in
[`RESULTS-frequency-signal.md`](RESULTS-frequency-signal.md); `k/E` is ruled
out by OLMoE and gpt-oss sharing 0.125 at opposite ends; and top-k is ruled
out here, within-model, with the workload held fixed.

## The registered gate failed, and what it actually found

The prereg required a native-k derived capture to reproduce the committed
OLMoE trace **id for id**, and said any mismatch means the derivation is
wrong and nothing is scored. It matched **18.07%**.

The derivation is not wrong. Three comparisons in the same environment
separate the two possibilities:

| comparison | agreement |
|---|---|
| **module-emitted vs derived, this environment** | **8192 / 8192 = 100.00%** |
| committed vs module-emitted, this environment | 18.07% |
| committed vs derived | 18.07% |

Deriving routing from the router's own weights reproduces the module's own
indices **exactly, on every layer of every step**. The *module path itself*
disagrees with the committed trace by the same 18%. So the mismatch is
environmental — **the committed OLMoE traces do not reproduce under
transformers 5.15.1** — and has nothing to do with the override.

**That is a finding about the repository, not about this experiment**, and it
is the more consequential half of the day. `olmoe_*.jsonl` is the trace behind
R4, R10, the crossover threshold and the policy-headroom numbers. Those
results are not thereby wrong — they are internally consistent, and every
comparison in them was made against traces captured together — but the traces
are no longer regenerable from the model id and prompt alone, and nothing in
the repo records the transformers version they were captured under.

The k-sweep is unaffected: all 32 traces were captured in one environment, and
the comparison is between them, never against the committed set. The
registered gate was testing derivation correctness by proxy; the direct check
of the same property passed at 100%, and the results below are reported with
that substitution stated rather than performed silently.

## What this leaves

Four explanations for the frequency/recency split are now eliminated —
popularity skew, working-set pressure, reuse distance, and top-k — three of
them registered before measurement. The split itself is robust: gpt-oss sits
at 0.92–1.01 across a factor of eight in k while OLMoE reaches 0.729.

The remaining candidates are the ones no experiment here has touched: expert
count, layer count, and router temperature. None is testable within a model
the way k was, so the next honest step is more models rather than another
manipulation — and after five refutations the prior on any single-variable
explanation should be low.

## Method

`--top-k` changes what the recording hook writes, never what the model
computes, so all four k values share one decode trajectory — identical hidden
states, identical tokens — and differ only in how deep the router's ranking is
read. That property was pointed out by Bugbot on #191 before capture and is
what makes the within-model comparison clean; re-running the model at each k
would have given every k a different workload.

Metadata records `top_k_native` and `top_k_overridden` on every trace, so a
counterfactual readout can never be mistaken for a native capture.
