# PREREG — what does the NF4 KV cache actually cost, against not quantizing at all?

**Tier: CONFIRMATORY. Status: STAMPED before the harness was written.**
Code under test: e4b `claude/e4b-gemma-inflight-d41f93` @ `9a4fec9`,
gnf4 `kernel/nf4-kv-cache` @ `b9fdee2`. Both local, unpushed.

## Why this exists

**Every latency arm in `docs/context-budgets.md` compares NF4-cache
configurations against each other.** Resident vs streamed, prefetched vs not,
split vs binary, fused kernel vs dequant-then-SDPA — sixteen findings, and the
control is always another NF4 configuration. The one comparison never run is
against **not quantizing the cache at all**.

D1 made that gap impossible to ignore: at T=32768 / GQA 16:1 the shipped path
(dequantize a layer, hand it to stock attention) measures **10.750 ms** against a
bf16 cache's **0.324 ms**. That is a microbenchmark of one layer, not a decode
step, and the honest end-to-end number is unknown. This measures it.

The stakes are the module's framing. #10 characterizes NF4 KV as "3.56× memory
for ~2.1% perplexity", which reads as nearly free. If the latency cost is large,
that sentence is materially incomplete everywhere it appears.

## Fixture

OLMoE-1B-7B, **4-bit weights, resident** — same model, loader and shape as E1/E2
so the numbers compose with them. Greedy decode, 32 new tokens, contexts 4096
and 16384.

Note the geometry deliberately: OLMoE is **GQA 1:1**, so none of D1's
`enable_gqa` effect is in play here. Any difference measured is the **dequant**,
which is what this module actually ships, not the kernel it does not use.

**Arms.**

- `DynamicCache` — transformers' own bf16 cache. What a user runs by default.
- `NF4KVCache` resident — the shipped path.
- `NF4KVCache` host-resident — the streamed tier.

## Predictions

Grounded in E2's measured 261.68 ms/step for the NF4-resident arm at 4096, and
in the dequant being pure per-step overhead that `DynamicCache` does not pay.

- **F1a.** NF4-resident / DynamicCache step time at ctx 4096 ∈ **[1.2, 2.0]**.
  *Falsified outside [1.0, 3.0].* Below 1.0 would mean quantizing makes decode
  faster, which would indicate a broken measurement rather than a discovery.
- **F1b.** The ratio does not shrink with context: ratio(16384) ≥ ratio(4096).
  Both attention and dequant scale with context while the MLP does not, so the
  ratio should climb toward an asymptote. *Falsified if it falls by more than
  0.05.*
- **F1c.** VRAM is the other half of the trade and must be measured, not
  assumed: DynamicCache peak minus NF4-resident peak at 4096 ∈
  **[300, 450] MB** (bf16 KV is 524 MB there, NF4 147 MB).
  *Falsified outside.*
- **F1d.** NF4-streamed / DynamicCache at ctx 4096 ∈ **[1.4, 2.6]**.
  *Falsified outside [1.0, 3.5].*

**Reported, not predicted.** Greedy token ids will diverge between bf16 and NF4 —
the cache is lossy and #10 already measured that at ~2.1% perplexity. The
divergence point is recorded so the fidelity claim and the latency claim sit
next to each other rather than in separate documents.

## Pre-committed decisions

- **If F1a > 1.5**, the module is reframed wherever it is described: from "3.56×
  memory for ~2.1% perplexity" to that **plus an explicit latency cost**, with
  the measured number carried in `kv_cache.py`'s docstring, finding #10, and any
  surface that quotes the 3.56×. A capacity feature that is quietly also a
  slowdown is precisely the silent-wrongness this document set exists to remove.
- **If F1a < 1.2**, the current framing stands and this is recorded as a
  negative — the D1 microbenchmark did not generalize to a decode step.
- **If F1a < 1.0**, the measurement is presumed broken and is debugged before
  anything is claimed either way.

## Known confounds, stated in advance

1. **One model, one geometry, one device**, and OLMoE is GQA 1:1 — the least
   favourable case for the NF4 kernels and a *neutral* one for the dequant path
   under test. A high-GQA model would change D1's numbers, not this one's.
2. `DynamicCache` and `NF4KVCache` are different objects on different code
   paths; part of any gap is Python overhead rather than dequant arithmetic.
   Not separated here, and the ratio is therefore an upper bound on the
   dequant's own cost.
3. Weights are resident and 4-bit, so nothing else contends for the link — the
   same choice E1 made, for the same reason.

## Outcome — all four confirmed, and the reframing decision fires

OLMoE-1B-7B, 4-bit weights resident, greedy, 32 new tokens, median of 3.

| ctx | cache | ms/step | vs bf16 | peak | KV bytes |
|---:|---|---:|---:|---:|---:|
| 4096 | bf16 `DynamicCache` | 143.58 | 1.00× | 5406.4 MB | 541.1 MB |
| 4096 | NF4 resident | 270.96 | **1.89×** | 5039.6 MB | 152.2 MB |
| 4096 | NF4 host-resident | 312.17 | **2.17×** | 4900.1 MB | 0 on device |
| 16384 | bf16 `DynamicCache` | 237.27 | 1.00× | 8169.5 MB | 2151.7 MB |
| 16384 | NF4 resident | 605.58 | **2.55×** | 6697.5 MB | 605.2 MB |
| 16384 | NF4 host-resident | 741.88 | **3.13×** | 6121.5 MB | 0 on device |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| F1a nf4/bf16 @4096 | [1.2, 2.0] | **1.887** | **CONFIRMED** |
| F1b ratio must not fall | ≥ 1.887 | **2.552** @16384 | **CONFIRMED** |
| F1c peak saved @4096 | [300, 450] MB | **+366.8 MB** | **CONFIRMED** |
| F1d nf4-host/bf16 @4096 | [1.4, 2.6] | **2.174** | **CONFIRMED** |

Four for four is not something this document set has produced before, and the
reason is that these predictions were grounded in an existing measurement (E2's
261.68 ms/step) rather than in a mechanism I found plausible. Every falsified
prediction across the last three preregs was a mechanism argument.

**Greedy ids diverge at position 1 of 33** — the very first generated token can
differ. That is expected of a lossy cache and consistent with #10's ~2.1%
perplexity, but it belongs beside the latency number rather than in a separate
finding, because together they are the actual trade.

**Pre-committed decision fires (F1a = 1.887 > 1.5).** The module is reframed
wherever it is described. "3.56× memory for ~2.1% perplexity" reads as nearly
free and is materially incomplete: the same dial costs **1.9× decode at 4K,
rising to 2.6× at 16K**, and the streamed tier **2.2× rising to 3.1×**. The
number now travels with the claim.

**The cost grows with context, which is the worst direction.** The dial exists
*for* long context, and its price rises exactly there — 1.89× at 4K, 2.55× at
16K, still climbing. Both attention and dequant scale with context while the MLP
does not, so the ratio is heading for an asymptote it has not reached at 16K.

**Confound honoured, not quietly dropped:** part of the gap is Python and
code-path overhead rather than dequant arithmetic, since `DynamicCache` and
`NF4KVCache` are different objects. The ratio is therefore an **upper bound** on
the dequant's own cost, and the finding says so.

## Scoring

Results in `receipts-vs-bf16-20260725/`. Each prediction marked
**confirmed / falsified / void** with the measured value beside the interval.
Falsified entries stay.
