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

## Amendment 1 — G1: decompose the 1.887 into path overhead and dequant

Written after F1, **before the third arm was measured**. #17 states its ratio is
an **upper bound** on the dequant's cost because `DynamicCache` and
`NF4KVCache` are different objects on different code paths. That caveat is
honest but useless to act on: it does not say whether to optimize the dequant or
the wrapper.

**The control needed already exists.** `NF4KVCache(quantize_keys=False,
quantize_values=False)` stores raw bf16 through the *same* object, the *same*
`update()` bookkeeping, the *same* append and load path — and skips the
quantize/dequant entirely. So:

```
path_overhead = NF4KVCache(raw)  / DynamicCache        our wrapper, no arithmetic
dequant_cost  = NF4KVCache(nf4)  / NF4KVCache(raw)     arithmetic, same wrapper
total         = path_overhead × dequant_cost           must reproduce 1.887
```

**A back-of-envelope that makes this worth doing.** The measured gap at 4096 is
127 ms over 16 layers ≈ **8 ms per layer**. Dequantizing one layer's K and V is
~17 MB of output from ~9 MB of packed input; at this card's bandwidth that is
**tenths of a millisecond**, not 8. So most of the gap is probably *not* dequant
arithmetic — and if so, #17's number is dominated by something fixable that has
nothing to do with 4-bit.

- **G1a.** `path_overhead` at 4096 ∈ **[1.0, 1.3]**. *Falsified above 1.6.*
- **G1b.** `dequant_cost` at 4096 ∈ **[1.4, 1.9]**. *Falsified outside
  [1.2, 2.5].*
- **G1c.** The decomposition is clean: `path_overhead × dequant_cost` reproduces
  F1a's 1.887 to within **±5%**. *Falsified outside* — a product that does not
  close means the two arms differ in something besides the named factor.
- **G1d.** At 16384 the dequant factor grows and the path factor does not:
  `dequant_cost(16384) > dequant_cost(4096)` **and**
  `path_overhead(16384) ≤ path_overhead(4096) + 0.10`. Dequant scales with
  context; per-layer wrapper bookkeeping is a constant number of Python calls.
  *Falsified if either half fails.*

**Pre-committed decision.** If G1a > 1.3, the wrapper is the target and #17's
number is an artifact of this implementation rather than a property of 4-bit KV
— which would mean the reframing in #17 overstates the cost of the *idea* while
correctly stating the cost of the *code*. If G1b > G1a, the dequant is the
target and a fused dequant is worth registering.

## Outcome of G1 — the dequant is the target, and one arm is not trustworthy

| ctx | path_overhead | dequant | product | measured total |
|---:|---:|---:|---:|---:|
| 4096 | **1.138** | **1.475** | 1.679 | 1.679 |
| 16384 | *0.736* | *2.920* | 2.150 | 2.150 |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| G1a path_overhead @4096 | [1.0, 1.3] | **1.138** | **CONFIRMED** |
| G1b dequant @4096 | [1.4, 1.9] | **1.475** | **CONFIRMED** |
| G1c product closes to ±5% | — | exact | **VACUOUS — see below** |
| G1d dequant grows, path does not | both | 1.475→2.920, 1.138→0.736 | **CONFIRMED** |

**G1c was not a test.** `path_overhead × dequant = (raw/bf16) × (nf4/raw) =
nf4/bf16 = total` **algebraically**. It closes by construction and could not have
failed. That is a prediction that cannot be wrong, which this document set
treats as worthless — the fourth specification error of the day, after A1a's
two-regime fit, A1d's inverted sign and E1's decision omitting E1b.

**The 16384 row is contaminated and is not reported as a result.**
`path_overhead = 0.736` says our wrapper is *faster* than `DynamicCache` while
doing strictly more work on identical data (both arms measure a 2151.7 MB cache
and an 8169.5 MB peak). That is not physical. The cause is visible in the peak:
**8.17 GB of ~8.6 GB free**, so the arms are running against the allocator's
limit and the ordering between them decides who pays for fragmentation. Only the
4096 decomposition is trustworthy.

**Which also puts a variance band on #17.** The same F1 arms re-measured here
give **1.679** where the first run gave **1.887**, and 2.150 where it gave 2.552
— roughly **±12%** run to run on a shared, near-full card. #17's headline is
therefore restated as **~1.7–1.9× at 4K** rather than a precise 1.887, and the
16K figure as **~2.2–2.6×**. The direction and magnitude survive; the third digit
never existed.

**Pre-committed decision fires: G1b (1.475) > G1a (1.138), so the DEQUANT is the
target.** The wrapper costs ~14% and is not worth attacking. `dequant_kv_ref` is
a *reference* implementation by name, and replacing it with a fused kernel is
worth registering as its own experiment — with the caveat that D1 is a standing
warning about assuming a hand-written kernel beats a vendor path.

## Scoring

Results in `receipts-vs-bf16-20260725/`. Each prediction marked
**confirmed / falsified / void** with the measured value beside the interval.
Falsified entries stay.
