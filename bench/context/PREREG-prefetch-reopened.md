# PREREG — prefetch, reopened because the number it was closed on moved

**Tier: CONFIRMATORY. Status: STAMPED before the measurement.**
Code under test: gnf4 `kernel/nf4-kv-cache` @ `b0f5604`,
e4b `claude/e4b-gemma-inflight-d41f93` @ `5594538`. Both local, unpushed.

## Why this is being reopened, and why that needs justifying

Finding #16 closed prefetch after three registered attempts, on this reasoning:

> The transfer is 24.4 ms of a ~262 ms step — **9%** — and the machinery to hide
> it costs more than the 9% it chases. At 9% of a step, transfer is not worth
> machinery.

That was correct and it is now stale. #18 removed the dequant, and with it most
of the step the transfer was 9% *of*. Measured after the fused kernel landed,
the streamed arm at 16384 exposes **270.6 ms of a 591 ms step — 46%**.

**A closure conditioned on a ratio is void when the ratio moves.** Reopening on
that basis is legitimate; reopening because I would like a different answer is
not, which is why this is a registration and not a re-run.

## Track record, stated because it should discipline the intervals

Prefetch is **0 for 3** on the real path:

| attempt | result |
|---|---|
| B1 (synthetic harness) | confirmed — then invalidated by E1, which found the harness never called `update()` |
| E1 (real model) | falsified, −22.5% hidden; found two correctness defects |
| E2 (per-layer events) | falsified, worse than the whole-stream barrier |

Three mechanism arguments, three falsifications. The intervals below are set so
that a marginal result reads as failure, not success.

## Predictions

Baseline, post-#18: at 16384, resident **320.68**, streamed **591.28** ms/step.
E2 measured prefetch's machinery costing ~25 ms at 4096.

- **P1a.** Prefetched-streamed / streamed at 16384 ≤ **0.95**.
  *Falsified at ≥ 1.00* — no better than not prefetching, for the fourth time.
- **P1b.** Hidden fraction at 16384 ≥ **25%**. *Falsified below 0%*, i.e. if it
  again makes exposure worse.
- **P1c — gate.** Greedy ids identical to the non-prefetched streamed arm over
  32 tokens. E1 caught a wrong-and-faster run this way and an earlier arm passed
  the same check by luck, so it is a gate and not a formality.
- **P1d.** At 4096 — where E2 measured −38.9% — it does not regress further:
  hidden fraction ≥ **−38.9%**. *Falsified below.* A change that helps at long
  context by hurting more at short is not a win.

**Harness defect fixed first, and it is prerequisite.** The comparison prompt is
`torch.randint` with **no seed**, so token comparisons are not reproducible
across runs (#18 recorded this). P1c cannot be scored until the prompt is
seeded; the seed is fixed at 0 and recorded here.

## Pre-committed decision

If **P1a** fails, prefetch is **closed permanently** — four registered attempts,
across two harnesses and three mechanisms, is more than the idea has earned, and
no further reopening happens on a ratio argument. If P1a and P1c hold, it ships
as an opt-in documented path with the regime stated: it pays when transfer is a
large fraction of the step, which is long context with cheap per-layer compute,
and not otherwise.

## Outcome — it works at 16K, costs at 4K, and both matter

| prediction | predicted | measured | verdict |
|---|---|---|---|
| P1c **gate** ids identical | exact | True at both contexts | **CONFIRMED** |
| P1a prefetched/streamed @16384 | ≤ 0.95 | **0.865** | **CONFIRMED** |
| P1b hidden @16384 | ≥ 25% | **29.0%** | **CONFIRMED** |
| P1d no worse than −38.9% @4096 | ≥ −38.9% | **−53.9%** | **FALSIFIED** |

| ctx | resident | streamed | streamed+prefetch | exposed transfer | share of step |
|---:|---:|---:|---:|---:|---:|
| 4096 | 196.28 | 238.82 | **261.74** (1.096×) | 42.54 → 65.46 | **17.8%** |
| 16384 | 234.43 | 439.85 | **380.47** (0.865×) | 205.42 → 145.83 | **46.7%** |

**The reopening was justified and the mechanism is unchanged.** Nothing about
prefetch was rewritten between E2 and here — #18 removed the dequant, the
transfer's share of a step went from 9% to 47%, and the same code flipped from
−38.9% to **+29.0%**. #16's closure was a correct reading of a ratio that #18
then invalidated.

**P1d's falsification is the useful half.** Prefetch does not merely fail to help
at 4096, it costs **9.6%** and hides −53.9% — *worse* than when E2 measured it.
So this is not "prefetch works now"; it is "prefetch works above some transfer
share and hurts below it". P1d existed to catch exactly a change that buys long
context by taxing short, and it did.

**The crossover is bracketed but not located.** It pays at a 46.7% transfer
share and loses at 17.8%. Where between those it turns is unmeasured, and no
number is offered for it.

**Pre-committed decision fires as written.** P1a and P1c hold, so prefetch
**ships as an opt-in documented path with the regime stated** — and P1d supplies
the regime rather than contradicting the decision. It stays **off by default**:
the default context for most callers is nearer 4K than 16K, where it is a 10%
tax.
