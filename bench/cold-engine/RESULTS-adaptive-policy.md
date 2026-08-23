# ARC does not rescue this: both predictions refuted

Receipt: [`adaptive-2026-08-22/adaptive.json`](adaptive-2026-08-22/adaptive.json).
Harness: [`adaptive_policy.py`](routing-trace/adaptive_policy.py).
Preregistration: [`PREREG-adaptive-policy.md`](PREREG-adaptive-policy.md),
merged before measurement. 48 cells, no box.

| prediction | outcome |
|---|---|
| **A1** — ARC within 2% of the better fixed policy, per model | **REFUTED** |
| **A2** — ARC closes ≥ 15% of the gap on gpt-oss | **REFUTED** |

Five single-variable explanations for when frequency beats recency are dead,
so the registered question was whether a policy could stop us needing one.
It cannot — at least not this policy.

## A1 — ARC loses badly to LFU wherever LFU works

Median over 12 cells per model:

| model | ARC ÷ min(LRU, LFU) | ARC ÷ LRU | ARC ÷ LFU |
|---|---|---|---|
| OLMoE | **1.258** | 0.970 | 1.258 |
| Granite | **1.318** | 0.967 | 1.318 |
| Qwen1.5-MoE | **1.111** | 0.983 | 1.111 |
| gpt-oss | 1.004 | 0.969 | 1.004 |

Registered at ≤ 1.02; three of four models are 11–32% over. The shape is the
finding: **ARC ÷ LRU is 0.967–0.983 on every model** — a uniform ~3% better
than LRU — and ARC ÷ LFU tracks whatever LFU was doing. ARC behaves like a
slightly improved LRU everywhere and never crosses into the frequency regime,
which is the one thing it was chosen to do.

## A2 — and it does not rescue gpt-oss either

Fraction of the LRU-to-Belady gap closed:

| model | ARC | LFU |
|---|---|---|
| OLMoE | 6% | **49%** |
| Granite | 6% | **49%** |
| Qwen1.5-MoE | 5% | 30% |
| **gpt-oss** | **6%** | 2% |

Registered at ≥ 15% on gpt-oss; ARC closes 6%. It is three times LFU's 2%
there, and still nowhere near enough to matter — the preregistration fixed 15%
precisely so that a small improvement could not be presented as a rescue.

The flatness across models is the striking part: **ARC closes 5–6% of the gap
on all four**, including the two where LFU closes half. Whatever ARC's ghost
lists are adapting to, it is not the thing that separates these models.

## Both preconditions were met, so the refutation is worth something

The preregistration made two checks preconditions on reporting, and both are
why this negative result can be believed rather than blamed on the harness.

**The implementation is validated**, not assumed:

| check | result |
|---|---|
| at capacity ≥ key space, ARC = LRU = LFU = Belady | pass at U = 8, 40, 137 |
| Belady ≤ ARC ≤ all-miss | pass, 40 random workloads |
| resident + ghost ≤ 2c (ARC's own invariant) | pass, 40 random workloads |

**Both predictions were falsifiable**, checked before scoring: across twelve
synthetic conditions A1 would have been refuted in 7 and A2 in 10. Neither is
arithmetic, so neither refutation is an artifact — and equally, a confirmation
would have meant something.

## What this leaves

Six things have now been tried against the frequency/recency split: five
explanations, refuted, and one policy meant to make the explanation
unnecessary, also refuted. LFU remains the best implementable policy found,
closing about half the gap on two models, a third on one, and nothing on the
fourth.

The practical conclusion is unchanged and now better supported: pick the
victim rule **per model, on a measurement**, as
[`RESULTS-policy-headroom.md`](RESULTS-policy-headroom.md) already concluded.
Adaptivity was the obvious way to avoid having to, and it does not work here.

The ~1.9× gap to optimal stays open. What is now known is that it will not be
closed by swapping in a general-purpose replacement policy — ARC is the
standard answer to exactly this problem and it recovers 6%. Anything that
closes it will have to use something about MoE routing that general cache
theory does not have, and this program has spent five experiments failing to
find what that something is.
