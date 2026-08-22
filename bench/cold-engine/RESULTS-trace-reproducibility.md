# The traces do not reproduce; the conclusions do

Receipts: [`repro-2026-08-22/`](repro-2026-08-22/) — 12 fresh traces, all
carrying `env` provenance, plus
[`policy-repro.json`](repro-2026-08-22/policy-repro.json). Preregistration:
[`PREREG-trace-reproducibility.md`](PREREG-trace-reproducibility.md), merged
before capture. RTX 5090, **$0.93**, box destroyed.

| test | outcome |
|---|---|
| **control** — gpt-oss re-captures identically | **PASS, 100.00% on all four prompts** |
| **T1** — R10's refutation | **REPRODUCED** |
| **T2** — one-step crossover (registered as a control) | **HOLDS, 12 of 12** |
| **T3** — per-model LFU ÷ LRU | **REPRODUCED** |
| **T4** — R4's refutation | **REPRODUCED** |
| Granite | **NOT RE-CAPTURABLE** |

## The traces really do not reproduce

Same model id, same prompt, same greedy decode, transformers 5.15.1:

| model | prose | code | math | dialogue |
|---|---|---|---|---|
| OLMoE | 18.07% | 79.65% | 2.65% | 88.28% |
| Qwen1.5-MoE | 90.87% | 7.58% | 3.04% | 18.68% |
| **gpt-oss (control)** | **100%** | **100%** | **100%** | **100%** |

Agreement swings from 2.65% to 90.87% because decoding is greedy: a single
flipped token early sends the trajectory somewhere else and everything after
it differs, while a flip at step 400 leaves most of the trace intact. gpt-oss
is identical because it was first captured under this same transformers.

The control passing at 100% is what licenses reading the rest: the re-capture
procedure is exact, so the movement above is real drift and not a botched run.

## The conclusions survive it

**T3, the consequential one, first.** Median LFU ÷ LRU, same statistic on both
sides, tolerance 0.05:

| model | published | re-captured | delta | |
|---|---|---|---|---|
| OLMoE | 0.775 | 0.804 | **0.029** | reproduced |
| Qwen1.5-MoE | 0.893 | 0.929 | **0.036** | reproduced |
| gpt-oss | 0.992 | 0.992 | 0.000 | control |

The ordering that the whole frequency/recency thread rests on — OLMoE well
below Qwen, Qwen below gpt-oss — is intact. The five explanations eliminated
against that split were not fitted to a vanished environment.

**T1.** R10 refuted in **115 of 120** testable cells against 113 in the
published run over the same cells, with the exceptions still confined to
gpt-oss at the largest capacity. Qwen's two published exceptions disappeared;
nothing new appeared.

**T2**, registered as a control rather than a result: the sign pattern holds
on 12 of 12 fresh traces. It was expected to —
[`structural_check.py`](routing-trace/structural_check.py) shows it cannot
move — and it is reported only because a break would have meant a broken
capture.

**T4.** Frequency wins **28 of 33** cells, 84.8%, against the registered 80%.
Weaker than the published run, where frequency won everywhere: recency now
takes five cells, four of them OLMoE. Reproduced as registered, and the
softening is recorded rather than rounded away.

## Why a 2.65%-agreement trace still gives the same answer

The conclusions are statistics over the routing *distribution* — how often
experts recur, how concentrated they are, how much a step overlaps the last —
and those are properties of the model's router, not of which particular tokens
it happened to emit. A different trajectory samples the same distribution.

That is a real robustness property and it was not guaranteed. It also bounds
what the drift threatens: any result that depended on a *specific* sequence
rather than its statistics would not have survived, and none of the four does.

## Granite cannot be re-captured at all

The repository records no `ibm-granite/*` identifier anywhere. The trace
carries `model: /root/models/granite` and geometry (32 layers, 40 experts,
top-8), and prose in the derivation documents names "Granite-3.0-3B-A800M" —
which does not disambiguate `-instruct` from `-base`, and those route
differently.

Guessing was rejected: a wrong pick produces a trace that never corresponded
to the original, and comparing it would manufacture a "moved" result. Granite
is therefore untested here, and its published 0.745 stands on the original
capture alone.

This is the sharpest instance of the problem the `env` field now fixes — but
`env` records the *environment*, not the *identity*. A trace should also carry
the repo id it was captured from, which is a one-line follow-up and is the
next thing to do.

## What this changes

Nothing is retracted. R4, R10, the crossover threshold and the policy-headroom
ratios all survive on traces captured today, and the one that could not be
tested is flagged rather than assumed. The committed traces remain the record
of what was measured; they are now also, finally, labelled with what measured
them.
