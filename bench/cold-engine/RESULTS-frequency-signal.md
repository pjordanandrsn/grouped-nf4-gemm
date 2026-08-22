# Why frequency fails on gpt-oss: not skew, not pressure, not reuse distance

**Registered outcome: REFUTED**, and the two obvious follow-ups fail with it.
The three standard cache-theory quantities do not predict where a
frequency-aware victim rule helps.

Registered in
[`wall-hard-vs-soft/PREREG-frequency-signal.md`](wall-hard-vs-soft/PREREG-frequency-signal.md)
before measuring. Receipt: [`routing-trace/frequency-signal.json`](routing-trace/frequency-signal.json).
Harness: [`routing-trace/frequency_signal.py`](routing-trace/frequency_signal.py).
48 cells, the same ones [`RESULTS-policy-headroom.md`](RESULTS-policy-headroom.md)
measured. No box.

## The question

That document measured a frequency-aware rule closing **49%** of the
LRU-to-optimal gap on OLMoE and Granite, **30%** on Qwen, and **2%** on
gpt-oss — the one model every wall measurement in this program was made on. It
records the fact and not the cause, and the cause decides whether some other
implementable policy could work there.

## Registered hypothesis: popularity skew. REFUTED.

LFU keeps rows that recur because they are *popular*, so uniform routing should
be where it fails. Registered at Spearman |rho| ≥ 0.8; refuted below 0.5.

| model | entropy | gini | distinct | LFU ÷ LRU |
|---|---|---|---|---|
| granite | **0.9539** (most uniform) | 0.4395 | 1239 | **0.7454** (helps most) |
| qwen | 0.9349 | 0.4869 | 1439 | 0.8935 |
| gptoss | 0.9347 | 0.5005 | 635 | **0.9922** (helps least) |
| olmoe | 0.9203 (least uniform) | 0.5419 | 989 | 0.7745 |

**Spearman +0.253** (entropy), **−0.315** (gini) — far under the registered
0.8, and the ordering is *backwards*: the most uniform model is the one
frequency helps most. The registered rule has **two** clauses, and this fails
both: gpt-oss is not the most uniform model (granite is, 0.9539 vs 0.9347),
which is an independent REFUTED path regardless of rho. gpt-oss and qwen sit at effectively identical entropy
(0.9347 vs 0.9349) with gains of 0.992 and 0.894.

## Two follow-ups, both exploratory, both negative

Stated as exploratory: the registered test had already failed, so these are
hypothesis-fishing and would need their own confirmation.

**Working-set pressure** (distinct keys ÷ capacity) — **Spearman +0.277**,
wrong sign and weak. olmoe (5.15) and gptoss (5.24) are at nearly the same
pressure with gains of 0.775 and 0.992.

**Reuse distance** (LRU stack distance — the quantity that decides what LRU can
exploit at all) — **Spearman −0.214** median, **−0.204** mean, **+0.055**
normalised by capacity.

## The one thing that does line up, and why it is not a finding

Ordering by median reuse distance, three of four models are monotone:

| model | median RD | RD ÷ cap | LFU ÷ LRU |
|---|---|---|---|
| gptoss | 139.5 | 1.05 | 0.9922 |
| olmoe | 211.2 | 1.19 | 0.7745 |
| granite | 362.8 | 1.02 | 0.7454 |
| qwen | 435.5 | **3.28** | 0.8935 |

Shorter reuse distance → less benefit from frequency, for gptoss/olmoe/granite.
Qwen breaks it, and qwen is also the outlier on `RD ÷ cap` (3.28 against
~1.0–1.2): its median reference is *further away than the cache is large*, so
most references miss under any policy. That is a different regime.

**This is n=3 after excluding the fourth for a property noticed afterwards.**
It is the shape of a real effect and it is also exactly what hypothesis-fishing
produces. Recorded as a lead, not a result.

## What this is worth

Negative, and useful to the thread it serves: the ~1.9× gap to optimal is
mostly still open, and **the three quantities a cache person would reach for
first do not tell you where policy work will pay**. A frequency-aware rule
still has to be gated on a measurement against the target model — which is what
`RESULTS-policy-headroom.md` already concluded, now with the obvious
explanations ruled out rather than untried.

## The harness enforces the registered rule, after not doing so

The first version checked only `|rho|`, while the preregistration also requires
gpt-oss to be the most uniform model for CONFIRMED and treats it not being so
as its own REFUTED path. A future run with a strong rho and some other model
most uniform would have printed CONFIRMED where the registered rule says
REFUTED (Bugbot, gnf4#189). It also skipped missing traces silently.

That second one was not hypothetical. Run without the gpt-oss directory, the
correlation over the remaining 36 cells is **−0.629** — which the old code
would have reported as **PARTIAL**, a different verdict, on a hypothesis that
is *about gpt-oss*. Both now refuse rather than report.

## What this does not show

Four models, four prompts, three capacities — 48 cells, but only **four points**
on the axis that matters, since skew and reuse distance are near-constant within
a model. A correlation across 48 cells that vary mostly *within* four clusters
is weaker than its n suggests, and that is the honest reason the follow-ups are
labelled exploratory rather than merely negative.

Skew is computed from the routed `(layer, expert)` stream the cache observes,
not from gate logits.
