# PREREG — can layer L+1's routing be guessed early enough to prefetch it?

**Tier: EXPLORATORY→CONFIRMATORY. Status: STAMPED before the harness was written.**
Code: gnf4 @ `01b6a48`, e4b @ `687e7d2`. Both local, unpushed.

## The gap, and why it looked closed

Routed staging is **5.95–6.97×** and bit-identical at 94 layers (#22, #31), but it
is **synchronous**: it cannot prefetch, because layer L+1's routing is produced by
a router reading layer L's *output*. Prefetch (worth 1.11×) and routed staging
(worth ~6×) are therefore mutually exclusive, and phase3's ~2× advantage over
routed staging is exactly that missing overlap (#29).

I had written that exclusion down as structural and closed. **It is only
structural if the routing must be known exactly.** If it can be *predicted* well
enough, the experts can be staged speculatively and the rare miss corrected
synchronously — which converts the exclusion into an accuracy question.

## The two horizons

For layer L+1 the router input is `norm2(x_{L+1} + attn_{L+1})` where
`x_{L+1} = h_L + moe_out_L`. Predicting it early means dropping terms:

| horizon | predict from | terms dropped | prefetch window |
|---|---|---|---|
| **H1** | `x_{L+1}` (after L's MoE) | L+1's attention | L+1's attention only |
| **H2** | `h_L` (after L's attention, **before L's MoE**) | L's MoE **and** L+1's attention | **L's entire expert compute** + L+1's attention |
| **T** | previous token's set for L+1 | everything | unbounded |

**H2 is the one worth having.** The expert compute is 2.174 ms/layer grouped
(#23) and dominates attention at decode, so H2's window is essentially the whole
thing to be hidden. H1's window is small enough that it may not be worth the
machinery. T is the naive baseline and is included to be beaten.

## Predictions

Hit rate = |predicted ∩ true| / top_k, per layer per token, over the true set.

- **E1a — H2 beats chance by a wide margin.** H2 hit rate ≥ **0.50**. Random
  top-8-of-128 gives 0.0625. *Falsified below 0.20.* The residual stream is
  dominated by the accumulated activation; one MoE delta and one attention output
  should not re-route most experts.
- **E1b — H1 ≥ H2**, since it drops strictly fewer terms. *Falsified if
  H2 exceeds H1 by more than 0.05* — that would mean the harness is measuring
  something other than what it claims.
- **E1c — both beat the temporal baseline.** H2 > T. *Falsified otherwise*, in
  which case same-token structure is worth nothing over "what did this layer use
  last token" and the simpler scheme wins.
- **E1d — THE DESIGN GATE.** Staging the top-**16** predicted experts (of 128)
  under H2 covers ≥ **90%** of the true top-8, across layers. That is **8× fewer
  bytes than bulk** while overlapping, versus routed staging's 16× without
  overlap — so it is only worth building if the coverage is high enough that
  misses are rare. *Falsified below 75%.*

**Reported, not predicted:** hit rate as a function of layer depth. Early layers
may route differently from late ones, and a scheme that only works in half the
stack is worth knowing about before it is built.

## Pre-committed decisions

- **E1d confirmed** → speculative routed prefetch is worth building: stage the
  top-K predicted experts on the side stream during layer L's compute, correct
  misses synchronously, and the correction path is what must then be proven
  bit-identical.
- **E1d falsified but E1a confirmed** → the signal exists but not at usable
  precision. Record and stop; do not build a scheme whose miss path runs on most
  tokens.
- **E1a falsified** → routing is not predictable from the residual at this
  horizon, the exclusion stands as genuinely structural, and phase3's ~2× is
  simply unavailable to a correctness-preserving path.

## Confounds

1. Measurement only — nothing is staged speculatively here. A hit rate is an
   upper bound on what a real implementation achieves, since a real one also pays
   for the prediction itself (one extra router matmul per layer, on resident
   weights, cheap but not free).
2. Qwen3-30B-A3B (48 layers, 128 experts, top_k 8) — same expert geometry family
   as the 235B and loads in ~5 min. Depth differs; #31 found the grouped kernel's
   divergence did *not* scale with depth, so depth-extrapolation here is
   unwarranted either way.
3. Greedy decode on one natural prompt. Routing locality may differ under
   sampling or across domains.

## Amendment 1 — two wider horizons (registered before measuring them)

Written after the H1/H2/T harness launched, **before any result was read.**

H2's window is layer L's expert compute. Two signals give a far larger one:

| horizon | predict layer i from | window |
|---|---|---|
| **H3** | the **current token's embedding** (layer-0 input) | every layer before i |
| **H4** | the **previous token's final hidden state** | unbounded — starts before the token does |

H4 is the interesting one operationally: the entire token's expert working set
could be staged before the token begins, overlapping everything rather than one
layer. Both are strictly weaker signals than H1/H2 — more of the network sits
between the predictor input and the router — so this trades accuracy for window
in the opposite direction from H1 vs H2.

- **E2a — the trade is real.** H3 < H2 and H4 < H2. *Falsified if either equals
  or beats H2*, which would mean the intervening layers carry no routing
  information and the whole H1/H2 framing is wrong.
- **E2b — but still far above chance.** H3 ≥ **0.25** (chance is 0.0625).
  *Falsified below 0.12.*
- **E2c — H3 decays with depth.** Mean hit rate over the last quartile of layers
  is below the first quartile by ≥ **0.05**. The embedding is further from a deep
  layer's router input than a shallow one's. *Falsified if it does not decline.*
- **E2d — the whole-token design gate.** Under **H4**, staging the top-**32** of
  128 predicted experts per layer covers ≥ **0.85** of the true top-8. That is
  4× fewer bytes than bulk with an unbounded window — a worse byte ratio than
  routed staging's 16×, bought back by full overlap. *Falsified below 0.70*, in
  which case the whole-token scheme is not worth its miss path.

**Pre-committed:** if **E2d** confirms and **E1d** does not, the right design is
whole-token speculative staging rather than per-layer; if both confirm, per-layer
wins on bytes and the choice is made on measured overlap, not on this. If neither
confirms, prediction is not accurate enough at any horizon and the
prefetch/routed exclusion stands.

## Outcome (E1) — the MoE output determines routing; attention barely does

Qwen3-30B-A3B, 44 MoE layers, 128 experts, top_k 8, 24 greedy decode steps.
Chance = 0.0625.

| predictor | drops | hit rate | top-16-of-128 coverage |
|---|---|---:|---:|
| **H1** — from `x_{L+1}` | L+1's attention | **0.9089** | **0.9930** |
| **H2** — from `h_L` | L's MoE **and** L+1's attention | 0.2439 | 0.3824 |
| **T** — previous token's set | everything | 0.4513 | — |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| E1a H2 ≥ 0.50 | ≥0.50 | 0.2439 | **outside interval** |
| E1b H1 ≥ H2 − 0.05 | — | 0.9089 vs 0.2439 | **CONFIRMED** |
| E1c H2 > T | — | 0.2439 vs 0.4513 | **FALSIFIED** |
| E1d top-16 under **H2** ≥ 0.90 | ≥0.90 | 0.3824 | **FALSIFIED** |

**The single term between H1 and H2 is layer L's MoE output, and it carries
almost all of the routing signal.** Dropping L+1's *attention* costs 9 points of
hit rate; additionally dropping L's *MoE* costs 66 more. Routing is set by what
the experts wrote to the residual, not by what attention did.

**E1c is the one I got backwards.** I predicted same-token structure would beat
"what did this layer use last token", and the temporal baseline (0.4513) beats H2
(0.2439) nearly 2:1. Routing decisions are temporally stable even where the
router is highly sensitive to the residual — reusing the *decision* beats
recomputing it on a stale input.

**H2 is dead as a design.** Its window was the attractive one — layer L's whole
expert compute — and at 0.38 coverage its miss path would run on most tokens.
The pre-committed decision fires: record and stop; do not build it.

**H1 is alive and was not what this prereg was built to test.** 99.3% coverage at
top-16 is comfortably usable, but its window is only L+1's attention, which at
decode is far smaller than the 85 MB it would need to hide. Whether that is worth
anything is a *different* question — one of lookahead distance, not of accuracy —
and it is registered separately below rather than claimed here.

**Depth:** H2 by quartile 0.305 / 0.195 / 0.191 / 0.268 — weakest in the middle,
not monotonic. Recorded; not explained.

## Amendment 2 — lookahead distance (registered before measuring)

H1 predicts one layer ahead and buys only L+1's attention. The question its
result raises: **how far ahead does that accuracy survive?** Predicting layer
`L+d` from `x_{L+1}` for d ≥ 2 buys a window of `(d−1)` full MoE layers plus
attention — real overlap, if the accuracy holds.

- **E3a — decay is real but gradual.** top-32-of-128 coverage at **d=2** ≥ **0.85**.
  *Falsified below 0.65.*
- **E3b — d=1 dominates d=2 dominates d=3**, strictly, by ≥0.02 each step.
  *Falsified if the ordering breaks* — that would indicate the predictor is not
  actually using distance-sensitive information.
- **E3c — THE DESIGN GATE.** At the largest d whose coverage ≥0.85, the byte
  ratio `K/128` must still beat **1/4** (i.e. K ≤ 32). Bulk is 1/1 and
  synchronous routed staging is 1/16; a speculative scheme must land between and
  earn the gap back in overlap. *Falsified if no d ≥ 2 satisfies both.*
