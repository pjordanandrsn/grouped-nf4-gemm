# Gate E: nearly everything pays, no selector is needed, and the copies are 88% free

Receipts: [`elastic-2026-08-22-gate.json`](elastic-2026-08-22-gate.json)
(offline, 16 rank traces, 1.1 M invocations) and
[`elastic-2026-08-22-e3.json`](elastic-2026-08-22-e3.json) (one Genoa 9J14 +
RTX 5090 box, destroyed). Harnesses:
[`routing-trace/elastic_gate.py`](routing-trace/elastic_gate.py),
[`elastic_e3.py`](elastic_e3.py). Preregistration:
[`PREREG-elastic-promotion.md`](PREREG-elastic-promotion.md), merged — with
Bugbot's off-by-one correction — before anything was scored.

| gate | outcome |
|---|---|
| **E1** — ≥ 25% of invocations pay back a promotion (≥ 2 more uses in 32 steps) | **CONFIRMED — it is 69–96%** |
| **E2** — rank or frequency identifies the payers at ≥ 1.25× base | **REFUTED as registered; uninformative on 3 of 4, and that is the finding** |
| **E3** — the phase-0 constants transfer; n\*_direct ∈ [2, 5] | **PASS — n\* = 2.78, inside the modeled 2.62–2.80** |
| **E3b** — H2D hideability under real CPU GEMV (reported, not gated) | **88% hidden → effective n\* = 1** |

## E1 — the paying set is not a set, it is the population

Fraction of invocations whose expert recurs ≥ 2 more times within W steps at
that layer (medians over four prompts):

| model | W = 8 | **W = 32 (gate)** | W = 128 | ≥ 4 more at W = 32 |
|---|---|---|---|---|
| OLMoE | 56.8% | **91.2%** | 99.2% | 75–79% |
| Granite | 71.4% | **95.9%** | 99.6% | 88–92% |
| Qwen1.5-MoE | 20.4% | **71.1%** | 97.8% | 32–43% |
| gpt-oss | 65.2% | **91.9%** | 98.6% | 79–84% |

The registered bar was 25%. The worst cell is 68.7%. Even at the post-G2-fix
break-even (≥ 4 more uses), every model clears the bar. The economics do not
depend on the CPU kernel staying slow.

## E2 — refuted as registered, and the honest reading is better than a pass

Neither identifier reached 1.25× on all four models. But the numbers force a
structural observation the preregistration did not anticipate:

| model | base rate | **max possible lift (1/base)** | rank-1 | frequency |
|---|---|---|---|---|
| OLMoE | 91.2% | 1.10× | 1.06× | 1.07× |
| Granite | 95.9% | **1.04×** | 1.02× | 1.04× |
| Qwen1.5-MoE | 71.1% | 1.41× | 1.01× | 1.18× |
| gpt-oss | 91.9% | 1.09× | 1.04× | 1.06× |

**On three of four models the registered bar was mathematically unreachable
given E1's outcome** — with a 92–96% base rate, no subset can be 1.25× better
than everyone. This program's rule is that a prediction that cannot fail is
uninformative; the mirror holds: a prediction that cannot *succeed* is too,
and E2 is reported as **uninformative** on those three, not as a clean
refutation. On Qwen, the one model where the ceiling (1.41×) left room, both
identifiers genuinely failed: rank 1.01×, frequency 1.18×.

The preregistration's miss-clause read "E1 passes, E2 fails ⇒ promotion is a
placement-time story, not a runtime one." **That inference assumed a modest
base rate and does not survive the measurement.** When 90%+ of invocations
pay, the payers do not need to be *found* — "retain what you just executed"
is within a few percent of perfect targeting. E2 failed because
identification is unnecessary, which is the opposite of the registered
interpretation, and is stated here as such rather than silently absorbed.

## E3 — the constants transfer, to within rounding

| quantity | phase-0/2 (2026-08-16 box) | this box | |
|---|---|---|---|
| CPU grouped GEMV | 134.0 GB/s | **134.4 GB/s** | 0.3% apart |
| H2D pinned | 52.3–56.0 GB/s | 56.2 GB/s | top of band |
| GPU GEMV (bf16 proxy) | 1572 (triad) | **950.7** | a real `mv` does not hit triad; disclosed in the receipt, and n\* below uses the measured value |
| **n\*_direct** | modeled 2.62–2.80 | **2.78** | **PASS**, dead centre |

## E3b — the number the controller is built on

The real phase-2 kernel running continuously on CPU; 4,933 pinned 13.22 MB
H2D copies (~65 GB) issued concurrently on a side stream:

| | wall |
|---|---|
| CPU batch alone | 1.792 s |
| H2D batch alone | 1.165 s |
| **both concurrently** | **1.931 s** |

The copies added 0.139 s — **88.0% of their standalone wall hidden**, at full
link rate (56.0 GB/s alone). Joint completion inflated 7.8% while absorbing
the entire copy stream. Plugging the measured hideability into the break-even:
the first promoted invocation is *already cheaper than executing on CPU* —
**effective n\* = 1**. Under measured overlap, promotion pays from first use.

## What gate E establishes for the design

The elastic-saturation controller's three open questions now have measured
answers:

* **Who to promote:** nearly anyone — retain-on-execute, no selector. The
  rank and near-miss machinery is not needed for this and stays unused.
* **What a copy costs:** ~12% of its nominal wall when CPU work is in flight,
  with the link running at full rate concurrently.
* **What binds:** the transient pool's VRAM budget and eviction back-pressure
  — which is the capacity question this program has already characterised
  (LFU-style retention is the one lever that moves transfers;
  `protected = rows − k`).

Phase 2 — the controller itself, two residency populations, hysteresis, the
`min max(T_GPU, T_CPU, T_storage)` loop — is **justified and remains out of
scope here**, exactly as registered. Its spec should start from effective
n\* = 1 and the E1 table above.

## Preconditions

All five harness preconditions passed before scoring (known counts exact, the
corrected qualify bar pinned, window edges leave the denominator, a no-reuse
synthetic scores E1 = 0.000, rank-shuffle collapses the E2a lift 1.047 →
1.002). The E3 CPU number comes from the committed phase-2 bench driven
through its own arena builder and entry point, not a reimplementation. The
preregistration's off-by-one (total invocations vs remaining recurrences) was
caught by Bugbot on #201 and corrected **before** any scoring; the prereg
records the original bar.
