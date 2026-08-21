# Which Stage-3 verdicts could have come out the other way — and R10 on four models

No box. Harnesses: [`structural_check.py`](routing-trace/structural_check.py),
[`score_r4.py`](routing-trace/score_r4.py),
[`score_r10.py`](routing-trace/score_r10.py). Traces: OLMoE, Granite and
Qwen1.5-MoE in [`routing-trace/`](routing-trace/), gpt-oss-20b in
[`wall-real-routing-2026-08-21/`](wall-real-routing-2026-08-21/) where it was
captured — four prompts each, not duplicated here.

Reproduce, per trace:

```
python score_r10.py --trace <trace>.jsonl --k <that trace's top_k>
```

`--k` matters; see the harness defects at the end.

[`RESULTS-third-model.md`](routing-trace/RESULTS-third-model.md) established
that a confirmation is worth nothing if no input could have refuted it, and
that three of this program's offline claims are in exactly that position. This
applies the same test to the R-series verdicts that have offline scorers, and
then does the thing the test says is worth doing.

| claim | verdict as published | could routing have flipped it? |
|---|---|---|
| cache crosses at one step | CONFIRMED | **no** — 0 of 24 conditions |
| LRU/FIFO zero-hit below one step | CONFIRMED | **no** — 0 of 24 |
| `headroom ≤ 1` ⇒ demand wins | CONFIRMED | **no** — 0 of 27 |
| **R4** — recurrence beats frequency | REFUTED | **yes** — flips in 1 of 9 |
| **R10** — reclaimable residency cuts refills | REFUTED | **yes** — flips in 5 of 9 |

The R-series verdicts are the informative ones. Both were scored on a single
model, and both are falsifiable, so both are worth re-scoring out of sample.

## The falsifiability test

Synthetic routing at the captured geometry, stickiness ∈ {0, 0.6, 0.95} ×
Zipf popularity exponent ∈ {0, 1.5, 3.0}, nine conditions.

**R4** (`recency` vs `frequency` cells, OLMoE geometry) flips to *holds* at
independent-draws + Zipf-3, and its recency count varies from 0 to 2 across
conditions. **R10** flips in five of nine, reaching 6 of 10 cells *holding*
at stickiness 0.6 with Zipf-3.

So neither refutation is arithmetic. Where do real models sit on that grid?

| model | stickiness | Zipf exponent |
|---|---|---|
| OLMoE | 0.33 | 1.27 |
| Granite | 0.30 | 1.05 |
| Qwen1.5-MoE | 0.07 | 1.16 |
| gpt-oss-20b | 0.36 | 1.30 |

All four sit *below* the region where R10 flips — between the (0, 1.5) cell
that refutes 10 of 10 and the (0.6, 1.5) cell where 4 of 10 hold. Close enough
to the boundary that interpolating would be guessing, so the traces were
scored directly instead.

## R10, scored on four models instead of one

`RESULTS-tribrid-reclaimable.md` scored R10 on OLMoE alone. 160 cells now —
4 models × 4 prompts × 10 (rows, protected) points:

| model | REFUTED | holds |
|---|---|---|
| OLMoE | 40 | 0 |
| Granite | 40 | 0 |
| Qwen1.5-MoE | 38 | 2 |
| gpt-oss-20b | 35 | 5 |
| **total** | **153** | **7** |

**R10's refutation replicates.** The seven exceptions are not a counter-result
and should not be read as one:

| trace | rows | prot | Δ reads | Δ churn |
|---|---|---|---|---|
| gptoss prose | 512 | 256 | −0.3% | −0.6% |
| gptoss prose | 512 | 508 | −0.8% | −1.4% |
| gptoss code | 512 | 256 | −0.2% | −0.3% |
| gptoss code | 512 | 508 | −0.3% | −0.4% |
| gptoss dialogue | 512 | 508 | −0.4% | −0.6% |
| qwen math | 512 | 256 | −0.4% | −0.5% |
| qwen math | 512 | 508 | −0.4% | −0.5% |

**Every one is at rows = 512**, the largest capacity swept, and every one is
under 1%. The scorer's verdict is binary — any improvement at all reads as
*holds* — and R10 was registered as *reduces* churn and rereads, which a 0.3%
reduction satisfies literally and not usefully. The replay is deterministic,
so these are exact rather than noise; they are simply negligible.

The honest summary is that R10 is refuted on four models, and that at the top
of the capacity sweep the two top-4 models show a sub-1% improvement that
changes nothing operationally.

## Two harness defects found on the way

**`score_r10 --k` sets the protected budget**, as `rows - k`, and defaults to
8. Running a top-4 trace at the default therefore sizes the budget for a
top-8 engine — the mis-sizing that
[`RESULTS-third-model.md`](routing-trace/RESULTS-third-model.md) documents as
producing total cache thrash. Every number above was re-run with `--k` matched
to each trace's own `top_k`; the tallies were identical, but that is a fact
that had to be checked rather than assumed.

**Granite's trace carries `n_experts: null`** and crashed `build_arena` with a
bare `TypeError`. That null is the `num_experts` / `num_local_experts` bug
fixed in `capture_routing.py` earlier; the trace predates the fix. `score_r10`
now falls back to the largest routed id and labels the count `INFERRED lower
bound` rather than presenting it as the real one.
