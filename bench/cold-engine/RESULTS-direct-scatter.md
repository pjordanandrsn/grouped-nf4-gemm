# RESULTS — preadv scatter vs copy on the cold fill path

Measures what [#118](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/118)
bought. #118 shipped the mechanism and explicitly claimed no speedup; this is
the number.

Receipts in `direct-scatter/` (5 runs + the harness).

## Verdict: the direct scatter is **~43% faster** on the fill path

Host: Xeon E5-2697A v4 (32 threads, 2.6 GHz), 62 GB RAM, NVMe advertising
3372 MB/s, torch 2.8.0+cu129, gnf4 `6bfc3db`. Synthetic arena at OLMoE-1B-7B
expert geometry (2.36 MB rows, 8 layers × 64 experts), skewed top-8 routing
over 480 `ensure` calls. A/B/A ordering with a self-pair, per instrument
law 6.

| receipt | hot_rows | copy-A | copy-B | direct | self-pair | Δ | reads matched |
|---|---|---|---|---|---|---|---|
| bench_direct | 96 | 2164 | 2298 | **1233** | 6.2% | **−43.0%** | no |
| bd_r1 | 96 | 2053 | 2330 | **1243** | 13.5% | **−39.4%** | no |
| bd_r2 | 96 | 2256 | 2203 | **1245** | 2.4% | **−43.5%** | no |
| bd_r3 | 96 | 2211 | 2439 | **1255** | 10.3% | **−43.2%** | no |
| bench_direct_nofail | 512 | 1515 | 1834 | **950** | 21.1% | **−37.3%** | **yes** |

Median Δ on the tight config: **−43.1%**, range −39.4% to −43.5% over four
independent runs.

## The direct arm is also far more stable

Across the four `hot_rows=96` runs:

* direct totals: 1233 / 1243 / 1245 / 1255 ms — **1.8% spread**
* copy totals: 2053 … 2439 ms — **18.8% spread**

The copy path's variance is what makes its self-pair loose (2.4–13.5%). The
memcpy is not just costly, it is *erratic*, and the loose self-pair in the
copy arm is a symptom of the thing being removed rather than of the
instrument.

## Where the time goes: achieved read throughput

Same bytes, same syscall count — one `preadv` per expert row either way.

| | achieved |
|---|---|
| copy | **1.42–1.53 GB/s** |
| direct | **2.45–2.49 GB/s** |

Against a drive advertising 3372 MB/s. The copy path leaves roughly a
gigabyte per second on the floor because the per-segment memcpy sits
*between* reads on the same thread: the read cannot overlap the copy of the
row before it, so achieved bandwidth collapses toward the copy's speed.

## Attribution, stated rather than assumed

The tight-config arms are **not byte-matched in work**: copy issues 1333
reads and direct 1301. That is a consequence of the change, not an
instrument defect — `segment_into` calls `tier.ensure` *again* per segment,
so the copy path re-touches keys and perturbs the LFU ranking into slightly
different eviction decisions. The measured gain therefore bundles two
things:

1. the removed per-segment memcpy, and
2. the removed redundant `tier.ensure` per segment.

**The `hot_rows=512` run separates them**: with no eviction, both arms read
exactly 469 rows, LFU cannot diverge, and direct is still **−37.3%**. So the
bulk of the effect survives when the read counts are identical. That run's
self-pair is loose (21.1%) because it is dominated by one-shot cold-start
fills, so it is reported as the *clean-attribution* arm rather than the
headline.

## What this does not establish

- **Not an end-to-end serving number.** This is the fill path in isolation —
  no model, no GPU, no router. The gate-1 cost attribution says software is
  ~90% of cold-path cost at 1–10% cold mass, but how much of a *served*
  step this recovers needs the full engine.
- **One host, and an old one.** A Xeon E5 v4's memcpy throughput is well
  below a Zen 5's, so this likely **overstates** the benefit relative to the
  gate-1 box. Instrument law 7 applies: state the host class with every
  number, and this one is a 2016-era Broadwell.
- **Synthetic routing.** Skewed top-8 by a power law, not a captured trace.
- The GPU cold path is untouched by this change and unmeasured here.
