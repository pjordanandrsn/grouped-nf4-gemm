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

## End to end: it pays where fills dominate, and nowhere else

The isolated number above is the fill path. This is the served step, with a
model, a router, attention and a GPU competing for the same wall. Only
`cold_direct` varies; each arm asserts the landing it actually took, so a
silent fallback cannot make both arms the same measurement.

Box: RTX 5090 + EPYC 9655 (Zen 5), gnf4 `9a85ab1` / e4b `3e1728e`,
OLMoE-1B-7B NF4 arena, 64-token prose prefill then 128 greedy decode steps,
A/B/A with a self-pair. Receipts `ab_direct.json`, `ab_direct_20.json`.

| cold mass | tier | reads in window | copy | direct | self-pair | Δ |
|---|---|---|---|---|---|---|
| 5% | hot_rows=384 | **37** | 55.43 / 56.15 ms | 55.69 ms | 1.31% | **+0.48% — null** |
| 20% | hot_rows=128 | **2963** | 80.49 / 81.12 ms | **70.40 ms** | 0.79% | **−12.53%** |

Tokens identical across every arm at both points.

**At 5% cold mass the fill path barely runs** — 37 reads across 128 steps,
about 0.3 fills per step — so a 43% faster fill is 43% of nearly nothing.
That is exactly what the gate-1 cost attribution predicted: at 1–10% cold
mass, storage (and therefore filling) is ~5–11% of what cold work costs, so
an optimization aimed at fills cannot move the wall there.

**At 20%, where the tier is under real pressure and fills dominate, it is
worth −12.5%** at sixteen times the instrument's own spread.

Worth noting the direct arm issued **more** reads than copy (3761 vs 2963)
and was still 12.5% faster — it is not winning by doing less I/O. The extra
reads come from the self-heal path (gnf4#121) dropping rows it had not
landed, plus the generation bumps that perturb LFU.

### One receipt was lost and the point re-run

The first 20% measurement (−15.90% on a different EPYC 9655 instance,
`B_dram` 453.4 GB/s against this one's 370.8) was destroyed by operator
error: the scp and the instance teardown were issued in one command, the
scp failed on a local path that did not exist, and the destroy ran anyway.
Rather than cite console scrollback, the point was re-run on a fresh box of
the same class — hence −12.53% here. Both runs agree in direction and
magnitude; only the second has a receipt, and only the second is cited.

## What this does not establish

- ~~Not an end-to-end serving number.~~ **Measured** — see the section
  above. It is worth nothing at 5% cold mass and −12.5% at 20%.
- **One host, and an old one.** A Xeon E5 v4's memcpy throughput is well
  below a Zen 5's, so this likely **overstates** the benefit relative to the
  gate-1 box. Instrument law 7 applies: state the host class with every
  number, and this one is a 2016-era Broadwell.
- **Synthetic routing.** Skewed top-8 by a power law, not a captured trace.
- The GPU cold path is untouched by this change and unmeasured here.
