# Gate 1 re-run: the published read counts were warmup-inclusive, and it is now measured

Receipts: [`gate1_rerun.json`](gate1_rerun.json), [`calib.json`](calib.json),
[`gate1_rerun.log`](gate1_rerun.log). Box destroyed after the pull.

`RESULTS-tribrid-gate1.md` carries a correction saying its read counts are
warmup-inclusive and that the size of the error was unmeasured. This is that
measurement. It is **not** a wall-clock re-run of gate 1 — see "what does not
transfer" — it is a correction to the read counts and to the cost attribution
built on them.

## Host

RTX 5090 · **AMD EPYC 7B12 (Zen 2)** · driver 580.159.03 · torch 2.13.0+cu130 ·
triton 3.7.1 · under `numactl --interleave=all` (2 NUMA nodes).

**This is not the published run's host.** That was 5090 + **Zen 5**. This box
is materially weaker off the GPU: DRAM triad **148.5 GB/s** against 380.1, and
NVMe sequential **2.42 GB/s** against 5.51. Same model, same arena geometry
(L=16, E=64, row 3,538,944 B), same config — sweep, `steps=128`, `warmup=8`,
`seq=64`, `hot_rows=384`, `vram_frac=0.25`, `order=tail`, `source=dram`.

## The instrument change

`run_gate1.py` now snapshots `cold_stats` **twice** — once at model load and
once at the measurement boundary — so a single run reports both windows and
the size of the difference is measured on one trace rather than inferred
across runs. `reads_since_load` is the quantity the pre-fix harness was
unknowingly differencing against; `reads_in_window` is what gate 1 asks for.

## The published counts reproduce as `reads_since_load`

| cold | arm | **published "win reads"** | `reads_since_load` here | agreement | `reads_in_window` here | overstatement |
|---|---|---|---|---|---|---|
| 1% | cold-GPU | 105 | 112 | +7% | **25** | 4.5× |
| 1% | cold-CPU | 106 | 113 | +7% | **26** | 4.3× |
| 5% | cold-GPU | 238 | 241 | +1% | **27** | 8.9× |
| 5% | cold-CPU | 238 | 249 | +5% | **36** | 6.9× |
| 10% | cold-GPU | 340 | 335 | −1% | **26** | 12.9× |
| 10% | cold-CPU | 335 | 347 | +4% | **38** | 9.1× |
| 20% | cold-GPU | 3400 | 3437 | +1% | **255** | 13.5× |
| 20% | cold-CPU | 2025 | 1597 | −21% | **536** | 3.0× |

Seven of eight agree within 7%. **That is the finding.** The published
figures were not approximately warmup-inclusive — they *are* the since-load
counts, reproduced on different silicon to within a few percent.

They reproduce because a read count is a property of the **routing trace and
the placement**, not of the hardware: which cold rows are touched over 128
decode steps against a 384-row tier is fixed by the model and the manifest,
and both are identical here. The same argument is why the corrected counts
transfer back to the published box.

The 20% cold-CPU row is the one loose end at −21%. At 20% the tier genuinely
thrashes, so the count becomes sensitive to eviction order in a way the
lighter points are not; it is reported rather than smoothed.

## Corrected cost attribution for the PUBLISHED box

Published Δ ms/step and the published box's **sequential** ceiling
(5.51 GB/s — not the 6.26 GB/s the addendum used, which is that box's
*random* qd16 rate), with the read counts corrected to the decode window:

| cold | arm | Δ ms/step | corrected reads | disk ms/step | **disk share of Δ** | (as published) |
|---|---|---|---|---|---|---|
| 1% | cold-GPU | 8.33 | 25 | 0.125 | **1.5%** | 5.6% |
| 1% | cold-CPU | 4.29 | 26 | 0.131 | **3.0%** | 10.9% |
| 5% | cold-GPU | 10.30 | 27 | 0.136 | **1.3%** | 10.2% |
| 5% | cold-CPU | 11.00 | 36 | 0.181 | **1.6%** | 9.6% |
| 10% | cold-GPU | 16.66 | 26 | 0.131 | **0.8%** | 9.0% |
| 10% | cold-CPU | 14.87 | 38 | 0.191 | **1.3%** | 9.9% |
| 20% | cold-GPU | 24.53 | 255 | 1.280 | **5.2%** | 61.2% |
| 20% | cold-CPU | 29.86 | 536 | 2.690 | **9.0%** | 29.9% |

**Storage is 0.8–3.0% of what cold work costs at 1–10% cold mass**, not
5–11%. Even at 20%, where the addendum called disk "the dominant term" at
61%, it is **5–9%** — the thrashing it described was warmup traffic.

## What this does to gate 1

**The MISS stands, and its reframing gets substantially stronger.** The
hide-ratio clause asks whether NVMe latency can be hidden under scheduled
work. A perfect prefetcher can remove at most the storage fraction of cold
cost; that fraction is ~1–3%, not ~10%. The verdict itself never rested on
these counts — it rests on prefetch coverage, under 1% of demand misses,
which is a ratio of two counters *inside* the same window and is unaffected
by where the window starts.

The addendum's conclusion — "the other ~90% is the cold path's own software
cost" — becomes "the other ~97–99%".

## What does NOT transfer, and is not claimed

**Wall-clock numbers from this box are not gate 1.** Δ ms/step here is 24–42×
the published values (e.g. 699 ms at 10% cold-GPU against 16.66 ms) because
Zen 2 is far slower at the cold path's per-call software cost — which is
precisely the term the attribution says dominates. Those walls are in the
receipt and are **not** used above; every Δ in the corrected table is the
published box's own.

**This is not a re-run of gate 1's verdict.** The gate's clauses were not
re-scored on this host and the receipt should not be read as re-scoring them.

**The device row cache was not exercised.** It is wired into
`Mxfp4NvmeResidency`; gate 1 runs the NF4 hybrid tier. Measuring the cache
against a real routing trace needs an MXFP4 model and a different harness,
and remains open.
