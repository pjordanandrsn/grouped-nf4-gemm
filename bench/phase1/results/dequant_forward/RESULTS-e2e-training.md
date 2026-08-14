# RESULTS — end-to-end: the two arms inside a real QLoRA finetune

### 2026-08-13 · RTX 4090 (sm_89) + H100 80GB HBM3 (sm_90) · torch 2.8.0+cu128 · **published wheels** e4b 0.17.5 · driver [`e2e_train_arms.py`](../../e2e_train_arms.py)

**Grades [`kernel/prereg_dequant_forward_e2e.json`](../../../../kernel/prereg_dequant_forward_e2e.json)** (OTS-stamped pre-data).
Model `allenai/OLMoE-1B-7B-0924`, 16 layers, 64 experts, 58.7M trainable, seq 512,
24 steps per arm with the first 4 dropped, medians of the rest.

Every other leg in this program is a microbenchmark — synthetic activations,
routing histograms captured offline, one op at a time. This is the same
comparison inside `experts4bit_qlora`'s real training pipeline. **The baseline is
e4b's own per-expert dequant-and-project loop**: the same family as GenON's
`QuantizedNaiveMoE`, but our code. Nothing here measures GenON's implementation.

## The result

`speedup` is reference-loop seconds per step over the arm's. All four
configurations, both devices, per cell:

| config | arm | RTX 4090 | H100 |
|---|---|---:|---:|
| **text, resident** | `fast_train` | **4.468×** | **4.697×** |
| | `fast_train_dgrad` | **4.504×** | **4.746×** |
| | `batched` | 1.960× | 2.504× |
| **text, offload** | `fast_train` | 2.527× | 4.063× |
| | `fast_train_dgrad` | 2.530× | 3.983× |
| | `batched` | 1.612× | 2.081× |
| **random ids**¹**, resident** | `fast_train` | 2.691× | 2.610× |
| | `fast_train_dgrad` | 2.753× | 2.813× |
| | `batched` | 1.034× | 1.072× |
| **random ids**¹**, offload** | `fast_train` | 1.754× | 2.325× |
| | `fast_train_dgrad` | 1.806× | 2.378× |
| | `batched` | 1.004× | 0.994× |

¹ **Random-id cells UNDERSTATE the fused advantage by 1.6–1.7×** — measured on
every matched pair in this run, mechanism below. Standing rail since 2026-08-14:
any table citing a random-id cell states this factor beside it, and real prose is
now the harness default (`e2e_train_arms.py --data`, which defaults to `text`;
random ids are opt-in for work that genuinely needs content-independence).

**P1 confirmed on both cards.** Predicted ≥1.5× for `fast_train_dgrad` on
OLMoE; the weakest of eight cells is 1.806×. The speed result is not
architecture-bound.

**Self-pairs, all eight cells: 0.967 – 1.032**, inside the registered
1.00 ± 0.05. `reference` ran first and last, bracketing each sweep, so this
covers drift across the whole run and not just noise. Every margin above is far
outside it.

## The finding that changes how this should be benchmarked

**Random token ids understate the fused advantage by 1.6–1.7×, on every matched
pair.** The driver this adapts feeds random ids, arguing that arm-vs-arm on
identical inputs makes content irrelevant. That is true for a dense model. It is
false for MoE, because routing is a function of token content:

| data | occupancy | cv | 4090 resident | H100 resident |
|---|---:|---:|---:|---:|
| real prose (wikitext-2) | 0.984 | 0.687 | 4.504× | 4.746× |
| random ids | 0.875 | 1.463 | 2.753× | 2.813× |

Random ids route to **fewer** experts and **far more unevenly** — the opposite of
the intuition that random input spreads load. Fewer hit experts means fewer
iterations of the baseline's Python loop, which is precisely the cost the fused
path removes, so the fiction flatters the baseline.

Routing was measured off the LIVE router during the timed run and is identical
across devices to three decimals, as it must be — routing is a property of model
and data, not silicon. This also closes the open item asking for occupancy on the
real model rather than derived: **OLMoE at seq 512 on prose hits 98.4% of 64
experts, cv 0.687**, consistent with the offline 2048-token histogram's 1.000 /
0.506.

**This makes the prior run's headline a substantial understatement.**
[`dgrad-gate`](../../../../../experts4bit-qlora/bench/dgrad-gate/RESULTS-dgrad-gate.md)
(2026-08-06, e4b 0.11.0, A5000, random ids + offload) reported **1.99×** for
OLMoE. The comparable cell here — random ids, offload — reads **1.806×** (4090)
and **2.378×** (H100), a clean replication two e4b minor versions and two
architectures later. But the same model on real prose with resident weights is
**4.50–4.75×**, more than double the number that study published, because of the
fixture rather than the kernel.

## P2 — memory does not improve, and my earlier framing was wrong

Registered against my own prior claim, and confirmed. Resident cells, text:

| arm | 4090 peak | H100 peak |
|---|---:|---:|
| reference | 5.31 GB | 5.35 GB |
| `fast_train` | 5.65 GB | 5.70 GB |
| `fast_train_dgrad` | 5.87 GB | 5.92 GB |
| `batched` | 6.95 GB | 7.00 GB |

The fused arms peak **higher**, not lower. Legs 1–4 measured 18.7–48.6×
**transient bytes per op** against a plain-`F.linear` baseline that saves its
weight for backward; that is a different quantity from peak VRAM in a finetune
and does not imply it. Measured here, transient does improve consistently
(1.581 → 0.821 GB, ~1.9×) and it does not reach peak. Any claim that this kernel
lowers the memory needed to finetune is unsupported by this run.

**And the self-pair says fine-grained peak claims are unsupportable anyway.**
Under offload, `reference_selfpair` peaked at 3.567 GB where `reference` peaked
at 2.688 GB — **1.33× apart on identical work**, allocator state alone. Every
peak difference in this leg below ~1.35× is noise, which is exactly the range all
the interesting ones sit in. Only the transient decomposition survives.

## Fidelity, and the arms that lose

Composed gradient error against the reference at step 0: fused lanes ~4.1e-02,
`batched` ~4.5e-03 — the same ordering `dgrad-gate` found and explained (the
kernel-free path shares the reference's dequantize-then-matmul rounding, so a
small vs-reference number measures similarity, not truth). Loss trajectories
track the reference throughout.

**`batched` loses on random ids** — 1.034× / 1.072× resident, 1.004× / 0.994×
under offload, i.e. at or below parity on both cards, including one cell
genuinely below 1. It only wins on real prose. Recorded because the standing rail
requires the losing cells.

**Frozen storage: 3.00 GiB hashed, 0 tensors changed**, every arm, both cards, in
the resident cells. No arm mutated quantized base bytes.

## What this does NOT license

* Not a measurement of GenON's implementation. The baseline is e4b's loop.
* **Absolute s/step is not comparable across the two pods.** The H100's reference
  loop is *slower* than the 4090's on the same work (2.5684 vs 1.8627 s/step,
  text resident) because the per-expert Python loop is host-bound and the pods'
  CPUs differ. Only within-pod ratios are meaningful, which is what is reported.
* One model, 16 layers, seq 512, 24 steps. Not a convergence or quality result:
  nothing here says the adapter trains to a better model, only that steps are
  cheaper.
* The `offload=1` integrity check is **not applicable**, not passed — see below.

## Two instrument faults found, both mine

**The frozen-storage check was near-vacuous under offload and then falsely
positive.** A params-only scan hashed 32 tensors / 0.00 GiB, because e4b keeps
the 4-bit experts in pinned CPU RAM and streams a layer at a time; the only
resident uint8 tensors are per-layer *staging buffers* whose contents change
every step by design. That produced "frozen bytes CHANGED" for all five arms
including `reference` against itself. The resident cell hashes 3.00 GiB of the
same model and reports zero changes. The check is now recorded as NOT APPLICABLE
under offload rather than as a pass or a failure, and the integrity claim is
carried by the resident cells only.

**A runner reported `OK` having measured nothing.** An earlier campaign uploaded
one tarball name and extracted another, so the repo never landed; an empty pytest
run passed a `failed|error` grep because "no tests ran" contains neither word;
and piping python to `tail` made `$?` the exit status of `tail`. Three guards now
stand there, and on this leg's first outing the first one tripped correctly
instead of writing a false OK.

Receipts: [`leg-e2e/`](leg-e2e/) — `{E2A,E2H}-off{0,1}.json`.
