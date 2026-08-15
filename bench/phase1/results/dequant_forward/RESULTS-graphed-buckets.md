# RESULTS — dense-groups bucketing: the graphed race, run fairly at last

**2026-08-15 · fidelity on RTX A2000 (sm_86), race on a whole-machine RTX 3060
12 GB (12/12 cores, GDDR6) · torch 2.8.0+cu128 · total metered spend ~$0.02.**
Grades [`kernel/prereg_graphed_buckets.json`](../../../../kernel/prereg_graphed_buckets.json)
and its three amendments, every one OTS-stamped before the data it grades.
Receipts in [`graphed_buckets/`](graphed_buckets/).

The capturability arc (§§1–12 of
[`RESULTS-capturability.md`](RESULTS-capturability.md)) made the fused training
step graphable and said, every time: capturability is a precondition. This is
the experiment it was a precondition FOR — the graphed-vs-graphed race
`probe_graphed_race.py` defined as the fair fight and could not run.

## The scheme

`G_pad = E` groups, every group padded to the same `R` (buckets {1…128},
`R = next_pow2(max group size)`), `eids = arange(E)`, `sizes = [R]*E` — every
launch grid step-invariant. Per step, *outside* the graph: routing on the host,
token activations and token→slot indices written into persistent buffers.
The graph: zero-fill, scatter, forward, loss over the (constant-count) real
rows, backward. **Waste is reported in both currencies**: rows read 8.0× padded;
tiles ~1.0–1.1× — and tiles are what cost.

## F2 — fidelity, reached through two falsified controls (both stamped)

* **Amendment 1**: bitwise-vs-unbucketed was a broken oracle for the LoRA path —
  cuBLAS bmm kernel choice varies with batch shape (the leg-1 Ada mechanism),
  and the unbucketed path does not equal *itself* across routing draws. The
  base path stays bitwise, and passes: outputs, a-grads, replay≡eager.
* **Amendment 2**: the garbage-contamination positive control was
  *unsatisfiable* — `grad_out` is exactly zero on padded rows, so ordinary
  garbage cannot touch the LoRA grads, falsifying my "zero-fill protects grad
  accumulation" rationale. The zero-fill's true load-bearing role is **NaN
  poisoning** (`0 × Inf = NaN` through the same reductions); the control now
  proves that channel exists and that zero-fill closes it.

Final state: **8/8 cells PASS** on the race card (leak bitwise, determinism
bitwise, replay bitwise, NaN control fires, values 1.4e-4–1.5e-3 against the
6.5e-3 floor).

## F3 — the fair race: the fused advantage SURVIVES graphing

Both arms bucketed identically, both replayed; fresh routing draws cycled
through replays. Registered band [1.4, 2.8] on GDDR6, from leg 4's measured
GPU split (d/g_gpu 2.062 on GDDR6X):

| cell | `D_graphed / G_graphed` | replay self-pairs |
|---|---:|---|
| gate_up, T=32 | **1.647** | 1.000 / 1.000 |
| gate_up, T=128 | **1.511** | 1.000 / 1.000 |
| down, T=32 | **1.833** | 1.000 / 1.000 |
| down, T=128 | **1.461** | 1.000 / 0.999 |

**4/4 inside the band.** With both host floors removed, the fused kernel beats
the dequant-on-forward loop **1.46–1.83×** at the decode band on consumer
memory — the memory-traffic component leg 4 isolated, finally measured with
nothing else in the frame. And the self-pairs say something worth keeping:
**a replayed graph is the cleanest instrument this program has ever held** —
1.000 to three decimals, because the host is out of the loop entirely.

## F4 — FALSIFIED: graphing does not pay at the decode band on this card

Registered band [2.0, 15.0] for graphed-vs-shipped-eager; measured **0.91× /
0.53× / 0.93× / 0.53×**. The registered falsifier's own words apply: replay and
staging overhead ate the host floor, and graphing does not pay here. The
registered interpretation (amendment 3, fixed before the completion run): the
band was calibrated from fast-GPU measurements where the eager decode step is
~90% host; on a 3060 the GPU is slow enough that the eager step is already
GPU-dominated, so there was no host floor to remove — **the decode-band host
floor is a fast-GPU phenomenon**, and at T=128 the padding's element-wise
overhead (zero-fill, scatter, loss over 8× rows) actively costs. A fast-GPU F4
would need its own registration; nothing here licenses one.

## Three registered refusals fired in this experiment, and each was right

1. The bitwise LoRA oracle refused (amendment 1) — wrong oracle, not wrong code.
2. The garbage control refused to fire (amendment 2) — wrong mechanism claim.
3. The bucket cap refused T=128 (amendment 3) — the registration's cell list
   and bucket set were mutually inconsistent, and the code would not guess.

## What this licenses, and does not

* **Licensed**: at the decode band, under graphed execution on GDDR6-class
  memory, the fused kernel outperforms the graphed dequant-on-forward loop
  1.46–1.83× with 8× row padding and ~1.0–1.1× tile overhead, at bitwise (base)
  / floor-level (LoRA) fidelity. This is the claim the leg-4 headline reached
  for and could not have; it is scoped to graphed execution and this memory
  class.
* **Not licensed**: any eager small-batch claim (the graphed-race verdict
  stands); any claim that graphing accelerates the fused path on this device
  class (F4 falsified it); anything at all on HBM3 until a registered run
  exists there.

## Errata (receipts, reported fields)

`arena_owned_ints` reads **0 in every committed race receipt** and is WRONG —
the readout keyed `_ARENAS` with `"cuda"` while live tensors key it `"cuda:0"`,
so a reported-not-graded field silently nulled. Caught by Bugbot on PR #90
after I printed the impossible zero without noticing; the receipts stay as
produced, this note marks the field, and the readout is fixed for future runs.
A second latent defect (fidelity `not_run` rows vetoing the race gate via a
missing key) is fixed in the same commit; it never fired in these runs.

## The HBM3 half (shared H100, ~$0.25 across two pods)

Grades [`kernel/prereg_graphed_buckets_hbm3.json`](../../../../kernel/prereg_graphed_buckets_hbm3.json)
(stamped pre-data). Receipts in [`graphed_buckets_hbm3/`](graphed_buckets_hbm3/).
F2 re-passed **8/8** on the H100 before racing — a third card class.

| cell | F3 `D_g/G_g` [0.9–1.4] | F4 [2–20] | replay self-pairs |
|---|---:|---:|---|
| gate_up T=32 | 1.245 ✓ | 1.15 ✗ | 0.980 / 0.997 |
| gate_up T=128 | 1.147 ✓ | 0.52 ✗ | 0.992 / 1.001 |
| down T=32 | **1.695 — ABOVE band** | 1.22 ✗ | 0.972 / 1.002 |
| down T=128 | 1.178 ✓ | 0.60 ✗ | 0.971 / 0.990 |

**F3: 3/4 inside the parity band** — HBM3 absorbs the dequant round-trip's
extra traffic, as leg 4's GPU split predicted — **with one cell above it**:
`down` at T=32 keeps a 1.695× fused advantage even on fast memory. Per the
registered falsifier that cell says the advantage is not purely
memory-class-bound; the smallest per-expert GEMM in the census is where
per-launch and occupancy effects would show, and naming the mechanism there
needs its own registered work, not this paragraph.

**F4: falsified 4/4 — and the second falsification names the real defect.**
The band was calibrated from the host-heavy probe/e2e drivers (9–33% GPU-busy
eager steps); the harness's own eager comparator is a lean pre-built loop with
almost no host work to remove. The registered premise never applied to the arm
that was graded. F4 across both card classes now reads: graphing the fused
step buys nothing against a lean eager loop, ~1.2× at best at T=32, and loses
at T=128 where the 8× padded rows cost elementwise work.

**Shared-pod immunity: falsified as registered.** Self-pairs read 0.971–1.002
against the registered [0.99, 1.01]: per-replay staging still touches the
host, so graphs REDUCE contention sensitivity (~±3% vs the eager e2e's ±11%)
but do not eliminate it. The standing rule refines rather than falls: **shared
pods are valid for graphed bands wider than ~±3%; whole machines remain the
rule for anything tighter.**

## The mechanism scan (why down/T=32 sat above the parity band)

Grades [`kernel/prereg_graphed_rscan.json`](../../../../kernel/prereg_graphed_rscan.json)
+ [amendment 1](../../../../kernel/prereg_graphed_rscan_amendment1.json), both
stamped pre-data. Uniform-full R-scan (proj × R ∈ {1…128}, E=64, lora=False,
zero padding), shared SECURE H100 — the refined shared-pod rule's second
deliberate use, and it held. Receipts:
[attempt 1](graphed_rscan_h100/) (REFUSED, 12/16 void) and
[attempt 2](graphed_rscan_h100_attempt2/) (16/16 live). Two pods ≈ $0.37;
three additional creates wedged at uptime 0 and were destroyed verified-404.

**The refusal was the instrument, and its repair generalizes a shipped law.**
Every attempt-1 void was the G-arm self-pair on the cell's FIRST timed block
(0.923–0.964) with the D-arm at 0.997–1.001 in all 16 cells; ~300 ms blocks
still voided, so block length was not the story — the first block reads a
settling ramp lasting about a full block, the e2e ten-drift law at block
scale. One full UNTIMED primer block per arm (the first-mode discard,
translated) took the scan from 4/16 to 16/16 live, self-pairs 0.996–1.003.

| R | gate_up d/g | down d/g | gate_up excess ms | down excess ms |
|---:|---:|---:|---:|---:|
| 1 | 1.081 | 1.586 | 0.112 | 0.444 |
| 2 | 0.952 | 1.431 | −0.076 | 0.369 |
| 4 | 0.967 | 1.440 | −0.053 | 0.379 |
| 8 | 0.992 | 1.458 | −0.013 | 0.400 |
| 16 | 1.080 | 1.490 | 0.131 | 0.440 |
| 32 | **1.212** | **1.593** | 0.363 | 0.569 |
| 64 | 1.076 | 1.296 | 0.205 | 0.443 |
| 128 | 1.106 | 1.098 | 0.493 | 0.258 |

**P1 FALSIFIED, both projections** (Spearman −0.40 down, **+0.5 gate_up**
against ≤ −0.7): the ratio is non-monotone with a peak at R=32 in both
projections, and at gate_up R=2–8 the per-expert dequant+cuBLAS loop BEATS
the fused kernel outright. The floor-dilution reading of the race's four
points is dead. **P2 CONFIRMED**: the flip replicates — down's excess is
1.567× gate_up's at R=32 (band ≥ 1.1); the anomaly is real. **P4 CONFIRMED**:
D-arm elasticity R=1→8 is 0.027/0.030 — at small R both arms are pure
weight-streaming (the profile agrees: the fused kernel is R-FLAT, 1544 µs at
R=8 → 1559 µs at R=32 on gate_up — it streams the full expert stack every
replay regardless of R).

**P3, by the registered rules: NO RULE FIRES — unattributed as registered.**
(a) GEMM kernel counts are EQUAL (128/replay = 2 per expert, both
projections; the splitk/reduce name matches are the same two loss-reduction
kernels in both tables); (b) down's per-GEMM mean is 4.79 µs vs gate_up's
5.93 µs — LOWER, not ≥, and 4× the 1.20 µs floor-probe yardstick; (c) the
excess does not sit outside the GEMM family — GEMM scales worst. Both
attempts' profiles agree bit-for-bit on structure.

**Post-hoc localization (reported, NOT graded)** — the excess is a compound
no single registered hypothesis owned:
1. an inter-kernel gap of ~115–123 µs/replay in the ~400-kernel base graph
   (wall exceeds summed device time 8.8–8.9%) vs ~0 in the 23-kernel fused
   graph — a true kernel-count cost, but EQUAL across projections, so it
   cannot produce the flip;
2. the projection flip lives in device time: going gate_up→down (half the
   bytes), the fused kernel scales 0.53× (byte-proportional) while the base's
   GEMM family scales only 0.81× (fwd nvjet_64x8 0.85×, dgrad nvjet_64x16
   0.76×) — the per-expert cuBLAS path streams down's [1024,2048] expert
   weights less efficiently per byte than gate_up's [2048,2048];
3. the R=32 peak in both projections: the base's 212-kernel elementwise chain
   grows ~4.4× faster with R in absolute µs than the fused arm's 22-kernel
   one (gate_up: +437 µs vs +98 µs from R=8→32), and above R=32 real GEMM
   growth dilutes it again.

**What this closes and opens.** The race's above-band cell is explained by
(2)+(3) at its exact operating point, sitting on top of (1); uniform-full
R=32 reads 1.593 vs the race's padded 1.695 (residual = real-row loss/index
work). It licenses no throughput claim. Two engineering targets fall out,
each needing its own registration if pursued: the fused kernel LOSES to
dequant+cuBLAS at gate_up R≤8 on H100 (headroom at large-K, tiny-R shapes),
and the base arm's elementwise chain — not its GEMMs — is what makes it a
bad graphed baseline at mid R.
