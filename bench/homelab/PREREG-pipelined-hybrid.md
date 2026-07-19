# PREREG — pipelined hybrid ladder (gpt-oss-120b, one H200-class pod)

**Status: STAMP-READY, NOT STAMPED (operator's key). NO POD BEFORE THE STAMP.**
Written before any pod data for the pipelined engine exists. Code under test:
e4b `7864955` + `e7f8535` + `8911e0d` (pipelined residency), kernel working
tree as shipped in the bundle (state recorded in the run's RESULTS).

## Receipts the bands derive from (nothing else is admissible)

- `RESULT-hybrid-curve-v0-h200.md` — measured pipe 55.2 GB/s; v0 250 ms/tok
  at K=0 (transfer floor ~34 ms ⇒ the **215 ms bounty**); v0 61 ms/tok at
  K=128 (zero transfer); llama pinned rows (resident 241.18/240.83, ncmoe32
  24.34/23.28, ladder 72.58/43.55/30.44/21.26).
- `RECEIPT-pipelined-phase1-devbox.txt` — 4000 Ada: eager 437 / graphs 262
  μs/layer zero-transfer bundle at HALF dims (E=128, k=4, H=I=1536, 3.8
  MiB/expert); K=0−K=128 delta ≈ 452 μs for 15.2 MiB ⇒ cold path at line
  rate; 25/25 + 18/18 gates; compute-sanitizer clean; A2000 (sm_86) 25/25.
- Stamped router receipts (specstream reanalysis) — held-out per-layer
  top-16 capture ≈ 30%. Per-K capture for K≠16 is derived **at run time by
  the committed reducer from the same stamped receipts** (paths + hashes
  cited in RESULTS); no capture number in this slip is invented.

## Named uncertainties (each carries width, not a point)

- **U1 — dim/silicon extrapolation of the overhead class** (operator-ruled
  the derivation's soft joint). Launch **counts** hold exactly (fixed
  enqueue sequence; sync-audited), but per-launch cost on the pod's silicon
  vs consumer Ada is unconstrained by any receipt, and occupancy shifts
  with dims are unmodeled. Width: launch-share multiplier **s ∈ [0.5,
  2.0]**. Compute share scales by receipts: weight bytes 3.8→13.1
  MiB/expert; pod HBM BW measured at run.
  ⇒ OH_graphs/token = 36×(220 μs·s + c_full) ∈ **[4.4, 16.2] ms**;
  OH_eager/token = 36×(395 μs·s + c_full) ∈ **[7.5, 28.8] ms**
  (c_full ≈ 52.4 MiB / HBM_BW ≈ 11 μs at H200-class).
- **U2 — non-expert per-token share** (attention/router/norms through our
  stack): unmeasured on the pod. Bounds from receipts: ≥ ~2 ms (llama's
  whole-stack resident 4.15 ms/tok is the silicon floor; ours does not beat
  their full stack), ≤ 45 ms eager (v0's K=128 total 61 ms is an upper
  bound on nonMoE + v0's hot-branch overhead). Bands use U2_graphs ∈ [2,
  20] ms, U2_eager ∈ [4, 45] ms. The protocol includes a decomposition arm
  to pin U2 with the first measurement.
- **U3 — capture-vs-K**: only K=16 is receipted (≈30%). Reducer-derived
  values cover other K before grading; if underivable, transfer-band
  grading applies only at K ∈ {0, 16, 128}.

Pipe/HBM substitution rule: bands are stated at 55.2 GB/s; grading
substitutes the pod's own measured pipe + HBM BW into the same arithmetic
(the formulas above are the prediction; the constants are measured).

## Predictions

- **P1 — correctness (void gate, never scores):** next-token-logit b_rel
  < 3e-2 vs the bare reference forward at **every K, eager AND graphs-on**.
  A graphs-arm miss demotes that arm to eager and is reported, not hidden.
  Any eager miss voids the run.
- **P2 — zero-transfer overhead collapse (graphs, K=128):** total token
  time ∈ **[6, 36] ms** (OH_graphs + U2_graphs), vs v0's measured 61 ms.
  *Falsify above:* overhead did not collapse — read via the split suspects
  below. *Falsify below:* U1/U2 widths over-conservative; tighten in
  RESULTS.
- **P3 — pure streaming (graphs, K=0):** token time ∈ 33.4 ms(transfer, at
  measured pipe) + [6, 36] ⇒ **[39, 70] ms ⇒ 14.3–25.6 tok/s**; achieved
  fraction of transfer floor ∈ **[48%, 85%]** (v0: 13.6%). Two-sided
  falsifiers; under-band reads via the split suspects.
- **P4 — eager arm (K=0):** **[45, 108] ms ⇒ 9.3–22.3 tok/s** (honestly
  wide — U2_eager dominates; graphs is the headline arm).
- **P5 — crossing:** graphs-on hybrid crosses llama's pinned **24.3 tok/s**
  at some K* ≤ 64, conditional on reducer-derived capture(64) ≥ 40% (U3);
  if capture is underivable, grade the crossing question at receipted-K
  points only. *Falsify (no crossing anywhere ≤ max-K):* overhead
  under-modeled — split suspects. *Falsify (crossing already at K=0):*
  bands far too conservative — over-band reading.
- **P6 — dial monotonicity:** median tok/s non-decreasing in K (r≥3;
  adjacent-K inversions within 1 sd do not falsify).
- **P7 — traffic accounting:** at K=0, cold_pcie_bytes/token ∈ [95%, 100%]
  of 1.842 GiB (near-uniform routing per receipts ⇒ slot-cache hits rare);
  at K=128, cold ≈ 0 and hot_d2d_bytes/token ≤ 36·4·13.1 MiB (the accepted
  fast-shelf re-copy, on the record via its counter). Counters must
  reconcile with the time attribution.
- **P8 — fidelity (measurement-only, unbanded):** matched-prompt perplexity
  through both stacks. No receipt exists to band a cross-stack ppl delta;
  this is its first measurement, recorded not scored. Cross-format caveat
  (verbatim, both here and in RESULTS): **the two stacks serve different
  quantization formats of the same checkpoint — the GGUF-native shipped
  quant vs our NF4 conversion — so every comparative number bundles format
  + engine differences.** [Rail-tension note for the operator: the Phase-5
  clause asks for the format names verbatim; the rails ban one of those
  format-name strings in slips. This slip uses the format-neutral phrasing
  above; raw tool logs in receipts remain verbatim. Ruling requested at
  stamp time.]
- **Energy (measurement-only):** pynvml GPU energy/token for our arms and
  llama's GPU arms; RAPL (`/sys/class/powercap`) attempted for llama's
  CPU-expert arms — if unreadable on the rental, the absence is recorded,
  never estimated.

## Pre-committed readings (before any data)

**Under-band** — three separate suspects, each with its named diagnostic:
1. *Full-dims overhead grew* (U1 was real): pod layer-step decomposition —
   launch share vs compute share vs the half-dims receipt.
2. *Sync regression* (a host sync crept into the full-model loop): the pod
   sync-audit arm (`set_sync_debug_mode` over N decode steps) + a kineto
   trace; distinct signature from (1).
3. *Transfer inflation* (moving more bytes than the math): traffic counters
   vs bytes arithmetic — distinguishes traffic inflation from time
   inflation.

**Over-band** — name which term (U1 width, U2 upper, capture) was
overestimated; tighten it in RESULTS with the measured value. No silent
re-derivation in either direction.

## Protocol (Phase 6, fires only after the stamp)

Same pod class (H200 141 GB). Re-measure pipe AND HBM BW (they enter the
band arithmetic). K-ladder {0, 8, 16, 32, 64, 128-if-VRAM} × {graphs,
eager} × r≥3 (24-token greedy decode, median-of-medians). Hot sets derived
on-pod by the committed reducer from the stamped router receipts (paths +
hashes in RESULTS). llama overlay re-run on the same box at the **frozen
pinned configs** (same image, flags, threads — any deviation is a steelman
documented in RESULTS, not a silent change). Sync-audit arm + per-layer
decomposition arm (pins U2). Perplexity arm both stacks, matched prompts.
Energy columns per above. Watcher armed; independent hardcap; artifacts
land before the pod self-destructs; destroy + 404-verify. Budget ≤ ~2.5 h
H200 (≈ $9) hardcapped; one session — a second only on operator GO.

## Confidence adjuncts (no bands attached)

- A2000 (sm_86, gen3-class link, consumer allocator): **25/25** — all three
  engine suites incl. graph capture+replay and the fetch-kernel unit suite.
  The target-class silicon shows no capture or UVA quirk.
- 4000 Ada dev box: 25/25 + interpreter-mode hardening; compute-sanitizer
  memcheck clean on the fetch kernel.
- Phase-4 gate substitution ratified by operator ruling (DO gates + A2000
  adjunct in place of an A2000-as-blocker).
