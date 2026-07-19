# Session-3 note — robustness + steelman (EXPLORATORY, pre-data)

Committed before firing. Tier: exploratory (unstamped; git timestamp is the
pre-data evidence). Protects the stamped RESULTS' crossing claim before any
publication. Same box class; engine commits unchanged.

- **E1 llama thread steelman** (the frozen -t 8 pin is MY harness choice,
  not llama's optimum; box has 24 vCPU): ncmoe32 tok/s rises with threads,
  sublinear; expect -t24 ∈ [28, 55] ⇒ the crossing K* moves right into
  [32, 128] but SURVIVES. *Falsify low:* -t24 ≤ 24.5 ⇒ t8 was already
  memory-bound; headline stands unchanged. *Falsify high:* -t24 > 62 ⇒
  llama beats even our K=128 on fat-CPU boxes; the crossing claim is then
  weak-host-only and gets restated plainly.
- **E2 prompt panel** (8 prompts × 64 tok, graphs, K ∈ {0,16,32}): median
  across prompts at K=16 ∈ [22, 27] tok/s; worst prompt ≥ 20. Tests capture
  generalization + slot-cache flattery from the single-prompt greedy loop.
- **E3 soak**: 512-replay run at K=32; second-half median within 3% of
  first half.
- **E4 clean attribution** (counters + energy snapshotted AFTER capture):
  per-replay cold traffic at K=0 ∈ [1.3, 1.9] GiB/tok.
- **E5 fidelity, methodology-matched**: ours computed chunked-512 like
  llama's tool, longer matched text (~2.4k tok); expect ratio ours/llama ∈
  [0.9, 1.3]. Measurement-tier.
- **E6 thread ladder** (llama ncmoe32 -t 1/2/4/8/16/24): monotone in t;
  -t1 < 8 tok/s — the weak-CPU regime where the hybrid's CPU-independent
  20.8 (K=0) / 24.8 (K=16) wins outright; this is the registered crossover
  mechanism as a measured curve.

Readings pre-committed: any red here amends the RESULTS' claims BEFORE
publication, not after.
