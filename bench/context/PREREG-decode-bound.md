
## Outcome — both confirmed, and together they name the target

**(A) Decode isolation** (A40, 8 experts of [3072, 4096], 56.62 MB):

| kernel | ms | GB/s | % of gemm |
|---|---:|---:|---:|
| read (bytes only) | 0.0826 | 685.5 | 14.5% |
| decode (unpack + LUT + scale) | 0.1163 | **487.0** | 20.4% |
| gemm (shipped grouped) | 0.5710 | 99.2 | 100% |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| A1a decode ≥ 50% of gemm | ≥50% | **20.4%** | **FALSIFIED** |
| A1b read ≤ 15% of gemm | ≤15% | **14.5%** | **CONFIRMED** |
| A1c decode vs achievable | reported | **85.3%** | — |

**(B) Re-decomposition on bf16 KV** (235B, 94 layers, 48-token context):

| category | s/token | % of wall | was (nf4_host, #36) |
|---|---:|---:|---:|
| **experts** | 0.6613 | **71.3%** | 46.4% |
| attention | 0.1473 | **15.9%** | 44.5% |
| norms | 0.0828 | 8.9% | 4.8% |
| router | 0.0322 | 3.5% | 0.9% |
| lm_head | 0.0008 | 0.1% | 0.1% |
| GPU busy | 0.9245 | **99.7%** | 96.7% |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| B1a attention < 20% | <20% | **15.9%** | **CONFIRMED** |
| B1b experts > 60% | >60% | **71.3%** | **CONFIRMED** |
| B1c GPU busy > 90% | >90% | **99.7%** | **CONFIRMED** |

### Together

```
experts are 71.3% of the step
79.6% of the GEMM is neither reading nor decoding
-> 56.8% of the ENTIRE step is grouped-GEMM tiling and dispatch
```

At bs=1 the expert GEMM is an **M=1 GEMV** padded up to `tl.dot`'s **M≥16**
minimum, so ~94% of every tensor-core tile is padding. It is not memory-bound
(99 of 487 GB/s reachable on the same bytes), not compute-bound (0.95% of peak),
and not decode-bound (20.4%).

**If the GEMM reached its own decode floor: 0.9691 → 0.4428 s/token = 2.19×** on
the whole step. That is the largest identified prize left, and unlike CUDA graphs
(3.3% ceiling in #36, now 0.3%) it is not a scheduling problem.

**#36's attention figure is superseded.** Its 44.5% was inflated by the
`nf4_host` KV setting #37 showed to be wrong; corrected, attention is **15.9%**
and was never the target. GPU busy rises to **99.7%**, so the CUDA-graphs verdict
holds *more* strongly, not less.

**Pre-committed decisions fire:** A1a falsified → the target is the GEMM's tiling
and dispatch, not the decode primitive. B1a confirmed → #37's correction stands.
