# RESULTS — KDA on SM89 (PREREG `71181a3e`)

**B1 PASS · B2 PASS · B3 PASS — the consumer story stands.**

Run on the M1 pod's **L40S** (SM89, cc 8.9, torch 2.11+cu128, fla-core
current), K3 dims exactly (H=96, d_k=d_v=128, hidden 7168, bf16,
lower_bound −5), kernel + projection GEMMs = per-layer figure, outputs finite.

| bar | measured | limit | margin |
|---|---|---|---|
| B1 decode /layer | **1.212 ms** (kernel 0.182 + proj 1.030) | ≤ 3.0 ms | 2.5× |
| B2 prefill @32K /layer | **0.210 s** (kernel 0.077 + proj 0.133) | ≤ 0.63 s | 3.0× |
| B3 verify T=8 vs T=1 | **0.1855 vs 0.1824 ms (1.02×)** | ≤ 2× | near-constant |

Consequences: 69 KDA layers ≈ **84 ms/token** of decode attention — ~6% of
the 1.4 s consumer byte floor (B1's 15% budget, well inside). KDA prefill
hides ~3× under the expert stream at 32K chunks. Verification width is
effectively FREE at the kernel level — stronger than the sub-linearity the
premise needed, and stronger than the bar registered.

**Fixture deviation, disclosed:** the prereg named a 4090; two SECURE 4090s
wedged (RUNNING, empty runtime, no ports; both deleted + 404-verified) and
the bench ran on the L40S instead — which is the SLOWER SM89 part on every
relevant axis (864 vs 1008 GB/s, lower clocks). Bars passed here therefore
bound the 4090 from below (a-fortiori, conservative direction). Per the
prereg's own rule this deviation cannot flip a verdict; a literal-4090 run is
warranted only if a future result lands near-bar, and none of these are.

Evidence: `kda_sm89.json` inside `m1-evidence.tgz` (`a6cb37a6…`); bench
script `kda_sm89_bench.py` archived in the same bundle.
