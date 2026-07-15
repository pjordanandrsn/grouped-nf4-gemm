# Flagship 671B-class scaling point — the offload pipeline holds the PCIe waterfall at 3× the pinned store (S2/S3 PASS, two instances); the REGISTERED run's S4 drift guard FAILED on host contention, disclosed, with a post-hoc clean-host run supporting the attribution

**DRAFT — held for owner review; not stamped, not committed.**

**2026-07-15 · Protocol:** `kernel/prereg_flagship_scale671.json` (OTS pre-data,
`800e4f4`) · **Hosts:** two fresh RunPod SECURE 2× H100 80GB HBM3 pods (each
rented for its **468 GiB cgroup RAM allotment** — measured 502999998464 B, the
2×234 GiB the pin-probe predicted; GPU 0 ran, GPU 1 idle).

**The stamped protocol registers ONE host — that is instance A, and A's verdict
is the registered result.** Instance B is a **post-hoc run** added at the owner's
direction after A's drift guard tripped; it is reported as **supporting evidence**
for the contention attribution, NOT as part of the registered protocol. Per the
no-tune clause (below), a re-run does not convert a registered failure to a pass —
so the headline S4 verdict is FAIL-as-registered, with B corroborating *why*.

## What was registered

Phase A confirmed the waterfall at Qwen3-235B geometry (128 GB store,
8.0 GB/token). This point asks: **does the fused offload pipeline still saturate
the PCIe wire when the host-RAM expert store grows 3×?** Geometry:
**DeepSeek-V3-class — 61L × 256E × k8, hidden 7168, expert intermediate 2048** →
**387 GB NF4 store, 12.09 GB/token** (predicted 12.1). Synthetic weights by design
(isolates the pipeline; a ~1.3 TB real checkpoint is out of scope — the claim is
"pipeline scales," not "ran real DeepSeek-V3"). GQA attention stands in for MLA as
a compute load, declared.

## Result — two instances, four fused arms

| instance | arm | tok/s | link GB/s | % of waterfall | VRAM | drift |
|---|---|---|---|---|---|---|
| A | fused | 4.408 | 55.5 | 96.0% | 30.9 GB | — |
| A | fused2 | 4.062 | 55.5 | 88.4% | 30.9 GB | 7.85% ✗ |
| B | fused | 4.416 | 46.9 | 113.7% | 30.9 GB | — |
| B | fused2 | 4.414 | 53.5 | 99.7% | 30.9 GB | **0.05% ✓** |

| criterion | bar | A | B |
|---|---|---|---|
| S1 link | report | 55.5 GB/s | 46.9 / 53.5 GB/s |
| **S2 scaling** | ≥0.85× wf, both arms | **PASS** (0.96/0.88) | **PASS** (1.14/1.00) |
| **S3 vram** | ≤45 GB | **PASS** (30.9) | **PASS** (30.9) |
| **S4 drift** | arms within 5% | **FAIL** (7.85% — contention) | **PASS** (0.05%) |

## The headline: the pipeline is model-size-agnostic to the wire

**A ~0.69-trillion-expert-parameter MoE decodes at ~4.41 tok/s — tracking the
PCIe physical limit — on 30.9 GB of VRAM**, streaming a 387 GB NF4 expert store
from host RAM. The waterfall model validated at 128 GB / 8.0 GB/token (Phase A)
holds at **387 GB / 12.09 GB/token** — 3× the store, same physics. Across the
program the fused offload pipeline now saturates the wire over a **3× model-store
range** (128 → 387 GB) and a **44 → 55 GB/s link range** (Phase A 44 GB/s 102%;
B5 pods 55 GB/s 93–94%; here 47–55 GB/s 96–114%), VRAM staying flat at
two-expert-staging-buffers plus resident attention. The claim moves from "235B on
≤16 GB VRAM" to **"the offload pipeline is model-size-agnostic up to DeepSeek-V3
scale, bounded only by host RAM."**

**Throughput is the stable quantity; the waterfall fraction is not.** The four
fused arms cluster at **4.408 / 4.416 / 4.414 tok/s** (the three uncontended ones
within 0.2%), i.e. a sustained ~53 GB/s effective H2D. The *fraction*-of-waterfall
spreads 96–114% only because the on-box link microbench itself sampled 46.9–55.5
GB/s run-to-run — the documented microbench-conservatism / link-variance effect
(instance B's fused arm read a low 46.9 GB/s link, so the stable 4.416 tok/s came
out at 113.7% of that arm's own conservative ceiling). The kernel's behavior is
constant; the denominator wobbles.

VRAM (honest): 30.9 GB here vs 13.6 GB at 235B — entirely the **larger resident
attention geometry** (hidden 7168, 128 heads, longer KV), not the MoE path, which
stays the two expert-sized staging buffers. Under the registered 45 GB cap;
reported, not hidden. Still a 0.69T-class MoE on under a third of one H100's HBM.

## S4 drift — FAILED as registered; attributed to contention; not re-graded

Instance A's two arms differed 7.85%, tripping the 5% guard — **S4 FAILS on the
registered host.** The per-token traces pin the cause: **arm 1 was dead flat
(227 ms on all 64 tokens, median = p10 = p90); arm 2 shared that same 227 ms
floor (p10 = 227) but carried a noisy opening tail** (p90 = 286; first four tokens
613/595/646/356 ms before settling) — background contention on the shared 2-GPU
host during arm 2's opening, not the kernel.

**The no-tune-clause tension, stated plainly:** the clause is "no re-runs until
green," and instance B is a re-run of a criterion that had failed. So B is NOT
credited as satisfying S4 — the registered verdict remains **S4 FAIL**. What B
legitimately provides is *evidence for the attribution*: on a fresh, uncontended
pod (same frozen commit, same protocol) both arms are flat (p10 226 / p90
227–228, drift **0.05%**), which is what "the drift was the host, not the kernel"
predicts. Both instances' receipts are kept in full; nothing was swapped for a
nicer number. The honest one-liner: *S4 failed on the registered host; a clean-host
run corroborates that the failure was environmental, but the registered failure
stands.*

## Standing state

**S2 (scaling) and S3 (VRAM) PASS on both instances; S4 FAILS as registered
(contention), with instance B as disclosed supporting evidence — not a
re-grade.** The headline scaling claim rests on S2, which is unambiguous on both
hosts: the pipeline holds the PCIe waterfall at 3× the pinned store. The clean
1.0T point remains unmeasured (parked: RunPod cgroup caps at 234 GiB/GPU; the g4
bare-metal node that could hold 1 TB was declined at $47.98/hr).

## Evidence / teardown

`scale671/instanceA/` + `scale671/instanceB/`:
`s671_{fused,fused2,smoke}.{json,log}`, `S671_STATE`, `SHA256SUMS`. Pods
`of46tpfjygtdpv` + `vf9uouc367vxvn` both DELETE → 404-verified, 0 pods. Spend
≈ $18 total (two 2×H100 runs, ~1.6 h each incl. the per-arm 387 GB store builds).
