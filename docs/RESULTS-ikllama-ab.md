# RESULTS — same-box ik_llama.cpp A/B (B3 stretch), Qwen3-235B-A22B-Instruct-2507

Grades `PREREG-ikllama-ab.md` (OTS `71c467fdad808aee`). Same-box pair landed
2026-07-20 00:58 CDT on RunPod A100-80G-PCIe `kfamw449nudi7g`; both arms one
box, evidence-complete teardown, 404-verified. Receipts:
`docs/mxfp4/receipts-train/` companion + pod evidence
`mx7-evidence/kfamw449nudi7g/ab/{receipt.json,ours.json,ik_bench.log,ours_redux.log}`.

## Result (one config per arm, same NVIDIA A100 80GB PCIe, link 26.74 GB/s)

| arm | quant | storage | decode tok/s (tg128) | VRAM |
|---|---|---|---|---|
| **ours** (stamped Phase-B harness, verbatim) | NF4 4-bit + fp32 absmax/64, **~4.5 bpw eff**, from the released bf16 | host-streamed, fused-kernel decode | **2.29** (K=0 pure-stream; K-dial 2.00–2.35 route-dependent) | 15.1 GB |
| **ik_llama.cpp** (author's 24 GB-VRAM recipe) | ubergarm **mix-IQ3_K, 3.4325 bpw** | 106.64 GiB, experts→CPU, `-fa -rtr -fmoe` | **3.17 ± 0.13** | — |

## Grades

| id | bar | got | verdict |
|---|---|---|---|
| ours-band | ∈ [2.23, 3.02] tok/s (transfer law, prereg) | **2.29** (K=0) | **GREEN — in band** |
| ik-record | first in-house same-box measurement (no band) | 3.17 ± 0.13 | recorded |
| pre-committed reading | "ik above ours on a fat-CPU box EXPECTED — CPU-compute + ~0.6 bpw smaller quant" | ik 3.17 > ours 2.29; ik 3.43 vs our 4.5 bpw | **HELD** |
| B3 supremacy ban | no throughput-leadership claim either way | honored (see framing) | **GREEN** |

**Our arm's transfer law held on a new box**: measured 2.29 tok/s at the
A100's 26.74 GB/s link sits in the prereg band. The K-dial is route-dependent
here (residency helps to 2.35 on high-hit prompts, costs to 2.00 on
low-hit) — consistent with the stamped ladder's regime map; K=0 is the clean
comparison point.

## Sanctioned framing (the only publishable form — B3/R7)

*"On one A100-80G box: ik_llama.cpp serves Qwen3-235B at 3.17 tok/s and our
NF4 host-streaming path at 2.29 tok/s. ik is faster here — it runs a ~1-bit-
smaller quant (IQ3_K 3.43 bpw vs our NF4 ~4.5) and computes its experts on
the CPU, which suits a fat-CPU host. Our differentiator is not inference
throughput: it is 4-bit NF4 fidelity from the exact released checkpoint bytes
and the byte-provenance receipt — neither of which a 3-bit repack provides."*

No "fastest / beats" claim; ik named as the baseline; quant + host caveats
inline. This is the same-box number that pre-empts the first hostile
launch-thread question (its intended purpose).

## Ops

~$8 total across two pods (a null H100 first attempt — three of MY dep/env
misses: `transformers`, `bitsandbytes`, and the ik-build nvcc PATH; all fixed
in the kit) + this A100. The A100 run itself salvaged the sunk 545 GB
download: arm 1 re-ran in place after arm 2 (serialized, no GPU contention)
via `arm1-redux.sh` + a watcher hold, so the same-box pair cost one pod's
download, not two. Operator cap $25; spend ~$8.
