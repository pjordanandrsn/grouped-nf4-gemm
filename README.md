# grouped-nf4-gemm — one-launch 4-bit GEMM over fused MoE expert stacks (NF4 + native MXFP4)

[![CI](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grouped-nf4-gemm)](https://pypi.org/project/grouped-nf4-gemm/)

A Triton kernel that runs the grouped expert GEMM **directly on
4-bit-packed weights**: one launch for all active experts, LUT decode to
fp32 in registers, blockwise scaling, fp32 accumulation, bf16 epilogue.
No per-expert dequantise-then-`bmm`, no bf16 weight materialisation.
Two codebooks: **NF4** on the bitsandbytes `gemm_4bit` layout, and
**MXFP4** (OCP e2m1 + e8m0) on a checkpoint's exact released bytes.

Beside the GEMM, the package carries what a 4-bit MoE serving path needs
around it: an **fp8 paged decode attention** (sliding windows, attention
sinks, custom scale, per-layer KV geometry), the **int4-b32** decode GEMV
and its calibrated packer, the **decode glue kernels** (fused RMSNorm,
residual-fold, rotary, router epilogue), and the host-streaming and NVMe
tiers for models that do not fit.

**Current position, one page:** [`docs/STATUS.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.24.0/docs/STATUS.md).
**Every number, with its evidence and tier:** [`docs/claims.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.24.0/docs/claims.json).

## See it on your own hardware first

```bash
pip install grouped-nf4-gemm bitsandbytes
python examples/dequant_tax.py          # ~1 min, one GPU, no model download
```

[`examples/dequant_tax.py`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.24.0/examples/dequant_tax.py)
times the dequantise-then-GEMM round trip against computing on the
packed bytes at three points on the M axis, prints a **self-pair** beside
every ratio (a ratio inside the instrument's own spread is not a
measurement), and says what the run does *not* show.

## Install

```bash
pip install grouped-nf4-gemm          # nf4gemm and gnf4 are aliases
```

Trusted publishing; every wheel carries a PEP 740 attestation. The fused
GEMM is CUDA-only (`triton>=3.4`, Linux); the reference decode and the
provenance hashing are pure torch and run anywhere.

## Which entry point? Pick by where the weights live

| the bytes are in… | call |
|---|---|
| **VRAM**, NF4-packed | `nf4_grouped.gemm_4bit_grouped(...)`; backward via `dgrad_4bit_grouped` |
| **VRAM**, native MXFP4 | `mxfp4_grouped.gemm_mxfp4_grouped(...)` |
| **host DRAM**, all rows pinned | `mxfp4_pipelined.Mxfp4PipelinedGptOss` |
| **NVMe**, too big for DRAM | `mxfp4_residency.Mxfp4NvmeResidency` over a baked arena |
| nowhere yet — you need to make an arena | `nvme_arena.bake_expert_tensors(...)` (relocates MXFP4) or `nvme_bake_nf4.bake_nf4` (re-quantises bf16) |
| a checkpoint you want to **verify**, not run | `verify_provenance` |

**Do not quantise-bake a checkpoint that is already MXFP4.** Relocation
keeps the bytes and hands packed nibbles to the kernel; re-quantising to
NF4 costs a dequant per read — measured ~4× slower per request on
DeepSeek-V4-Flash.

Training goes through `nf4_qlora` / `mxfp4_qlora`, which is what
[`experts4bit-qlora`](https://pypi.org/project/experts4bit-qlora/)
drives (`enable_fast()`, `enable_fast_train()`). This package makes one
expert-stack matmul cheap; e4b decides which bytes are where.

## What is measured

Tiers, used strictly: **confirmed** = pre-registered, OpenTimestamps-
stamped blind confirmatory run; **measured** = a run with a committed
receipt here; **measured-private** = a real run whose receipt lives in a
private audit tree, so you cannot check it from this repository.

| | result | tier |
|---|---|---|
| Fidelity, fused vs dequantise-to-bf16 | more accurate in every cell ever measured (fp32 accumulate) | confirmed |
| Decode, census MoE shapes vs the dequant path (sm_86) | 1.16–2.73× at median | confirmed |
| Energy, J/token | below baseline in 104 of 112 cells | confirmed |
| Real OLMoE QLoRA finetune, fused vs per-expert loop, real prose | 4.50× (4090), 4.75× (H100) | confirmed |
| vs Unsloth's own kernel, 4-bit-storage regime, decode | 1.70× (H100, their TMA live), 2.79× (4090) | confirmed |
| vs `torch._grouped_mm` on bf16, Qwen3-30B cell (RTX 5090) | 2.1–6.0×, on half the bytes | measured |
| Training backward in one launch, E=256 step | 403.7 → 26.5 ms | measured |
| Single-stream decode anchor, Qwen3-30B-A3B on the 5090 class | 7.37 ms/step ±4.2% (≈130–142 tok/s) | measured |
| Qwen3-235B-A22B from pinned host RAM on ≤16 GB VRAM | 4.3–4.4 tok/s, five pods; `t ≈ c_box + bytes/link` | confirmed |
| gpt-oss-120b served on its exact MXFP4 bytes | ppl 26.72 vs shipped reference 26.75; the NF4 requant tax deleted | confirmed |
| gpt-oss-120b QLoRA on native bytes | 9.82 GB peak; 144/144 hashes identical after training | confirmed |
| int4-b32 decode GEMV, dense M=1 (5090) | 1,044 GB/s; 6.9–7.2× over the NF4 GEMV | measured-private |

**Three limits, stated here rather than in a footnote:**

1. **Against a CUDA-graphed baseline the fused path loses at decode**
   (0.949× on a 4090, 0.858× on an H100). What graphing cannot touch is
   the memory-traffic win at training shape on bandwidth-limited cards
   (1.489× on the 4090; parity on the H100). The position is narrow on
   purpose: competitive at equal VRAM, wins when VRAM binds.
2. **Unsloth wins its own regime.** Against their bf16-resident kernel
   they run 2.6–5.3× faster at prefill on an H100.
3. **Known losers:** `top_k=1` cells are instance-unstable in both
   directions; shapes under ~5 M weight elements lose outright and are
   routed back to the dequant path.

And one about your benchmark: **random token ids understate this kernel
by ~1.6×.** Prose routes to 98.4% of experts, random ids to 87.5%, and
fewer hit experts means fewer iterations of the loop this replaces.
Benchmark on real text.

## The receipts

Six blind confirmatories (v1–v6); the first five did not fully pass as
registered, each results doc says what failed, and the sixth passed
clean. Pre-registrations, amendments, evidence JSONs and reducers are
committed; `.ots` files anchor the protocols. The Unsloth head-to-head
has its own stamped protocol. All under
[`kernel/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.24.0/kernel)
(`RESULTS-*.md`, `prereg_*.json`). The 235B flagship and the closed
prefetch programme are under
[`bench/phase3/flagship/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.24.0/bench/phase3/flagship).
The MXFP4 lane and Kimi K3 provenance chain are under
[`docs/mxfp4/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.24.0/docs/mxfp4)
and [`docs/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.24.0/docs).
What each of the 22 docs is, and whether it is current:
[`docs/INDEX.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.24.0/docs/INDEX.md).

## What was retired

Kept findable in `docs/STATUS.md` and as `retired` entries in
`docs/claims.json`: the "sm_120 parked" roadmap line (sm_120 has been the
primary serving target since 0.15.0); split-K on the decode GEMV
(refuted, ships dormant as the evidence); a fixed fraction-of-waterfall
as the offload law; the cold-engine "free floor" premise; expert
prefetch (closed, negative, four arcs). The "4.67× vs the grouped-bf16
execution class" number is superseded by the head-to-head — that backend
never ran Unsloth's own kernel.

## What is open

[#319](https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/319) the
f32 paged compute modes miss their reference on torch 2.8 / triton 3.4
(the fp8 modes, the sm_120 default, pass);
[#87](https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/87) int32
offset overflow at large `max(expert_ids)`; #73, #60, #58 arena/NVMe
efficiency; #71 pinned-row factor on cgroup v2. Every non-CUDA row is a
`port target` — `PROJECTIONS-multiarch.md` is stamped arithmetic that
invites refutation, and `docs/PORTABILITY.md` is the hazard register.

## Reproduce

[`REPRO.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.24.0/REPRO.md):
suite, benchmark and verdict reduction are each one command from a
frozen tree.

```
python -m pytest kernel/test_nf4_grouped.py -q
python -m pytest kernel/test_fp8_paged_attn.py -q -k "f8dot or pf8"   # the sm_120 serving modes
```

## Layout

`kernel/` — the kernels, packers, reference decodes, property suites,
pre-registrations and results · `bench/phase1..3/` — the confirmatory
harnesses, the census, the flagship · `bench/sm120-census/` — the 5090
census · `bench/cold-engine/` — the cold-tier research record ·
`docs/` — contracts, tolerance spec, MXFP4 and K3 receipts, `STATUS.md`,
`claims.json`, `INDEX.md` · `census/`, `roofline/` — shape census and
ceilings · `router_probe/` — the router-predictability probe.

## License & attribution

MIT. Portions developed with Claude Code as an AI assistant under the
author's direction and review — see
[ATTRIBUTION.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.24.0/ATTRIBUTION.md).
All claims are the author's responsibility.

## Contact

Cerin Amroth Research takes contract and pilot engagements on this work —
kernel ports, offload integration, sponsored research lanes with stamped
receipts. **jordan@cerinamroth.com**.
