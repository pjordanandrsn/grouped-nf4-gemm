# grouped-nf4-gemm — one-launch 4-bit GEMM over fused MoE expert stacks (NF4 + native MXFP4)

[![CI](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grouped-nf4-gemm)](https://pypi.org/project/grouped-nf4-gemm/)

**The problem in one line:** a 4-bit Mixture-of-Experts runs its experts
as a per-expert loop, and wherever that loop dequantises each active
expert to bf16 before its matmul (bitsandbytes before 0.50.0, cells
outside its packed 2-D inference forward, the conventional 4-bit
backward) the round trip is the bottleneck and the bf16 copies cost the
VRAM the quantisation was meant to save; this package computes the grouped
expert GEMM on the packed bytes themselves, in one launch for all routed
experts, and ships the decode kernels and host/NVMe primitives a 4-bit
MoE serving path needs around it. **Canonical package:** `grouped-nf4-gemm` on PyPI
(flat modules: `nf4_grouped`, `mxfp4_grouped`, `int4_b32`,
`fp8_paged_attn`, `nvme_arena`, `mxfp4_loader`, …); `nf4gemm`, `gnf4` and
`grouped-mxfp4-gemm` are lookup aliases. **Two repositories:** this one is the kernel side —
GEMMs, decode kernels, packers, references, host and NVMe primitives;
[`experts4bit-qlora`](https://github.com/pjordanandrsn/experts4bit-qlora)
is the consumer that loads and quantises models, trains adapters, places
bytes across tiers and serves, and installs this package through its
`[fast]` extra (the kernel version each consumer release requires is the
`compatibility` record in
[`docs/system-manifest.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/system-manifest.json),
not a number copied here). **Environment:** Linux, an NVIDIA GPU of sm_80 or newer
with Triton ≥ 3.4 for the kernels (sm_120 is the primary serving target;
Python 3.11 is what CI tests); the pack references, `dequant_ref`, the
relocation arena bake and the provenance hashing are pure torch (the NF4
quantise-bake, `nvme_bake_nf4.bake_nf4`, needs bitsandbytes and CUDA
unless a `quantize_fn` is injected). **The material
limitation:** the fused path is not faster everywhere — at small shapes
and against a CUDA-graphed per-expert loop at some decode shapes it loses,
and the register says so. Machine-readable capabilities and evidence:
[`docs/capabilities.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/capabilities.json)
and [`docs/claims.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/claims.json).

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

**Current position, one page:** [`docs/STATUS.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.30.1/docs/STATUS.md).
**Every number, with its evidence and tier:** [`docs/claims.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.30.1/docs/claims.json).

## Use this when

- Dequantise-then-GEMM is the expert bottleneck: you want one launch over
  the NF4-packed expert stacks (bitsandbytes `gemm_4bit` layout) with
  fp32 accumulation and no bf16 weight materialisation.
- Your experts are native MXFP4 (gpt-oss, DeepSeek-V4, Kimi lineage) and
  you want to compute on the released bytes (e2m1 blocks + e8m0 scales)
  without re-quantising them.
- You need a decode-time INT4 GEMV with 32-wide scales and a calibrated
  (GPTQ-style) packer for expert or attention projections.
- You need an fp8 paged decode attention (sliding windows, attention
  sinks, custom scale, per-layer KV geometry) and the fused decode glue
  (RMSNorm, residual fold, rotary, router epilogue, activation, combine).
- The experts do not fit VRAM, or host RAM: you want the pinned-DRAM tier
  and the NVMe arena bake, reader and cold tier.
- You must prove the bytes a kernel serves are the checkpoint's released
  bytes, unchanged (sha256 provenance of safetensors tensor ranges).
- You work on CUDA/Triton kernels for quantised MoE serving and want the
  pure-torch references every kernel is asserted against.

## Do not use this when

- The model is dense (no experts): cuBLAS, bitsandbytes' own 4-bit
  matmul or torch are the right tools; nothing here helps a single
  `nn.Linear`. Since bitsandbytes 0.50.0 (upstream #1949, merged
  2026-05-21) its supported ordinary 2-D inference cells compute directly
  from the packed weights (`torch.ops.bitsandbytes.gemm_4bit`); what
  upstream has no contract for is the grouped routed-MoE GEMM, which is
  this package.
- The model already fits in bf16 with headroom, or the shapes are small:
  the grouped kernel loses below its routing threshold and to a
  CUDA-graphed per-expert loop at some decode shapes
  (`gnf4.kernel.graphed-baseline-decode-loses` in the claims register).
- You expect a serving engine: this package is kernels and primitives,
  driven by `experts4bit-qlora`; it is not a vLLM replacement.
- You need Windows or macOS for the kernels: the Triton kernels need a
  CUDA GPU and Triton is Linux-only. The pure-torch surface (pack
  references, `dequant_ref`, provenance, the relocation arena bake/verify)
  imports and runs without triton via `_triton_shim`, but macOS and
  Windows are not exercised by CI. ROCm/XPU are port targets, not
  supported.
- Your expert tensors are not in a layout listed in
  [`docs/KERNEL_CONTRACT.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/KERNEL_CONTRACT.md) —
  nothing falls back silently: `nf4_grouped.gemm_4bit_grouped` and
  `dgrad_4bit_grouped` refuse CPU tensors with an error that names
  `dequant_ref`; `gemm_mxfp4_grouped` and the `int4_b32` kernels carry no
  device guard and fail inside the Triton launch, and the `fp8_kv` appends
  and the paged attention refuse with a CUDA+Triton message that names no
  reference.

## Start here

| | |
|---|---|
| [`docs/SOLUTIONS.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/SOLUTIONS.md) | one page per problem: symptoms, cause, install, smallest example, verification, limits |
| [`docs/capabilities.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/capabilities.json) | the machine-readable capability contract (entry points, layouts, environments, limitations, claim IDs) |
| [`docs/STATUS.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/STATUS.md) | the current position — measured, retired, open |
| [`docs/claims.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/claims.json) | every number with its evidence and status |
| [`docs/INDEX.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/INDEX.md) | what each document is and whether it is current |
| [`experts4bit-qlora`](https://github.com/pjordanandrsn/experts4bit-qlora) | the consumer package (`pip install "experts4bit-qlora[fast]"` installs this one) |
| [PyPI: grouped-nf4-gemm](https://pypi.org/project/grouped-nf4-gemm/) | the canonical distribution |
| [`llms.txt`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/llms.txt) · [`AGENTS.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/AGENTS.md) | orientation for language models and coding agents |
| The routing page for this project on cerinamroth.com (problem-first index, status, compatibility) | [https://cerinamroth.com/ml/grouped-nf4-gemm/](https://cerinamroth.com/ml/grouped-nf4-gemm/) |

## See it on your own hardware first

```bash
pip install grouped-nf4-gemm bitsandbytes
python examples/dequant_tax.py          # ~1 min, one GPU, no model download
```

[`examples/dequant_tax.py`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.30.1/examples/dequant_tax.py)
times the dequantise-then-GEMM round trip against computing on the
packed bytes at three points on the M axis, prints a **self-pair** beside
every ratio (a ratio inside the instrument's own spread is not a
measurement), and says what the run does *not* show.

## Install

```bash
pip install grouped-nf4-gemm          # nf4gemm, gnf4 and grouped-mxfp4-gemm are lookup aliases; install and cite grouped-nf4-gemm
```

Trusted publishing; every wheel carries a PEP 740 attestation. The fused
GEMM is CUDA-only (`triton>=3.4`, Linux); the reference decode and the
provenance hashing are pure torch and run anywhere.

## Try it on CPU right now

No GPU needed for the pack/decode/provenance surface — the fused GEMM is
CUDA-only, but the reference decode and the provenance hashing are pure torch.

Triton is declared `triton>=3.4; platform_system == 'Linux'`, so it is absent
on macOS and Windows; there the imports below resolve through `_triton_shim`,
which binds a stand-in so the pure-torch surface (pack references,
`dequant_ref`, provenance, the relocation arena bake/verify) imports and runs
and a kernel launch raises naming the CPU alternative. The Triton kernels
need a CUDA GPU. These three blocks are extracted and executed by CI
(`test_readme_cpu_block.py`) on Linux only, so they cannot drift from the
API; macOS and Windows are not exercised by CI.

<!-- CPU-QUICKSTART-START -->
**1. NF4 round-trip** — pack a weight, decode it back, check the error:

```python
import torch
from nf4_pack_ref import quantize_pack_nf4
from nf4_grouped import dequant_ref

w = torch.randn(256, 512)                      # a per-expert weight [N, K]
packed, absmax = quantize_pack_nf4(w)          # [256, 256] uint8, [256, 8] fp32
wq = dequant_ref(packed, absmax, 256, 512)     # decode back to [N, K]
print("nf4 rel-err:", round(((wq - w).norm() / w.norm()).item(), 3))     # ~0.09
print("nf4 re-pack idempotent:", torch.equal(quantize_pack_nf4(wq)[0], packed))  # True
```

**2. MXFP4 round-trip** — the gpt-oss expert format, same shape story:

```python
import torch
from mxfp4_pack_ref import quantize_pack_mxfp4, dequant_mxfp4

w = torch.randn(128, 256)                      # [.., K], K a multiple of 32
blocks, scales = quantize_pack_mxfp4(w)        # [128, 8, 16] u8, [128, 8] u8 (e8m0)
wq = dequant_mxfp4(blocks, scales)             # [128, 256]
print("mxfp4 rel-err:", round(((wq - w).norm() / w.norm()).item(), 3))   # ~0.12
```

**3. Provenance in four lines** — hash on-disk bytes, catch a tampered one:

```python
import torch, json, struct, tempfile, os
from mxfp4_loader import file_tensor_sha256, tensor_sha256

t = torch.arange(64, dtype=torch.uint8)        # stand-in for an expert's packed bytes
hdr = json.dumps({"w": {"dtype": "U8", "shape": [64], "data_offsets": [0, 64]}}).encode()
path = tempfile.mktemp(suffix=".safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hdr))); f.write(hdr); f.write(t.numpy().tobytes())
print("prov bytes match:", file_tensor_sha256(path, "w") == tensor_sha256(t))    # True
b = bytearray(open(path, "rb").read()); b[-1] ^= 0xFF; open(path, "wb").write(bytes(b))
print("prov tamper detected:", file_tensor_sha256(path, "w") != tensor_sha256(t))  # True
os.remove(path)
```

That's the same instrument the 144/144 training receipt used.
<!-- CPU-QUICKSTART-END -->

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
NF4 costs a dequant per read and breaks provenance; no registered claim
carries a figure for that cost, so none is quoted here.

Training goes through `nf4_qlora` / `mxfp4_qlora`, which is what
[`experts4bit-qlora`](https://pypi.org/project/experts4bit-qlora/)
drives (`enable_fast()`, `enable_fast_train()`). This package makes one
expert-stack matmul cheap; e4b decides which bytes are where.

## What is measured

Tiers, used strictly: **confirmed** = pre-registered, OpenTimestamps-
stamped blind confirmatory run; **measured** = a run with a committed
receipt here; **measured-private** = a real run whose receipt lives in a
private audit tree, so you cannot check it from this repository.

| | result | tier | claim ID in `docs/claims.json` |
|---|---|---|---|
| Fidelity, fused vs the dequantise-to-bf16-then-GEMM comparator | not less accurate in any registered confirmatory cell (fp32 accumulate) | confirmed | `gnf4.kernel.fused-more-accurate-than-dequant-bf16` |
| Decode, census MoE shapes vs the dequant path (sm_86) | 1.16–2.73× at median | confirmed | `gnf4.kernel.decode-speed-census` |
| Energy, J/token | below baseline in 104 of 112 cells | confirmed | `gnf4.kernel.energy-104-of-112` |
| Real OLMoE QLoRA finetune, fused vs per-expert loop, real prose | 4.50× (4090), 4.75× (H100) | confirmed | `gnf4.kernel.e2e-training-real-prose` |
| vs Unsloth's own kernel, 4-bit-storage regime, decode | 1.70× (H100, their TMA live), 2.79× (4090) | confirmed | `gnf4.kernel.h2h-unsloth` |
| vs `torch._grouped_mm` on bf16, Qwen3-30B cell (RTX 5090) | 2.1–6.0×, on half the bytes | measured | `gnf4.kernel.sm120-census-vs-grouped-mm` |
| Training backward in one launch, E=256 step | 403.7 → 26.5 ms | measured | `gnf4.kernel.dgrad` |
| Single-stream decode anchor, Qwen3-30B-A3B on the 5090 class | 7.37 ms/step ±4.2% (≈130–142 tok/s) | measured | `gnf4.serve.decode-anchor-5090` |
| Qwen3-235B-A22B from pinned host RAM on ≤16 GB VRAM | 4.3–4.4 tok/s, five pods; `t ≈ c_box + bytes/link` | confirmed | `gnf4.flagship.235b-phaseB` |
| gpt-oss-120b served on its exact MXFP4 bytes | ppl 26.72 vs shipped reference 26.75; the NF4 requant tax deleted | confirmed | `gnf4.mxfp4.serve-tax-deleted` |
| gpt-oss-120b QLoRA on native bytes | 9.82 GB peak; 144/144 hashes identical after training | confirmed | `gnf4.mxfp4.train-9.82gb` |
| int4-b32 decode GEMV, dense M=1 (5090) | 1,044 GB/s; 6.9–7.2× over the NF4 GEMV | measured-private | `gnf4.serve.int4-b32-gemv` |

The tier column is the claim's `status` in the register; the ID is the entry
to check before quoting a row, and a row whose entry is later retired or
superseded is no longer current whatever this table says.

**Three limits, stated here rather than in a footnote:**

1. **Against a CUDA-graphed baseline the fused path loses at decode**
   (0.949× on a 4090, 0.858× on an H100). What graphing cannot touch is
   the memory-traffic win at training shape on bandwidth-limited cards
   (1.489× on the 4090; parity on the H100). The position is narrow on
   purpose: competitive at equal VRAM, wins when VRAM binds
   (`gnf4.kernel.graphed-baseline-decode-loses`).
2. **Unsloth wins its own regime.** Against their bf16-resident kernel
   they run 2.6–5.3× faster at prefill on an H100 (`gnf4.kernel.h2h-unsloth`).
3. **Known losers:** `top_k=1` cells are instance-unstable in both
   directions; shapes under ~5 M weight elements lose outright and are
   routed back to the dequant path (`gnf4.kernel.decode-speed-census`).

And one about your benchmark: **random token ids understate this kernel
by ~1.6×.** Prose routes to 98.4% of experts, random ids to 87.5%, and
fewer hit experts means fewer iterations of the loop this replaces.
Benchmark on real text (`gnf4.kernel.e2e-training-real-prose`).

*Dated note (2026-09-04).* Every "dequant path" comparator in this table
is this repository's own per-expert loop as each receipt ran it:
bitsandbytes `dequantize_4bit` per active expert, then a bf16 matmul.
bitsandbytes 0.50.0 added a packed 4-bit CUDA inference forward for
supported ordinary 2-D cells; no receipt here times that path, and it has
no grouped routed-MoE contract. The ratios stay what their receipts
measured. The version-aware statement is on the
[NF4 solution page](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/nf4-grouped-gemm-without-bf16-materialization.md).

## The receipts

Six blind confirmatories (v1–v6); the first five did not fully pass as
registered, each results doc says what failed, and the sixth passed
clean. Pre-registrations, amendments, evidence JSONs and reducers are
committed; `.ots` files anchor the protocols. The Unsloth head-to-head
has its own stamped protocol. All under
[`kernel/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.30.1/kernel)
(`RESULTS-*.md`, `prereg_*.json`). The 235B flagship and the closed
prefetch programme are under
[`bench/phase3/flagship/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.30.1/bench/phase3/flagship).
The MXFP4 lane and Kimi K3 provenance chain are under
[`docs/mxfp4/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.30.1/docs/mxfp4)
and [`docs/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.30.1/docs).
What each document is for, and whether it is current:
[`docs/INDEX.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.30.1/docs/INDEX.md).

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
(the fp8 modes, the sm_89+ default, pass; the f32 path is the sm_80–sm_88
default and every explicit f32 request, and `docs/capabilities.json`
carries it as its own `unsupported` entry, `fp8-paged-attention-f32-compute`,
beside the supported `fp8-paged-attention-fp8-compute`, until this closes;
claim `gnf4.open.f32-compute-modes-triton34`);
#73, #60, #58 arena/NVMe
efficiency; #71 pinned-row factor, conservative on cgroup v1 (v2 unmeasured).
[#87](https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/87) (int32
offset overflow at large `max(expert_ids)`) is closed by observation in
every carrier (PR #342; boundary test `kernel/test_expert_offset_boundary.py`;
GPU run on an RTX 5090 2026-09-05: 10 passed; claim
`gnf4.kernel.expert-offset-boundary.5090.2026-09-05`). Every non-CUDA row is a
`port target` — `PROJECTIONS-multiarch.md` is stamped arithmetic that
invites refutation, and `docs/PORTABILITY.md` is the hazard register.

## Reproduce

[`REPRO.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.30.1/REPRO.md):
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
[ATTRIBUTION.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.30.1/ATTRIBUTION.md).
All claims are the author's responsibility.

## Contact

Cerin Amroth Research takes contract and pilot engagements on this work —
kernel ports, offload integration, sponsored research lanes with stamped
receipts. **jordan@cerinamroth.com**.
