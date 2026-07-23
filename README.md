# grouped-nf4-gemm — single-launch 4-bit codebook GEMM over fused MoE expert stacks (NF4 + native MXFP4)

[![CI](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grouped-nf4-gemm)](https://pypi.org/project/grouped-nf4-gemm/)

A Triton kernel that runs the grouped expert GEMM **directly on 4-bit-packed
weights** — one launch for all active experts, LUT decode to fp32 in
registers, blockwise fp32 scaling, fp32 accumulation, bf16 epilogue. No
per-expert dequantize-then-`bmm` round trip, no bf16 weight materialization.
Both 16-entry codebooks ship: **NF4** on the canonical bitsandbytes
`gemm_4bit` layout ([#1949](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1949))
— `[E, N, K/2]` uint8 + fp32 blockwise absmax — and **MXFP4 (OCP e2m1 +
e8m0 per-32 scales), computing on a checkpoint's exact released bytes**
(see the native-byte lane below).

**Why:** for frozen 4-bit MoE experts, the standard path pays to decode the
weights into bf16 and then reads them again — at batch-1 decode that round
trip (plus ~3 kernel launches per active expert) dominates. Fusing the decode
into the GEMM deletes it. The measured side effect worth stating plainly:
**fp32 accumulation makes the fused path *more accurate* than the
materialize-to-bf16 baseline — the fused path has never measured less
accurate than the baseline.**

## Install

```bash
pip install grouped-nf4-gemm    # nf4_grouped + nf4_pack_ref + host_gather,
                                # plus the mxfp4_* lane + moonshot_gather + verify_provenance
```

`pip install nf4gemm` and `pip install gnf4` are equivalent aliases.
Published via trusted publishing; every wheel carries a PEP 740 attestation.

## Try it on CPU right now

No GPU needed for the pack/decode/provenance surface — the fused GEMM is
CUDA-only, but the reference decode and the provenance hashing are pure torch.
These three blocks are extracted and executed by CI (`test_readme_cpu_block.py`),
so they cannot drift from the API.

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

## The MXFP4 native-byte lane (0.2.0)

gpt-oss ships its experts as **MXFP4 blocks** — e2m1 is a 16-entry codebook,
so the same in-register-decode mainloop serves it by table swap. The lane's
point is **provenance**: compute on the checkpoint's *exact released bytes*
(no requantization), which makes the served weights verifiable and deletes
the conversion tax. Stamped, receipts in `docs/mxfp4/`:

- **Serve** ([`RESULTS-mxfp4-serve.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/docs/mxfp4/RESULTS-mxfp4-serve.md)):
  fused-native exact-chunk ppl **26.72** on gpt-oss-120b = the
  shipped-precision reference (26.75) — the measured **+9.4% ppl / KL 0.066
  NF4-requant tax is deleted**; per-shard provenance
  `sha256(loaded bytes) == sha256(file range)` 4/4 on real 120b shards.
- **Train** ([`RESULTS-mxfp4-train.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/docs/mxfp4/RESULTS-mxfp4-train.md)):
  **gpt-oss-120b QLoRA at 9.82 GB peak VRAM on native bytes**
  (recompute-in-backward + per-expert LoRA), step-0 ppl inside the stamped
  serve band, **144/144** `sha256(file) == sha256(loaded) == sha256(post-train)`
  — the frozen base is byte-identical after training.
- **Verify it yourself**: `verify_provenance` re-hashes a checkpoint's expert
  byte ranges against a served/trained arena from the artifact alone
  (96/96 on the real shipped 20b bytes).
<!-- update same-day when shards land: format verdict (MXFP4 vs INT4) + first measurements or their absence -->
- **Kimi K3** (native MXFP4 via QAT, weights announced for July 27, 2026):
  `moonshot_gather` — the DeepSeek-lineage per-expert → fused gather loader —
  is included and **K2-verified** (live K2 index, 9/9 gates). Every
  K3-specific number stays unclaimed until the weights drop and the oracle
  re-adjudicates against K3's own dequant reference (the per-model STOP gate).

The engine composes with the hot/cold serving work in
[experts4bit-qlora](https://pypi.org/project/experts4bit-qlora/): hot sets
are format-independent, and the pipelined-residency integration rail is the
next e4b increment.

**Using it inside a model?** [experts4bit-qlora](https://pypi.org/project/experts4bit-qlora/)
ships this kernel as its optional inference path:
`pip install "experts4bit-qlora[fast]"` then `enable_fast(model)` routes the
frozen NF4 expert projections through `gemm_4bit_grouped` (measured 3.65× over
its reference per-expert loop at bs=1 decode, OLMoE geometry, A2000) with
automatic fallback for training and ineligible modules.

```python
from nf4_grouped import gemm_4bit_grouped, dequant_ref
```

## The claim (blind-confirmed, receipts in-repo)

Everything below is from **pre-registered, OpenTimestamps-stamped blind
confirmatory runs** (protocol + pass/fail criteria stamped before data; two
devices; n=3 fresh-process reps; worst/median-rep reduction; failures
reported at full volume). On sm_86 at batch-1 decode, versus the
dequantize-then-matmul baseline on the same stacks:

- **Fidelity:** property suite green on every device, every run (35 → 44
  tests as the kernel grew); fused output error **below the baseline's in
  every cell ever measured** (fp32 accumulate).
- **Energy:** fused J/token **below the baseline in 104 of 112
  confirmatory-grade cells across v1–v3**. Six of the eight misses are the
  `top_k=1`/tiny class (named below); the other two are parity-margin
  readings (1.005, 1.010) on a single instance. On bandwidth-bound cells the
  energy win has never failed to replicate.
- **Speed:** census MoE shapes (OLMoE, Qwen3-30B, Gemma-4, GPT-OSS-120B,
  gate_up + down) run **1.16–2.73× at median** (one census cell —
  gpt-oss `down`, 2880×2880 — is instance-sensitive: 0.7–2.0× across five
  instances). Fresh off-census shapes with `top_k ≥ 6` (DeepSeek-V3,
  granite-3.1, Qwen3-Next) run **1.0–1.8× at median**; `k=2`-large shapes
  (Grok-1, Mixtral-8x22B) 1.0–1.24×, never slower.
- **Versus the other execution classes** (same-run census on the v6
  kernel, [receipts](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase1/results/comparators_v6/RESULTS-comparators-v6.md)):
  the grouped-bf16-GEMM class that unsloth's MoE backend rides
  (`grouped_gemm.ops.gmm`, dequant inside the timed path as 4-bit storage
  requires) loses to the fused kernel on **every census cell — decode
  median 4.67×, prefill median 3.02×** (their kernel targets bf16-resident
  training, a job it is excellent at; this comparison is the 4-bit-storage
  regime, which both must serve when weights are quantized). Axolotl/PEFT
  QLoRA forwards run bitsandbytes `Linear4bit` — see the flagship bnb
  baseline. GPTQ-Marlin is fidelity-excellent but per-expert
  (launch-storm at MoE decode) and format-incompatible with NF4 checkpoints.
- **Known losers:** `top_k=1` cells are **instance-unstable in both
  directions** (Scout `down` measured 0.47–1.12 across six contexts on
  identical code — split-K helps paired but can't stabilize the class), and
  **tiny shapes (≲5 M weight elements) lose outright** (0.24–0.35× speed,
  4–7× energy). v4 adds a dispatch floor that routes tiny cells back to the
  dequant path.
- **Prefill** (compute-bound M): the v6 register-LUT mainloop rewrite
  (blind-CONFIRMED) runs **1.39–1.54× the prior mainloop on every census
  prefill cell**; against the dequant path the census reads 1.14–2.78× with
  all three large gate_ups above 1.15 — gate_up is no longer a loser class.
  One caveat carried at full volume: the dequant *baseline* itself swings
  ~25% between cloud instances (the fused kernel holds within 0.2 ms), and
  OLMoE gate_up (the smallest-expert shape) remains below parity at ~0.6×.

Six blind confirmatories have run; the first five **did not fully pass as
registered**, each results doc says exactly what failed and why, and the
sixth passed clean:
[v1](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/RESULTS-gate2-confirmatory.md) (caught the original per-shape
config table overfitting its census), [v2](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/RESULTS-v2-confirmatory.md)
(validated the replacement single-constant config on 64-SM parts and the
off-census `k≥6` wins), [v3](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/RESULTS-v3-confirmatory.md) (found the
v2-era SM-conditional premise was measurement noise, quantified the
`top_k=1` and tiny-shape loss classes, and established the methodology rule
that latency-bound cells only support paired claims),
[v4](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/RESULTS-v4-confirmatory.md) (dispatch floor + split-K work floor
+ prefill config; caught its own dispatch-point regression),
[v5](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/RESULTS-v5-confirmatory.md) (the load-time dispatch fix, clean on
the A5000 11/11 with energy 8/8 on both devices; one contended-A2000 noise
cell kept it from a full pass — the dispatch line is closed),
[v6](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/RESULTS-v6-confirmatory.md) (**CONFIRMED**, all five criteria:
the register-LUT M-tile mainloop, adjudicated on the instance-robust paired
rewrite ratio after the dress rehearsal exposed the dequant baseline's
host lottery). The preregs,
amendments, evidence JSONs, sweeps, and mechanical reducers are all
committed; `.ots` files anchor the protocols to Bitcoin before the data
existed.

## Flagship: a 235B MoE decoding at the PCIe physical limit on ≤16 GB of VRAM

`bench/phase3/` runs Qwen3-235B-A22B with **all expert weights NF4-packed in
host pinned RAM (~128 GB)** and streamed per-token over PCIe, with this
kernel as the sole MoE compute. Same discipline (prereg + OTS, receipts
in-repo):

- **[Phase A](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-offload.md)** (synthetic
  weights, real GQA attention + router): **5.57 tok/s = 102–103% of the
  measured 44.3 GB/s link's waterfall ceiling** — the stream fully hides
  compute — on a **13.6 GB** working set. The dequantize-then-matmul path on
  the identical pipeline: 1.81 tok/s (34% of ceiling). ALL PASS. (Fractions
  marginally above 100% are microbench conservatism: the 1 GiB×10 ceiling
  measurement brackets every copy with a host sync, paying launch +
  sync-return latency the pipeline's continuously-queued copy stream never
  pays.)
- **[The gap is architectural](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-bnb-baseline.md)** —
  we registered the prediction that bnb's own CUDA dequant kernel would
  also hide under the copy shadow (which would have narrowed our claim),
  and it was **refuted**: the standard path reaches **40% of waterfall**
  (per-expert dequant+GEMM compute outlasts the shadow), versus 93–94%
  fused on the same pod. Against the strongest standard comparator the
  fused path is **2.33× tokens/s and 2.21× J/token**.
- **[Phase B](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-phaseB.md)** (the real
  438 GB checkpoint, stream-quantized to NF4 in place): **coherent greedy
  text at 4.3–4.4 tok/s on 15.2 GB VRAM**, replicated across five pods —
  all at 45–55 GB/s datacenter links. The per-token rate is **link- and
  host-dependent**: `t_token ≈ c_box + bytes/link`, with the per-box floor
  `c_box` measured at 53.5–114.0 ms across seven hosts (gen4 desktop
  L40S: 2.6 tok/s; gen5 bare-metal H100 PCIe: 3.9 tok/s — same kernel,
  greedy-identical; see
  [the gen5 doc](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-gen5-metal.md)).
  A fixed "fraction of waterfall" is NOT the law — the two 0.77 readings
  that once suggested one were a two-host coincidence, retired 2026-07-22.
- **Expert prefetch is measured CLOSED, negative** — four registered arcs
  ([B2](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-phaseB2.md) speculation:
  token-to-token expert stickiness is only 0.44;
  [B3](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-phaseB3.md) early routing: the
  pre-attention router predicts the post-attention top-8 at **0.93** but the
  CPU sync tax eats the win;
  [B4](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-phaseB4.md) threaded issuance:
  GIL tax, 0.57×;
  [B5](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/bench/phase3/flagship/RESULTS-flagship-phaseB5.md) GPU-driven
  zero-copy gather: hit rate H makes speculation move (2−H)× the bytes, and
  the observed loss matches that law to ~1% — break-even needs H ≳ 0.95,
  above this model's 0.93 predictor ceiling).
- **Recommended configuration: `--prefetch-mode gpu`** — expert ids stay
  GPU-resident and a triton kernel ([`kernel/host_gather.py`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/host_gather.py))
  gathers expert rows straight from pinned host RAM over UVA (zero-copy),
  with no per-layer memcpy launches and no GPU→CPU syncs. It is the fastest
  measured arm (4.39–4.41 tok/s, +1.5% over serialized memcpy, byte-identical
  greedy output 6/6) and validates SM-issued UVA reads at ≥ copy-engine
  throughput at 7.98 GB/token.

Every comparative "first/only/faster" claim above is backed by a verified,
dated entry in [`docs/COMPETITIVE.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/docs/COMPETITIVE.md)
(the credited prior-art table + the same-box ik_llama A/B) — no entry, no claim.

## Reproduce

See [REPRO.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/REPRO.md) — suite, benchmark, and verdict reduction are each
one command from a frozen tree. Requires an sm_86 GPU, `torch ≥ 2.8`,
`bitsandbytes`, and a C compiler on PATH (triton builds launcher stubs at
runtime).

```
python -m pytest kernel/test_nf4_grouped.py -q        # 44 tests, ~2.5 min
python bench/phase1/harness.py --models OLMoE --regimes decode_bs1 \
    --backends dequant_grouped fused_nf4 --out receipts.json
```

## Layout

- `kernel/nf4_grouped.py` — the kernel (decode gemv path + M-tile path),
  packing helpers, torch reference decode
- `kernel/test_nf4_grouped.py` — property suite (bnb decode exactness at
  bf16 output precision, fidelity ordering, adversarial absmax, boundaries)
- `kernel/prereg_*.json` + `.ots` — pre-registered protocols, stamped
- `kernel/RESULTS-*.md` — results, including the failures
- `bench/phase1/` — backend-registry harness (dequant/gemv/grouped-mm/
  unsloth/marlin/fused), confirmatory evidence, reducers
- `bench/phase2/` — decode config sweeps (both devices); `arch/` —
  cross-architecture census (sm_86/89/90)
- `bench/phase3/` — the 235B offload flagship: `offload_decode_235b.py`
  (Phase A, synthetic), `offload_generate_235b.py` (real checkpoint,
  generation, prefetch arms), `flagship/` — results + receipts
- `kernel/host_gather.py` — GPU-driven zero-copy gather from pinned host
  memory (UVA), the recommended offload copy path
- `docs/KERNEL_CONTRACT.md`, `docs/TOLERANCE_CONTRACT.md` — op contract and
  fidelity spec; `census/`, `roofline/` — shape census + ceilings

Regenerate the machine-generated artifacts:

```
python3 census/make_census.py     # census/shape_census.json
python3 roofline/roofline.py      # roofline/ceilings.json
```

## Status / roadmap

Landed through v6: universal decode constant (the dense-sweep result),
split-K for starved grids (with a per-split work floor), a load-time
min-bytes dispatch floor (tiny cells route to the dequant path via
`decode_dispatch()`), the register-LUT prefill mainloop (v6, confirmed),
and the flagship offload pipeline (Phase A/B + the closed prefetch program
+ the UVA gather path + the bnb-CUDA-dequant baseline, whose registered
prediction was refuted — see the flagship section). The v6 A2000
report-only addendum landed 2026-07-20
([`kernel/RESULTS-v6-a2000-report.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/kernel/RESULTS-v6-a2000-report.md)):
paired prefill medians inside the confirmed band on 7/8 cells at 26 SM
(the eighth 0.9% below the floor, quantified in-doc) — the mainloop is
bracketed 26→170 SM with zero retune. Pending: a bare-metal gen4
replication when stock returns. Parked: sm_120 (three consecutive cloud
provisioning failures on 5090s — availability, not code). Ecosystem landing
is calendar-gated on the bitsandbytes v0.50.0 release; see the coordination
note on #1949.

**Downstream serving** lives in the sibling package
[`experts4bit-qlora`](https://pypi.org/project/experts4bit-qlora/): its
`[fast]` extra routes frozen-expert inference through this kernel (the measured
3.65× above), and its hot-expert residency runs hot and
cold stacks on the same kernel — with 2026-07-20 receipts showing decode
gain tracks routing coverage (informed hot sets +56–120% on gpt-oss, +44%
on Gemma-4). Division of labor: this kernel makes one expert-stack matmul
cheap; e4b decides which bytes are where.

**Cold-engine exploration** (CPU-resident third tier for the coldest
experts) is at phase-0: premise measurements on the target NAS host are in
[`docs/cold-engine/PHASE0-premise.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/docs/cold-engine/PHASE0-premise.md).
Honest status: the "free floor" premise (bnb's CPU `dequantize_4bit` as a
ready-made decode arm) is **refuted** on that box — no AVX-512 means bnb
falls back to its reference path at 0.041 GB/s against a ~12 GB/s DDR
ceiling — so an AVX2 decode port is the mandated phase-2 step before any
integration work. Design-stage; no registered claims.

## Cross-vendor projections (stamped, PROJECTED tier — help us confirm them)

The waterfall arithmetic doesn't care which vendor's bus you're on, so we've
extended it — under the same receipts discipline — into a stamped, pre-silicon
projection table for AMD, Intel, and NVIDIA unified-memory parts:
[`PROJECTIONS-multiarch.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/PROJECTIONS-multiarch.md) (protocol:
[`PROTOCOL-multiarch.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/PROTOCOL-multiarch.md); model + R1 anchor gate:
[`projections/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.2.2/projections/)). Both docs are OpenTimestamps-anchored (`.ots`)
**before any of this silicon was run** — the projections are a falsifiable
prediction, not a marketing table.

**Streaming rows are now graded against measurements** (Addendum 2): the
original pure-waterfall gen5 row (6.0–6.9) is **falsified**, as Addendum 1
pre-registered it would be; Addendum 1's revised **gen4 band (2.4–3.0) is
CONFIRMED** (desktop L40S measured 2.60–2.61) and its revised **gen5 band
(4.0–5.0) missed narrowly** (bare-metal H100 PCIe measured **3.924** — below
the band, above the 3.6 falsification line; that box's floor `c_box = 114 ms`
lies outside the five-pod fitted range). The standing model is **additive,
per-box**: `t_token ≈ c_box + bytes/link` with `c_box` measured (53.5–114.0 ms
across seven hosts, not ordered by link speed) — a fixed fraction-of-waterfall
is retired as a law (`PROJECTIONS-multiarch.md` Addendum 2,
`bench/phase3/flagship/RESULTS-flagship-gen5-metal.md`). Unified-memory rows
remain *ceilings only*: **17–22 tok/s ceiling** on 128 GB unified boxes
(Strix Halo / DGX Spark / Jetson Thor), real decode below them by that same
per-box floor. NF4-vs-bf16 is a **3.56×** byte reduction (absmax-inclusive),
not the round 4×.

**Call for confirmatories.** If you own any listed part, run
`PROTOCOL-multiarch.md` and file the result — **pass or fail** — as an issue.
A refuting measurement is as welcome as a confirming one; that's the point.
Template:

```
Title: [confirmatory] <platform> — <model>
Environment: vendor / device / driver / runtime / triton / torch / bnb;
  link measured via lspci + on-box microbench (streaming) OR mem-band spec
  (unified)
Correctness gate: max rel-err vs dequant_ref = <value>  (pass < 1e-2)
Measured decode: <tok/s> per census cell   Projected band: <from table>
Verdict: within band? / refutes row?   Attach: results JSONL
```

## License & attribution

MIT ([LICENSE](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/LICENSE)). Portions developed with Claude Code as an AI
assistant under the author's direction and review — see
[ATTRIBUTION.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.2.2/ATTRIBUTION.md). All claims are the author's responsibility.

## Portability program

The kernel is single-source Triton; everything that must differ per vendor
is being pulled into `backends/` — device detection, warp/wavefront/sub-group
width, per-arch autotune search spaces. `bench/hw_contract.py` validates
kernel correctness on any torch device **without a bitsandbytes build**; if
you have ROCm or XPU silicon, that is the entry point. `docs/PORTABILITY.md`
is the pre-port hazard register. Per the repo's tier language, every
non-CUDA row is `port target` until a confirmatory passes on that silicon.

## Router-predictability probe

`router_probe/` asks whether the measured H = 0.93 one-layer-lead prediction
ceiling is the router's conditional entropy or the probe's capacity limit.
The charter and procedure were OTS-stamped before any real-model capture; the
Phase-0 instrument gate passed 4/4 on planted fixtures. Phase 1 has run on
**five MoE families** (see `router_probe/RESULTS.md`, exploratory tier):
low-expert-count families pin cleanly at first data volume (gpt-oss-20b E=32
→ 0.83, granite E=40 → 0.90, OLMoE E=64 → 0.91, all model-limited ×3 from the
committed reducer), while **both E=128 families are data-unpinnable** (Qwen3-30B
k=8 ≥0.845 after three data doublings; gpt-oss-120b k=4 ≥0.787 after two) —
high expert count doesn't just lower H, it makes H unmeasurable by data
scaling on this ladder, at both k. Every observed plateau sits far below the
≈0.95 wire-law break-even for speculative expert streaming.

## Contact

Cerin Amroth Research takes contract and pilot engagements on this work —
kernel ports, offload integration, and sponsored research lanes with
stamped receipts. Contact **jordan@cerinamroth.com**.
