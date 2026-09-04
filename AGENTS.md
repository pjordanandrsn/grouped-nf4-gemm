# AGENTS.md — working in grouped-nf4-gemm

Operational notes for coding agents and contributors. Not a second README:
[`README.md`](README.md) argues the case, [`docs/STATUS.md`](docs/STATUS.md)
states the position, [`docs/claims.json`](docs/claims.json) holds the numbers.

## Purpose

`grouped-nf4-gemm` is the kernel side of 4-bit Mixture-of-Experts serving:
one Triton launch for the grouped expert GEMM directly on NF4-packed
(bitsandbytes `gemm_4bit` layout) or native MXFP4 (e2m1 + e8m0) expert
stacks, plus the int4-b32 decode GEMV and its calibrated packer, the fp8
paged decode attention, the decode glue kernels, the pure-torch references
every kernel is asserted against, and the host/NVMe arena primitives. The
consumer is [`experts4bit-qlora`](https://github.com/pjordanandrsn/experts4bit-qlora),
which loads and quantises models, trains adapters, places bytes across
tiers and serves, and installs this package through its `[fast]` extra. A
kernel lands here first; the consumer floors on the release that ships it.

## Repository map

| path | what |
|---|---|
| `kernel/` | the shipped flat modules (`pyproject.toml` `py-modules`): `nf4_grouped`, `mxfp4_grouped`, `int4_b32`, `int4_pack_ref`, `gptq_pack`, `nf4_pack_ref`, `mxfp4_pack_ref`, `mxfp4_loader`, `verify_provenance`, `fp8_kv`, `fp8_paged_attn`, `nvme_arena`, `nvme_bake_nf4`, `nvme_reader`, `nvme_residency`, `row_pool`, `arena_experts`, `mxfp4_residency`, `nf4_qlora`, `mxfp4_qlora`, `cpu_grouped`, … — and their tests (`kernel/test_*.py`) beside them; `_triton_shim` binds triton or a stand-in |
| `gnf4_native/` | the compile-at-first-use CPU kernels (the hybrid CPU path) |
| `examples/dequant_tax.py` | the one-minute on-your-own-GPU demonstration; CI runs it on CPU and asserts the CPU note |
| `docs/` | `STATUS.md`, `claims.json` + `claims-schema.md`, `KERNEL_CONTRACT.md`, `TOLERANCE_CONTRACT.md`, `PORTABILITY.md`, `SOLUTIONS.md` + `solutions/`, `capabilities.json`, `INDEX.md`, results and pre-registrations (many anchored) |
| `bench/`, `projections/`, `router_probe/` | receipts, the projection model and its anchor gate |
| `scripts/` | README link checker, wheel smoke, capability/discovery/llms checks |

## Canonical public API

The entry-point table in `README.md` ("Which entry point? Pick by where the
weights live") and `docs/capabilities.json` (`entrypoints`) are the list.
Kernels: `nf4_grouped.gemm_4bit_grouped` / `gemm_4bit_grouped_captured` /
`dgrad_4bit_grouped`, `mxfp4_grouped.gemm_mxfp4_grouped` /
`gemv_mxfp4_b32`, `int4_b32.gemv_int4_b32` / `quant_x_rows` /
`gemm_int4_b32_grouped_captured` and the glue (`rmsnorm_rows`,
`rmsnorm_resid_rows`, `scaled_resid_add_rows`, `rope_norm_heads`,
`rope_heads`, `router_epilogue`, `swiglu_rows`, `combine_rows`,
`reduce_partials`), `fp8_kv.*`, `fp8_paged_attn.fp8_paged_decode_attention`.
References and packers (pure torch): `nf4_grouped.dequant_ref`,
`nf4_pack_ref.quantize_pack_nf4`, `int4_pack_ref.pack_int4_b32`,
`gptq_pack.gptq_pack_int4_b32` + `HessianAccumulator`,
`mxfp4_pack_ref.*`. Storage and provenance: `nvme_arena.bake` /
`bake_expert_tensors` / `verify`, `nvme_bake_nf4.bake_nf4`,
`nvme_reader.ArenaReader`, `nvme_residency.ColdTier` /
`capacity_for_bytes`, `arena_experts.ArenaExpertSource`,
`mxfp4_residency.Mxfp4NvmeResidency`, `mxfp4_loader.file_tensor_sha256` /
`provenance_table` / `verify_arena_matches`, `python -m verify_provenance`.
Modules shipped in the wheel but not in that list (`cold_*`, `reuse_profile`,
`vram_slots`, `segmented_pool`, `dev_row_cache`, `run_mxfp4_20b_qlora`,
`gate_native_load_20b`, `_triton_shim`) are internal; shipping is not
publishing.

## Sources of truth

- **Whether a number is current: `docs/claims.json`**, not CHANGELOG prose
  or a README sentence. `status` is the vocabulary (`confirmed`,
  `measured`, `measured-private`, `projected`, `retired`, `superseded`,
  `open`); retired and superseded claims are never repeated as current.
- **The current position: `docs/STATUS.md`** — including the three limits
  where the fused path loses.
- **Which documents are current: `docs/INDEX.md`.** Anchoring here is a
  sibling `.ots` file (100 of them; `find . -name "*.ots"`): an anchored
  document is never edited in place; corrections go in a sibling file.

## Build and test

```bash
pip install -e . pytest numpy                      # CPU torch + triton (Linux) run the interpreter suites
cd kernel && TRITON_INTERPRET=1 python -m pytest test_interp_contract.py test_int4_b32.py -q
cd kernel && python -m pytest test_packaging_covers_kernel.py test_check_readme_links.py -q
python examples/dequant_tax.py                     # prints the CPU note without a GPU; ~1 min on one GPU
python -m build && python -m twine check dist/*
python scripts/wheel_smoke.py                      # from outside the tree, against the wheel
python scripts/check_readme_links.py               # README links are absolute; self-refs = v<version> or main
python scripts/check_capabilities.py               # docs/capabilities.json vs schema, pyproject, source, claims
python scripts/check_discovery_contract.py         # docs/discovery-queries.json vs docs/solutions/
python scripts/build_llms_bundle.py --check        # llms-full.txt is current
```

CI (`.github/workflows/ci.yml`) runs the anchor gate, the example on CPU,
the interpreter-contract suites under `TRITON_INTERPRET=1`, and the wheel
smoke; it asserts triton is importable so a skip can never be silent. GPU
measurements come from rented lanes with receipts; **a green skip is not
evidence a path was exercised** — an interpreter-mode file must be
registered in `kernel/conftest.py` (`_INTERP_FILES`) and named in the CI
step, and the meta-tests enforce both.

## Rules that have bitten

- Every kernel has a pure-torch reference and is asserted against it;
  parity with the reference outranks speed in review.
- Claims carry receipts, a self-pair beside every ratio, and the cells that
  lose; a pre-registered protocol is stamped before the data exists.
- Examples must not silently fall back: a kernel call on CPU raises and
  names the reference; `examples/dequant_tax.py` prints a CPU note and
  returns rather than pretending.
- Layouts are contracts (`docs/KERNEL_CONTRACT.md`): NF4 `[E, N, K//2]` u8 +
  absmax `[E, N, K//64]` fp32; MXFP4 blocks `[E, N, K//2]` u8 (low nibble
  first) + e8m0 scales `[E, N, K//32]` u8; int4-b32 packed `[E, N, K//2]` +
  scales `[E, N, K//32]` fp16. The consumer's stores are built to these.

## When you change something

- **A public kernel or reference** (signature, layout, return): update the
  module docstring, `docs/KERNEL_CONTRACT.md` if a layout moved, the
  README entry-point table, `docs/capabilities.json`, the affected
  `docs/solutions/*.md`, `CHANGELOG.md` under `## Unreleased`, and open the
  consumer-side change (or issue) in experts4bit-qlora.
- **A dependency floor** (`torch`, `triton`, `numpy`): the pyproject comment
  records why; the README install note and `docs/capabilities.json`
  environments follow.
- **The README opening, `docs/SOLUTIONS.md`, `docs/STATUS.md`,
  `docs/capabilities.json` or a listed document:** regenerate
  `llms-full.txt` (`python scripts/build_llms_bundle.py`); `--check` is a
  CI gate.
- **A measured position:** the claim entry first, then `docs/STATUS.md`,
  then prose that quotes the claim ID.

## Platform caveats

Kernels: Linux + NVIDIA sm_80+ with Triton ≥ 3.4 (sm_120 is the primary
serving target). CI tests Python 3.11 only, on Linux. On macOS/Windows the
wheel installs (triton is a Linux-only marker); the README's older note
reports the CPU quickstart failing at import there, `_triton_shim` binds a
stand-in so define-time imports resolve, and CI does not validate either
statement — say "not validated" rather than either. ROCm/XPU are port
targets (`docs/PORTABILITY.md`),
not supported. The fp8 paged kernel's f32 compute modes fail on triton 3.4
(`gnf4.open.f32-compute-modes-triton34`).

## Release notes

First paragraph in ordinary language — which problem changed, who is
affected, whether to upgrade — then mechanism, measurements with receipts
and tiers, corrections and caveats
([`docs/RELEASE_NOTES_GUIDE.md`](docs/RELEASE_NOTES_GUIDE.md)). Releases
are GitHub releases on `v<version>` tags cut by the maintainer
(`publish.yml` refuses a tag that disagrees with `pyproject.toml`); do not
tag or publish from a branch.
