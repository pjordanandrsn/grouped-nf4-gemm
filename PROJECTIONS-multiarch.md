# Cross-vendor decode-throughput projections — grouped-nf4-gemm

**Tier: PROJECTED. No row here has been confirmed on its silicon.** Every
number is model output (`projections/model.py`), not a measurement. The model
reproduces the published flagship rows to <2% (`projections/test_anchors.py`,
the R1 gate) — that anchors the *streaming* form on real hardware, but a
projection for a platform is refuted or confirmed only by running
`PROTOCOL-multiarch.md` there. Owners of this hardware: please run it and file
results, pass or fail (see the call-for-confirmatories in the repo README).

**Prepared 2026-07-15 · Cerin Amroth Research · stamped pre-silicon (see .ots).**

## The model in one line each

- **Streaming** (discrete GPU, experts stream from host RAM): `tok/s = link_GBps
  / bytes_per_token`. Anchored: 44.3 GB/s ÷ 7.98 GB/tok = 5.55 (Phase A);
  55.5 ÷ 12.09 = 4.59 (671B). Both measured, both reproduced.
- **Unified** (APU/SoC, weights resident in shared memory): `tok/s = mem_bw_GBps
  / (active_params_B × bytes_per_param)`. **Ceiling only — no measured anchor
  yet, and real decode falls well short of it** (see the Strix Halo reality-check
  below).

**NF4 vs bf16 is a 3.56× byte reduction on the binding resource — not the round
4×** — because NF4 carries fp32 blockwise absmax (0.5 + 0.0625 = 0.5625 vs
bf16's 2.0 bytes/param). This correction is baked in; the model computes 0.5625.

## Streaming regime — Qwen3-235B-A22B (7.98 GB/token)

| platform | link | eff. band | projected 235B decode | falsification |
|---|---|---|---|---|
| Desktop gen4 ×16 **(anchor)** | PCIe 4.0 ×16 | 24–28 GB/s | **3.0–3.5 tok/s** | measured outside band ⇒ model wrong |
| Desktop gen5 ×16 (5090 / RDNA4 host) | PCIe 5.0 ×16 | 48–55 GB/s | **6.0–6.9 tok/s** | <5.4 or >7.5 |
| Intel Arc A770 16 GB | PCIe 4.0 ×16 | 24–28 GB/s | **3.0–3.5 tok/s** | outside band |
| Intel Arc B580 | **PCIe 5.0 ×8** (≈ 4.0 ×16) | 24–28 GB/s | **3.0–3.5 tok/s** | outside band |
| AMD RDNA4 (RX 9070 / XT) | PCIe 5.0 ×16 | 48–55 GB/s | **6.0–6.9 tok/s** | <5.4 or >7.5 |

**Plan-table correction (R6):** the B580 is **PCIe 5.0 ×8**, not "PCIe 4.0 ×8."
Gen5 ×8 delivers gen4-×16-equivalent bandwidth, so its roof equals the anchor —
it is *not* a half-roof "wrong instrument." The A770 (gen4 ×16) is the clean
Intel streaming instrument; the B580 lands at the same band by a different route.

Secondary (Qwen3-30B-A3B, 1.02 GB/token, derived): gen4 ×16 → 23.5–27.5 tok/s;
gen5 ×16 → 47.1–54.0.

## Unified regime — CEILINGS ONLY (no anchor, real-world runs lower)

| platform | mem band (cited) | 235B ceiling | 30B-A3B ceiling | fits 235B-NF4 (~120 GB)? |
|---|---|---|---|---|
| AMD Strix Halo (Ryzen AI Max+ 395, 128 GB) | 256 GB/s theo, ~215 meas | 17.4–20.7 | 127–152 | borderline (120 in 128) |
| NVIDIA DGX Spark (GB10, 128 GB) | 273 GB/s | 22.1 | 161.8 | borderline |
| NVIDIA Jetson AGX Thor (T5000, 128 GB) | 273 GB/s | 22.1 | 161.8 | borderline |

**Reality-check that these are ceilings, not predictions:** independent Strix
Halo reviews report ~45–100 tok/s on 30B-class models — i.e. **~40–65% of the
127–152 ceiling above.** Real decode never hits 100% of memory bandwidth (KV/
attention traffic + <100% efficiency), so treat the unified column as an
*upper bound the hardware will sit below*, and weight these rows lower than the
anchored streaming rows until a confirmatory lands. The falsification criterion
for a unified row is inverted: a *measured* result **above** the ceiling refutes
the model.

## MI300X — resident, not streaming (noted, not headlined)

192 GB HBM3 at 5.3 TB/s holds a 235B-NF4 model (~120 GB) fully resident — no
streaming needed. The resident ceiling (5300 ÷ active-bytes) is ~428 tok/s
(235B) / ~3141 (30B) — memory-bandwidth-bound like the unified regime but at
HBM speed. This is a "big-HBM card runs it resident, fast" result, orthogonal
to the streaming thesis; the streaming projection only applies to MI300X for
models **larger than 192 GB**. Rented MI300X is the cheapest confirmatory
(~$1.85–3.49/hr across providers, verify at launch) and is the first target in
`PROTOCOL-multiarch.md`.

## Sources (R6)

- PCIe 4.0 ×16 = 32 GB/s, 5.0 ×16 = 64 GB/s (theoretical, unidirectional):
  TechRadar, "PCIe lanes explained." Effective ~75–85% of theoretical.
- Strix Halo 256-bit LPDDR5x-8000, 256 GB/s theo / ~215 GB/s measured: AMD
  product page + Strix Halo LLM reviews (localaimaster, runaihome).
- DGX Spark GB10 273 GB/s (256-bit LPDDR5x-8533): NVIDIA DGX Spark hardware
  docs; chiplog.io GB10 analysis.
- Jetson AGX Thor T5000 273 GB/s (256-bit LPDDR5x), devkit $3,499: CNX-Software,
  SCAN, ServeTheHome.
- Arc B580 PCIe 5.0 ×8; A770 PCIe 4.0 ×16: Tom's Hardware, VideoCardz.
- RX 9070 / XT PCIe 5.0 ×16: TechRadar.
- MI300X 192 GB HBM3, 5.3 TB/s; cloud ~$1.85–3.49/hr: AMD Instinct spec;
  getdeploying / thundercompute MI300X pricing (July 2026).

*Constants captured 2026-07-15; cloud rates and street prices move — re-verify
before committing spend (R6).*
