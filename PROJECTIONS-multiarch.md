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


---

## Addendum 1 (2026-07-18) — serialization term; gen5 bands revised

**Forward-only correction.** The rows above are stamped and untouched; this
addendum registers a term the original model omitted, refits the affected
streaming-regime bands, and pre-registers the expectation that the original
gen5 rows falsify while these revised bands hold. Tier: the revised bands are
**projections under an estimated term** — one tier below the original rows'
pure-waterfall status.

### What the original rows assumed

The streaming model above is `tok/s = link_GBps / bytes_per_token`
(`projections/model.py:9`, `streaming_waterfall_toks` at
`projections/model.py:33`): every per-token expert byte crosses the link, and
all other work hides under the copy. Its anchors (Phase A 44.3 GB/s → 5.55
tok/s; 671B 55.5 → 4.59) are fixed-token *streaming-form* measurements, where
nothing on the critical path waits for a generated token.

### What the data showed

Real autoregressive decode carries per-token work that does **not** hide under
the copy: the router/dispatch/launch path runs serially between transfers
(established across the prefetch arc — B3 measured the router-serialization
tax directly, B5 closed the mechanism). Five flagship Phase-B hosts (four RunPod H100s, one DigitalOcean H200) with
on-box link microbenches and off-mode (shipped-config) decode:

| receipt | link L (GB/s) | B/L (ms) | measured off (tok/s) | 1/off (ms) | implied t_s (ms) | off ÷ waterfall |
|---|---|---|---|---|---|---|
| `bench/phase3/flagship/phaseB2.json` | 45.45 | 175.7 | 4.303 | 232.4 | **56.7** | 0.756 |
| `bench/phase3/flagship/phaseB3.json` | 55.09 | 144.9 | 4.331 | 230.9 | **86.0** | 0.628 |
| `bench/phase3/flagship/phaseB4.json` | 51.78 | 154.2 | 4.351 | 229.8 | **75.7** | 0.671 |
| `bench/phase3/flagship/phaseB5.json` | 44.98 | 177.5 | 4.330 | 231.0 | **53.5** | 0.768 |
| `bench/phase3/flagship/do_replication/phaseB_do.json` | 55.35 | 144.2 | 4.430 | 225.7 | **81.5** | 0.639 |

(`B = per_token_gb = 7.9839` from the same receipts; off = median of each
receipt's `toks_per_s_off` runs; three-to-six runs each, greedy-identity
gated.) The faster the link, the smaller B/L — and the further
measured decode falls below the waterfall (0.77× at 45 GB/s → 0.63× at
55 GB/s). At gen5-class links the omitted term is no longer a correction; it
is comparable to the transfer itself.

### The revised model

```
tok/s ≈ 1 / (B/L + t_s)        t_s = per-token serialized work
```

Fitted per host: **t_s = 53.5–86.0 ms** (median 76 ms, spread 1.61× — same
order on every host; the sanity gate for a one-term correction passes).
Evidence tier: *estimated from 5 hosts' measured throughput; not independently
confirmed on desktop silicon.*

Honest limit of the fit, on the record: a **single** t_s constant does not
reconcile the five hosts — the measured per-token total is nearly flat
in-sample (225.7–232.4 ms while B/L spans 144.2–177.5 ms), so the per-host
t_s anti-correlates with link speed. Two readings are compatible with the
data: per-host serial-path variance (CPU/driver/launch differences across
pods), or partial overlap of the serial work with the transfer (t_s grows as
the copy window shrinks). The receipts cannot distinguish them; the fitted
*range* brackets both, and at gen5 links both readings land inside the
Addendum-1 band below (the flat-total reading pins ~4.3–4.4 tok/s; the
additive reading spans the band).

### Revised rows — Addendum-1 bands (supersede the original gen4/gen5 streaming rows for falsification purposes)

`tok/s = 1/(B/L + t_s)` with the file's own link bands and B = 7.9839,
t_s ∈ [53.5, 86.0] ms:

| platform | link band | Addendum-1 band (235B decode) | falsification |
|---|---|---|---|
| Desktop gen5 ×16 (5090 / RDNA4 host) | 48–55 GB/s | **4.0–5.0 tok/s** | <3.6 or >5.5 |
| AMD RDNA4 (RX 9070 / XT) | 48–55 GB/s | **4.0–5.0 tok/s** | <3.6 or >5.5 |
| Desktop gen4 ×16 / Arc A770 / Arc B580 | 24–28 GB/s | **2.4–3.0 tok/s** | <2.2 or >3.3 |

Two notes the derivation forces:

- **The gen4 rows are *not* immune.** At 24–28 GB/s the transfer (285–333 ms)
  still dominates, but t_s at 53.5–86.0 ms moves the band to 2.4–3.0 —
  entirely at-or-below the original 3.0–3.5 band-low. The original gen4 rows
  are as much at risk as gen5, just by a smaller ratio.
- **The 30B-A3B secondary line inherits the omission with an unmeasured
  magnitude.** t_s here was measured on the 235B pipeline (94 layers); the
  30B pipeline (48 layers, different serial path) has no measured off-mode
  anchor, and t_s does not obviously transfer. Because B/L is small for 30B
  (≈20 ms at gen5), *any* t_s of this order would dominate — the derived
  30B streaming numbers above should be read as **waterfall ceilings, not
  decode predictions**, until a 30B off-mode anchor is measured. No constant
  is fitted here.

### Standing prediction

A real gen5 desktop confirmatory is expected to land **inside the Addendum-1
band (4.0–5.0) and below the original band (6.0–6.9)** — i.e., we predict the
original gen5 rows falsify and the revised rows hold. Either outcome is
informative; both are pre-registered as of this addendum's stamp.

### What would falsify this addendum

A gen5 measurement at or above the original 6.0–6.9 band. That would mean t_s
is hideable on desktop topologies — plausible mechanisms would include a
different IOMMU/driver launch path or deeper driver-side pipelining than the
datacenter hosts showed — and the commitment is to investigate that mechanism,
not to hand-wave the number.

### Related surfaces (not edited here)

- `README.md` (call-for-confirmatories headline) restates the original gen4/gen5 headline numbers.
- `PROTOCOL-multiarch.md:49–50`'s streaming pass band requires
  `achieved ≥ 0.80 × on-box waterfall`; the five hosts above measured
  0.63–0.77×, so that clause embeds the same omission and would fail a
  faithful run with a perfect kernel.

Both are flagged for a forward-only pointer/amendment decision; neither is
touched by this addendum.
