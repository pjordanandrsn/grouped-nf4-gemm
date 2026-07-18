# Multi-architecture confirmatory protocol — grouped-nf4-gemm

**Stamped pre-data (see .ots).** The measurement protocol that turns a
`PROJECTED` row in `PROJECTIONS-multiarch.md` into a `CONFIRMED` (or refuted)
one. Reuses the existing reducer style (`bench/phase1/reduce_confirmatory_*.py`)
— no new verdict logic is invented here. One protocol, two regimes.

## Standing rules (inherited)

- **R1 anchor gate.** Before any new row, `projections/test_anchors.py` must be
  green on the measuring host's checkout. Red anchor ⇒ stop, do not report.
- **R3 tier language.** A platform is `PROJECTED` / `port target` until a run
  under this protocol passes. Only then may a doc say "confirmed on."
- **No-tune clause.** Failures publish at full volume; no re-run until a fix
  lands. A drift/contention failure is disclosed with its mechanism, not
  re-rolled (the flagship S4 precedent).
- **Per-port stop rule.** Two consecutive sessions without a green correctness
  gate ⇒ close the port with a mechanism writeup (the prefetch precedent:
  closed with constants, not fatigue).

## Environment fingerprint (record every field, per run)

Vendor, device name, driver/runtime version, compute-runtime/NEO or ROCm/CUDA
version, Triton (or backend) version, torch version, bitsandbytes version.
**Link/bandwidth measured, never assumed** — read `lspci -vv` (negotiated
PCIe gen × width) for streaming hosts; run the on-box `1 GiB × 10` pinned-copy
microbench for the effective streaming link; for unified hosts record the
memory config from `sysfs`/vendor tool and cite the spec bandwidth. Emit all
fields into the results JSONL header.

## Correctness gate (both regimes, precedes any perf number)

1. Generate the canonical test vector: `projections/`-style or
   `sycl/gen_testvec.py` — real bnb-NF4 weights, oracle = the parent
   `dequant_ref`. Ship it to the target.
2. Run the ported kernel; compare to the oracle. **Pass = max rel-err < 1e-2**
   (the parent's fidelity bound). Interpreter-mode (CPU) reference must also be
   reproduced where the backend supports it (Phase-1 suite).
3. Correctness FAIL ⇒ stop; perf numbers from an incorrect kernel are void.

## Streaming regime (discrete GPU, experts in host RAM)

- **Census cells:** the phase-1 census shapes (OLMoE, Qwen3-30B, Gemma-4,
  GPT-OSS) at decode_bs1, plus the flagship offload geometry (235B-A22B) if
  host RAM ≥ ~140 GB.
- **Warm-up:** 4 tokens discarded; ≥ 64 timed tokens.
- **Reducer:** median s/token → tok/s; on-box microbench link → waterfall
  ceiling; report `achieved / waterfall`.
- **Pass band:** decode tok/s within the projected band's ±15%, AND
  `achieved ≥ 0.80 × on-box waterfall`. Outside ⇒ row refuted (report the
  measured value; the model, not the kernel, is what failed).
- **Two arms** (A-A drift guard), within 5%; a shared-host drift failure is
  disclosed and attributed, not re-graded (flagship S4 precedent).

## Unified regime (APU/SoC, weights resident in shared memory)

- **No streaming, no host-link microbench.** Ceiling = `mem_bw_GBps /
  (active_params_B × 0.5625)` (absmax-inclusive). Record spec + any measured
  memory bandwidth (e.g. a STREAM-class probe) in the fingerprint.
- **Census:** the model must FIT resident — record measured resident footprint;
  30B-A3B fits 128 GB comfortably, 235B-NF4 (~120 GB) is borderline, report
  actual.
- **Pass band:** measured decode within `[0.40, 1.00] ×` the ceiling. **A
  measured result ABOVE the ceiling refutes the model** (the ceiling is an
  upper bound; exceeding it means the byte-accounting is wrong). Below 0.40×
  is reported as an efficiency finding, not a model refutation.
- **Also report** NF4-vs-bf16 measured ratio where both fit — the model predicts
  3.56×.

## Deliverable per attempt

`RESULTS-<platform>.md`, published **regardless of verdict**, carrying: the
environment fingerprint, correctness-gate result, the per-cell measured vs
projected table, the reducer verdict, and — on failure — the mechanism. Hash +
`ots stamp`. Update the `PROJECTIONS-multiarch.md` row's tier only via a
forward-only changelog line (R2 — the stamped projection file is never edited).

---

## Amendment 1 (2026-07-18) — streaming pass band follows Addendum-1 bands

**Forward-only.** The clauses above are stamped and untouched; this amendment
supersedes two of them for streaming-regime *real-decode* cells, following
`PROJECTIONS-multiarch.md` Addendum 1 (the serialization term `t_s`, fitted
from five flagship Phase-B hosts).

1. **Band clause.** "Decode tok/s within the projected band's ±15%" now reads
   the **Addendum-1 band** for the row (gen5 4.0–5.0, gen4-class 2.4–3.0
   tok/s for 235B-A22B), not the original pure-waterfall band.
2. **Floor clause.** "`achieved ≥ 0.80 × on-box waterfall`" is superseded:
   the five measured hosts achieved 0.63–0.77× of waterfall with a correct
   kernel — the floor as written embeds the omitted term and would fail a
   faithful run. It becomes **`achieved ≥ 0.80 × the Addendum-1 predicted
   tok/s for the measured on-box link`** (equivalently ~0.50–0.65× waterfall
   at gen5 links). Report `achieved / waterfall` as before — it is the
   diagnostic that localizes t_s.

Unified-regime clauses are unaffected. A measurement satisfying the *original*
clauses (≥0.80× waterfall) would simultaneously refute Addendum 1 — report it
at full volume; that outcome is pre-registered as informative in the addendum.
