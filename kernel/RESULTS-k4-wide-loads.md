# RESULTS — K4 wide loads: REFUTED-FOR-WIDTH, and the wall moved

Run 2026-08-25 against `PREREG-k4-wide-loads.md` (#246), EPYC 9-series
+ RTX 5090 (one boot-dead box auto-refused; both destroyed + verified
zero). Receipts in `receipts-k4-wide/`; verdict by `k4_verdict.py`
(self-tested, 9 branches).

## Verdict table

| gate/bar | registered | measured | result |
|---|---|---|---|
| G-B bitwise (kernels + e2e ×127) | exact | PASS | PASS |
| **H-M mechanism (floor ≤ 0.5×)** | wide floor halves | **0.18× / 0.47×** | **PASS** |
| H-K wide pair | ≤ 40 µs | 69.3 (legacy 72.9) | FAIL |
| H-E graph step | ≤ 12.0 ms | 13.34 (legacy 13.46) | FAIL |
| GS B=16 | in band | 129.2 ms | PASS |
| G0 arms | < 7.5% | 0.040% / 0.007% | PASS |

**⇒ REFUTED-FOR-WIDTH, the registered consequence honored exactly:
legacy stays the default** (wide remains opt-in via
`GNF4_GEMV_WIDE_LOADS=1` — it is 5% better and bitwise, but 5% was
not the claim and a certified bar is not a suggestion).

## The finding that outranks the verdict

The mechanism gate did its job in both directions at once. Width
FIXED the loads: the wide streaming floor is **9.98 µs against an
8.0 µs roofline** (gate_up) — the access pattern is no longer the
wall. Yet the full kernel moved 72.9 → 69.3: **~59 µs of compute now
stands exposed** — the nibble shifts, the LUT gather, and the
`tl.sum` mul-reduce chain.

This measures, rather than suspects, the overlap caveat of peel
attribution: K3's subtractive account was correct AT THE SCALAR
OPERATING POINT, where the serial load bottleneck hid parallel
compute costs (removing arithmetic barely moved the time because the
loads gated it). Change the loads and the account's operating point
changes with them. The `residual reported` discipline is what kept
that account honest enough to re-read now.

## The corrected next lane (named, not registered here)

The prereg's escalation clause (k-strip repacking) targeted LOADS —
its premise is refuted by these receipts: loads are at roofline in
the wide variant. The lane's true next registration is the COMPUTE
chain, and the codebase already contains the shape of the answer:
the M-tile kernel runs this same math through **tensor-core
`tl.dot` with the VARIANT-1 register-LUT**, both of which the GEMV
reduction lacks. K5's registration question: route B=1 decode through
the M-tile path (existing code, existing fidelity story, measurable
with config forcing) vs. a `tl.dot`-based GEMV rewrite — decided by
a cheap existing-code probe before any new kernel is written.

Single-stream state: **74.3 tok/s stands** (K1 configuration, still
the default). e2e receipts here: legacy 13.46 / wide 13.34 ms — the
74.96 tok/s wide number is real but unshipped with the refutation.
