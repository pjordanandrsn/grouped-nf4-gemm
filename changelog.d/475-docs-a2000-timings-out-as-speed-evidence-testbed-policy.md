### Docs: A2000 timings out as speed evidence (testbed-policy audit; docs and comments only)

- **Why.** The RTX A2000 is a correctness-only testbed (experts4bit-qlora's testbed policy, standing since 2026-07-27,
  re-stated 2026-10-05): an A2000 timing may not seed a prediction, filter a candidate, or appear as speed evidence. An
  audit of both repositories found such timings in this package's comments and docs. The findings table, including what is
  left for the owner, is [`docs/audits/a2000-timing-2026-10-05.md`](docs/audits/a2000-timing-2026-10-05.md).
- **Comments** (no behaviour change, no constant moved). `_triton_shim.py`, `cold_deadline.py`, `fp8_paged_attn.py`,
  `int4_b32.py`, `mxfp4_grouped.py`, `nf4_grouped.py`, `nf4_qlora.py`, `nf4_route.py` and `bench/calibrate.py` no longer
  quote A2000 timings, bandwidths or ratios. The dgrad default now cites experts4bit-qlora's rented A6000 dgrad gate (2.52x
  vs 1.72x). `link_eff` cites the two rented 5090 hosts. `fp8_paged_attn`'s precision table keeps its error column only.
- **Comments now say where a shipped choice came from an A2000 timing**, and that it is unverified on a target card. That
  covers the int4-b32 split-K R term and `SPLITK_TARGET_BLOCKS_PER_SM` (still on for parts with 64 SMs or fewer), MXFP4's
  N-only plan, `_TILE_D_DEFAULT = 96`, and `_DGRAD_DEFAULT`. Outputs do not depend on any of them.
- **STATUS, register and docs.** The A2000 `link_eff` of 1.0 leaves `STATUS.md` and `cold-engine/ARCHITECTURE-NOTES.md`.
  Two rows' notes drop A2000 timings: K17's pre-launch pilot and P69's control.
- **Left for the owner** (listed in the audit file): `gnf4.kernel.dgrad` (403.7 -> 26.5 ms, an A2000 timing quoted in the
  README and STATUS tables), `gnf4.serve.int4-b32-splitk-row-term.a2000.2026-09-10` (value 1.011x, A2000), the
  `SK_R_BOUND` test, which pins the plan against A2000 timing receipts, and the PREREG/RESULTS records.
