### A2000-timing audit, the owner's decisions applied: the dgrad row relabelled, the split-K row retired, its timing tests out, four lane errata (no output change)

- **Why.** The maintainer's decisions 1, 2, 3 and 5 on #475, for the items that need no rented compute. The RTX A2000 is a
  correctness-only testbed, so an A2000 timing may not be a headline, a CI gate or a band basis.
- **`gnf4.kernel.dgrad` is no longer a speed row.** Its id stays, and its value is now the correctness result the claim
  already stated: one launch, gradient ~2.9e-3 (relative) from the exact per-expert loop, inside the bf16 budget. The README
  and STATUS tables quote that, and the 0.7.0 entry keeps the A2000 step time as the record.
- **New row `gnf4.kernel.dgrad-step.a6000.2026-08-06`, the rented speed headline.** On a rented RTX A6000 with Qwen3-30B-A3B
  at 48 layers and the published wheels, the fused training step runs 2.52x the reference loop with dgrad and 1.72x without.
  The receipt is in experts4bit-qlora (`bench/dgrad-gate/RESULTS-dgrad-gate.md`), cited as cross-repository evidence. The
  README and STATUS tables quote it.
- **`gnf4.serve.int4-b32-splitk-row-term.a2000.2026-09-10` is retired**, with the reason "an A2000 timing; the A2000 is a
  correctness-only testbed". STATUS, `docs/solutions/int4-decode-gemv.md` and `docs/capabilities.json` no longer cite it as
  current.
  - **The plan is unchanged.** `SPLITK_R_TERM_MAX_SMS` still keeps the R term on for parts with 64 SMs or fewer. A rented
    read on such a card decides whether the term stays; flipping the gate is reorder-class.
- **`kernel/test_int4_b32.py` stops asserting the plan against A2000 timing receipts.**
  - Three tests go, with `SK_R_BOUND`: the worst-cell bound, the summed win over the N-only rule, and the per-cell
    never-slower comparison. The last is the same kind of timing gate, so it went with the other two.
  - The structural tests stay: R < 16 returns the N-only plan on every SM count; sk never grows with R; the grid-full
    collapse to sk = 1; KU divides the k-blocks; the N-only plan above 64 SMs.
  - New: `test_plan_is_a_pure_function_of_n_k_r` pins the property `_plan`'s caveat states.
- **Errata, registrations as stamped** (the experts4bit-qlora#1128 pattern). Each RESULTS gains a dated erratum naming its
  A2000-seeded basis, and says that its verdict rests on the 5090 alone:
  - K14, Amendment 1's redesign;
  - K16, the pilot's go-to-rent call;
  - K26, its DECODE and `tree / pair` bands;
  - K27, its speed band and verdict lean.
- **`prereg_dequant_forward_floorfree.json` is anchored**, so it is not edited. A VOID A2000 smoke motivated where its F1
  band sits; the erratum sits beside it (`kernel/prereg_dequant_forward_floorfree.ERRATUM-2026-10-05.md`), and
  `kernel/ERRATA.md` points there.
- **Not in this change:** the rented reads (the ≤ 64-SM split-K step read), and re-sweeping the speed-only constants.
