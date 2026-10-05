# Erratum (2026-10-05) to `prereg_dequant_forward_floorfree.json`: a VOID A2000 smoke motivated where the F1 band sits

*The registration is OpenTimestamps-anchored (`prereg_dequant_forward_floorfree.json.ots`; the pre-amendment-2 bytes
under `prereg_dequant_forward_floorfree.json.pre-amendment2.ots`), so it is not edited. This file sits beside it
instead, and `kernel/ERRATA.md` points here.*

**What the registration says.** Its `prior_work_disclosure` (registered 2026-08-13, after the 2026-07-27 testbed
instruction, which it cites) discloses a two-cell smoke on the QNAP A2000 before the stamp. It calls both cells VOID by
the protocol's own Q1 rule and says neither number is a measurement. It then says those VOID figures motivated
registering the F1 band where it is registered, while stating that they are not evidence for it and that F1 is
adjudicated only on the rented devices.

**The erratum.** Under the testbed policy (the A2000 is a correctness-only testbed; every timing, ratio or band basis
comes from rented compute on the target card; grouped-nf4-gemm#475, `docs/audits/a2000-timing-2026-10-05.md` §2), an
A2000 timing may not decide where a band sits, even a VOID one disclosed as such. That sentence of the disclosure names
a basis the policy does not allow.

**What it changes.**

- **The registration stands as stamped**, and so does its verdict. F1 was read only on the rented devices (the H100 and
  the RTX 4090), as the registration requires, and the A2000 appears in no verdict.
- **The band has a basis independent of the A2000.** `F1_floorfree_small_batch` registers leg 1's own S1 band
  (1.3–3.0×), unchanged, so that either outcome reads against leg 1. That is the band's stated rationale in the same
  file, and it stands without the smoke.
- **What the smoke legitimately established is wiring:** dequant calls 8/8 per forward, the base arms agreeing per row,
  finite non-zero gradients, the lora_A positive control firing. That is correctness, which the policy allows.
- **For later lanes:** a band's placement comes from a reading on the target card or a rented microbench there, never
  from an A2000 timing, VOID or not.
