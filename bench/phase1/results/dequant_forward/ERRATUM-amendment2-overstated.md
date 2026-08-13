# ERRATUM — amendment 2's premise was stated from one device, and the other
# device refutes it

**What was claimed.** After leg-2 run 2's H100 landed, I concluded that
amendment 2 (raising the iteration ceiling so the registered 250 ms block
target was actually met) had made the instrument worse, and that **"block
length was never the binding constraint; drift between blocks is."**

**What the second device says.** The RTX 4090 — the device amendment 2 was
written *for* — went the other way, and by more:

| device | median \|1 − self-pair\| | live cells |
|---|---|---|
| H100 (sm_90) | 0.0012 → 0.0027 (2.29× worse) | 32/32 → 26/32 |
| RTX 4090 (sm_89) | **0.0220 → 0.0039 (5.6× better)** | **10/32 → 17/32** |

Amendment 2 **worked on the 4090.** Block length *was* the binding constraint
there. It was not on the H100, which was already at ~0.1% and had nothing to
gain from longer blocks, so it collected only the extra drift exposure they
bring. Both facts are real; neither generalises.

**The error was mine, not the instrument's.** I read a general conclusion off
the first device to report and said it before the second existed. That is
precisely what this repo's two-card rule exists to prevent, and the rule caught
it — one device later.

**What this does and does not invalidate.**

- Leg-2 run 2's *numbers* stand on both devices. Nothing about them changes.
- **`kernel/prereg_dequant_forward_interleaved.json` (leg 3) carries the
  overstatement in its `purpose` and `the_evidence_that_block_length_was_never_the_problem`
  fields.** It is OTS-stamped and is **NOT edited** — this repo records gaps in
  distributed artifacts rather than repairing them, because re-stamping
  diverges from what any holder of the original has.
- Leg 3's **protocol is unaffected**. Iteration-level interleaving is sound on
  its own merits, its drift tests stand on synthetic data independent of any
  device, and its registered `P1` prediction — that the interleaved and block
  statistics agree on the H100 and diverge on the 4090 — is *still exactly the
  right falsifiable test*, and is now better motivated than when it was
  written: the two devices demonstrably have different noise composition.
- What leg 3 may **no longer** claim as settled background is that block
  pairing was uniformly the problem. It was the problem on one device and not
  the other, and leg 3's results doc must say so.

**Read this before reading leg 3.**
