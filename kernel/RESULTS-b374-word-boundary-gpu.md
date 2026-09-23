# RESULTS B374 — the word-addressed NF4 decode routes past THEIR 2^31 boundary, on an RTX 5090 (#374)

Read 2026-09-23 against `kernel/PREREG-b374-word-boundary-gpu.md`, which was registered before the run and
merged in #384 at `94a9ff4`. Run `b374-5090-1`:

- **Box:** Vast instance 52214324, an RTX 5090 at 575 W, driver 580.119.02, with 31,602 MiB free at start.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, gnf4 installed at `94a9ff4`. The runner is experts4bit-qlora
  `bench/b374/` at `cf80b0f`.
- **Time and cost:** 10:34:09–10:37:44Z, $0.0354 against a $0.66 estimate.
- **Teardown:** proven. Destroy returned HTTP 200, the instance is absent, and the list afterwards is empty.

Receipts are in `kernel/receipts-b374/5090/`.

## The instrument engaged as registered

`versions.txt` records the tripwires:

- The installed `nf4_grouped` is the site-packages copy, not the clone.
- The `shipped/` work dir resolves `nf4_grouped` to the installed package, and the `unpromoted/` work dir
  resolves it to the stripped copy.
- The copy has **4 `eid.to` + 2 dot-pad load promotions removed** (`unpromote.diff`).

Each GPU case ran in its own pytest process in both passes, and each asserted its route through
`dispatch_counts()`.

## P1 — the promotion holds past 2^31 words: HOLDS

| case | target expert | route asserted | shipped |
|---|---|---|---|
| wide loads, split 1 | 8192 (base = 2^31 words = 8 GiB) | `scalar` | PASS |
| wide loads, split 4 | 8192 | `scalar_splitk` | PASS |
| dot-pad (the default on ≥ 160-SM parts) | 5462 (base 1 MiB past 2^31 words) | `dotpad` | PASS |
| dot-pad split-K | 5462 | `dotpad_splitk` | PASS |

All four passed and none was skipped. The pure-arithmetic geometry test also passed. Every case read the
true tile to within the 5e-2 tolerance, and the decoy was more than 10× that tolerance away.

## P2 — the test can see the bug: HOLDS

Against the copy with the six promotions stripped, all four cases FAIL, each by reading the decoy at the
int32-wrapped address. There were no faults.

| case | rel vs the true tile | rel vs the decoy at the wrapped address |
|---|---|---|
| wide, split 1 | 1.467 | **1.822e-03** |
| wide, split 4 | 1.467 | **1.822e-03** |
| dot-pad | 1.405 | **2.408e-03** |
| dot-pad split-K | 1.405 | **2.408e-03** |

The unpromoted kernels computed the target expert's offset in int32, wrapped, and read the decoy's bytes.
The cases therefore do straddle the word boundary, which is exactly what the byte-geometry arms of
`test_expert_offset_boundary.py` could not show (#374).

## Decision (as registered): P1 ∧ P2

#374 is closed by option 2, a GPU case that straddles 2^31 words.

- **The word boundary is now observed on silicon.** The wide-load and dot-pad routes' 2^31-word boundary is
  observed on an RTX 5090, for the first time and with a test proven able to fail. This includes dot-pad,
  the shipped default decode route on this class of part.
- **Documentation.** `docs/KERNEL_CONTRACT.md` and `test_expert_offset_boundary.py`'s comment point here.
- **The earlier row is corrected, not rewritten.** `gnf4.kernel.expert-offset-boundary.5090.2026-09-05`
  gains a note: its wide and dot-pad arms straddle 2^31 *bytes*, not their own boundary.
- **Register row:** `gnf4.kernel.word-boundary-wide-dotpad.5090.2026-09-23` (measured).

Bounded: one box, one torch/Triton pair, the NF4 routes only. The MXFP4 and int4-b32 kernels are
byte-addressed, and `test_expert_offset_boundary.py` / `test_offset_boundary_interp.py` cover them at their
own boundary.

One recording note: the launcher prefixes the executor identity with the role, so passing
`POD_LAUNCH_AGENT=CTO/…` wrote `CTO/CTO/…` as `executed_by` in the launch receipt. That receipt stays in the
receipt store, not here, because it carries host paths. This affects the label only.
