# RESULTS — why the fused training path could not be CUDA-graphed, and what it took to fix

**2026-08-14 · RTX A2000 12GB (sm_86) · torch 2.8.0+cu128 · triton 3.4.0 ·
bitsandbytes 0.50.0 · report-only.**
Scope registered pre-data in
[`kernel/prereg_capturability_scope.json`](../../../../kernel/prereg_capturability_scope.json),
OTS-stamped 2026-08-14 while the throughput gate below was still in flight
(calendar-pending; `ots upgrade` once Bitcoin confirms). Because the gate has
not returned, the register is genuinely pre-data for the one number it grades.

> **CAPTURABILITY IS A PRECONDITION, NOT A SPEEDUP.** Nothing in this document
> is a performance claim, and capture success must not be reported as one. See
> [What this does not license](#what-this-does-not-license).
>
> **STATUS: capture-verified, correctness-verified, NOT throughput-verified.**
> The registered regression gate ran **twice** on rented L40S cards and returned
> UNUSABLE (§7) then VOID (§8). The second run completed cleanly and its own
> instrument self-pair — identical code, twice — varied by 11%, against a band
> of 3.3%. **The e2e leg is host-bound (12–25% GPU-busy), so this gate cannot be
> adjudicated on a shared rented pod at all.** §8 lists what would close it.
> A third gate, on a GPU-BOUND cell (§9), **worked as an instrument** —
> same-code self-pair 2.3% where the e2e leg gave 11.2% — and measured the
> change **6.5% FASTER** (median, 4 cells, none slower). Its registered band
> was two-sided, so that reads as **GATE FAILED**; the band was mis-specified
> by me and is not reinterpreted.
>
> **The mis-specification was corrected the legitimate way: a one-sided
> harm-bound re-registration, stamped pre-data, graded on a FRESH run (§10) —
> GATE PASSED.** All 4 cells ≤ the 1.032 bound (median 0.8471, observed ~15%
> faster, reported not claimed), instrument clean to 0.2%. §9's FAILED verdict
> stands in the record beside it. **Throughput: verified not harmed.**

[`FINDING-host-bound-small-batch.md`](FINDING-host-bound-small-batch.md)
recorded a structural asymmetry: the dequant-on-forward baseline captures into a
CUDA graph 4/4 on both rented devices, and gnf4's fused training path fails 8/8.
CUDA's error — `operation failed due to a previous error during capture` — is
`cudaErrorStreamCaptureInvalidated`, which is what a *later* call reports after
an *earlier* illegal one already killed the capture. It never names the
offender, so the cause was left NOT ESTABLISHED with two candidates named.

This establishes it. **Both named candidates were real. Neither was sufficient,
and three further hazards were never named** — one of them on the LoRA path,
which is the path real training uses.

Run on the A2000 because capture is a **boolean**, not a timing: it is a
correctness question, and the correctness testbed answers it for free
(`feedback: benchmark testbed policy`). No number in this document is a timing.

---

## 1 · The instrument

[`bench/phase1/probe_capture_bisect.py`](../../probe_capture_bisect.py). Each
attempt runs in **its own process** — a failed capture poisons the CUDA context,
so a shared process cannot tell a real failure from collateral damage. That
discipline is `probe_cudagraph_feasibility.py`'s and is reused unchanged.

Removals are **memoised hoists**: the offending tensor is built once, during
warm-up, so inside the capture region the call is a dict hit. That isolates a
hazard without touching kernel math or changing any value the kernel reads. It
is a probe device, not the shipped fix.

**Positive controls, because a probe that only ever prints FAIL proves nothing:**

* `D_base` runs through the same `_capture()` and the same isolation and **must**
  succeed. It did, on every run — making the A2000 the third device on which the
  baseline captures cleanly.
* the hazard census is itself positive-controlled against four constructs with
  known verdicts. **This caught a real blindness:** the first version hooked only
  `torch.tensor(..., device=cuda)` and reported "0 host-to-device transfers" for
  a call path doing a pinned `.to()` every call. A census that cannot see a
  hazard reports zero exactly like a clean path does.

## 2 · Which constructs are legal inside a capture

Measured first, before designing any fix — each in its own process, against a
trivial kernel, no gnf4 code involved:

| construct | capture |
|---|---|
| `torch.tensor(list, device='cuda')` | **FAIL** |
| `torch.tensor(list).pin_memory().to('cuda', non_blocking=True)` | **FAIL** |
| pre-allocated pinned → `.to('cuda', non_blocking=True)` | OK |
| pre-allocated pinned → `dev.copy_(host, non_blocking=True)` | OK |
| pre-allocated pinned → `dev.copy_(host, non_blocking=False)` | **FAIL** |
| `torch.arange(..., device='cuda')` | OK |
| `.pin_memory()` with the result discarded | OK |

Two things fall out, and the first one cost a whole iteration of the fix:
**"use pinned memory" is not enough — pinned memory *allocated inside the capture
region* still fails.** The staging buffer has to already exist when capture
starts. And `non_blocking=True` is required, not an optimisation: the blocking
form is `cudaMemcpyAsync` followed by `cudaStreamSynchronize`, and the sync is
the illegal part.

## 3 · The bisection

OLMoE-1B-7B-0924 `gate_up`, `decode_m8` (8 groups × 8 rows), 12 attempts, 12
processes. "hazards" counts host→device transfers that are pageable or blocking;
"syncs" is what `set_sync_debug_mode` reported.

| arm | eids | removals | verdict | hazards | syncs | sites still firing |
|---|---|---|---|---:|---:|---|
| `D_base` | list | — | **CAPTURED** | 0 | 0 | — |
| `G_base` | list | — | FAIL | 8 | 8 | tiles ×6, gemm eids, dgrad eids |
| `G_base` | devtensor | — | FAIL | 7 | **15** | tiles ×6, dgrad eids |
| `G_base` | devtensor | HA | FAIL | 6 | 6 | tiles ×6 |
| `G_base` | list | HB | FAIL | 7 | 7 | tiles ×6, dgrad eids |
| `G_base` | list | HC | FAIL | 2 | 2 | gemm eids, dgrad eids |
| `G_base` | list | HD | FAIL | 7 | 7 | tiles ×6, gemm eids |
| `G_base` | devtensor | **HA + HB** | **FAIL** | 6 | 6 | **tiles ×6** |
| `G_base` | devtensor | + HC | **CAPTURED** | 0 | 0 | — |
| `G_base` | devtensor | + HD | **CAPTURED** | 0 | 0 | — |
| `G_base` | list | all four | **CAPTURED** | 0 | 0 | — |
| `G_full` | devtensor | all four | **FAIL** | 2 | 4 | **`lora_delta_grouped`** |

Receipt: [`capture_bisect/capture_bisect_a2000.json`](capture_bisect/).

The census arithmetic is internally consistent, which is why it can be trusted
to attribute: `build_group_tiles` builds three tensors and is called twice per
step (forward M-tile path, backward dgrad) = 6; the `eids` conversions are one
each; and the devtensor form's **15** syncs are 7 transfers plus exactly **8**
per-element `int()` calls on an 8-group cell.

## 4 · The five hazards, and which were named

| | site | what | named pre-run? |
|---|---|---|---|
| **HA** | `nf4_qlora.py` `FusedGroupedNf4.forward` | `[int(e) for e in expert_ids]` — one D2H sync **per group** on a device tensor | ✅ yes |
| **HB** | `nf4_grouped.py` `gemm_4bit_grouped` | `torch.tensor(expert_ids, device=dev)` on a list — pageable H2D | ✅ yes |
| **HC** | `nf4_grouped.py` `build_group_tiles` | **three** pageable H2D per call, called **twice** per step | ❌ **no** |
| **HD** | `nf4_grouped.py` `dgrad_4bit_grouped` | HB again in the backward; `ctx` always handed it a list | ❌ **no** |
| **HE** | `nf4_qlora.py` `lora_delta_grouped` | two more pageable H2D, **plus** a `repeat_interleave` whose output length is read off a device tensor | ❌ **no** |

**HA and HB pull opposite ways** — HA wants a list, HB wants a tensor — which is
why neither `expert_ids` form captured and why swapping the form was never going
to work. **HC and HD are indifferent to the form**: they key off `sizes`, which
neither candidate touched. That is the answer the directive asked for: with both
named candidates removed, capture still failed, and the third hazard was the
deliverable.

**HE is the one that matters most in practice.** `G_base` is the bare kernel;
`fused_grouped_lora` is what a QLoRA finetune actually calls. Removing all four
of the others still left it uncapturable.

## 5 · The fix (call path only)

No kernel source, no tiling constant, no dispatch threshold, no dtype moves.

* **`to_device_i32()`** — one pinned, async transfer for a batch of small host
  int sequences, staged out of a **persistent** `_PinnedIndexArena`. The arena
  bump-allocates and only rewinds when the stream is not capturing *and* the
  event recorded after the last hand-out has completed, so no slice is ever
  reused while a copy off it could still be in flight, and the several call sites
  inside one step each get distinct bytes.

  **The arena must not grow inside a capture**, because growing means allocating
  pinned memory in the region, which is the construct measured to fail in §2.
  Inside a capture nothing completes, so the bump pointer never rewinds and one
  capture consumes the *sum* of every call in it — a whole-model step is
  hundreds of calls, not the handful a single-projection cell makes. The first
  version of this fix sized the arena for the cell it was tested on, which would
  have passed every test here and broken on a real step. It now defaults to
  1 Mi int32 (4 MiB pinned, once per device) and **refuses to grow while
  capturing, naming itself**, rather than producing the opaque error this whole
  change exists to remove.

  In practice the guard is a backstop and not the mechanism: warm-up runs
  *outside* the capture, where growth is legal, so a real step has already sized
  the arena before capture begins. That is also why the negative control below
  has to hit the guard directly — a step-shaped control cannot reach it.
* **`build_group_tiles`** — one transfer instead of three.
* **`gemm_4bit_grouped` / `dgrad_4bit_grouped`** — a device tensor passes
  straight through; a list is converted **once, at the boundary**.
* **`FusedGroupedNf4.forward`** — keeps `expert_ids` as given. `sizes` stays a
  host sequence by contract (the launch grid comes off it). Backward's
  per-expert fallback loop materialises host ints **once, on its own branch**,
  instead of every forward paying 2·G syncs for a path it usually does not take.
* **`lora_delta_grouped`** — never iterates `expert_ids` in Python; selects the
  surviving groups on device; passes `output_size=` to `repeat_interleave` so it
  no longer has to read a device tensor to learn its own output length.

Transfers per training step, `decode_m8`: **8 pageable-and-syncing → 4 pinned
and async** (list form), **7 → 2** (device-tensor form).

## 6 · Verification

| gate | result |
|---|---|
| capture, shipped path, no probe patches | **6 / 6** — `G_base` and `G_full` in both `expert_ids` forms, a 48-call `G_stack`, and `D_base` still captures |
| 48 fused+LoRA calls in ONE capture | **CAPTURED**, 144 pinned transfers, 0 hazards |
| arena exhaustion guard (negative control) | **REFUSED BY NAME** — not an opaque capture error |
| `pytest` compiled path | **145 passed** |
| `pytest` interpreter contract (separate process) | **18 passed** |
| bitwise A/B, pre-change vs post-change | **26 / 26 tensors bit-identical** |

`G_stack` exists because the single-projection cells the original probe used
cannot distinguish an arena sized for a cell from one sized for a step. It runs
48 fused+LoRA calls inside one capture — a large model's layer count, and still a
floor, since a real step makes two or three projection calls per layer.

The A/B ([`ab_capturability_bitexact.py`](../../ab_capturability_bitexact.py))
covers both `expert_ids` forms across `gemm_4bit_grouped`, `dgrad_4bit_grouped`,
`lora_delta_grouped`, `fused_grouped_lora` forward and input gradient, the
`dgrad_kernel=False` fallback loop, `build_group_tiles` at four `block_m`
values, and the `_PAD_WASTE_LIMIT` skew fallback. `torch.equal`, not a
tolerance: this change may move no value, so a tolerance would hide exactly the
mistake it could make.

## 7 · The throughput regression gate — RAN, and did NOT CLOSE

**RTX L40S (sm_89), SECURE on-demand, 2026-08-14 · torch 2.8.0+cu128 · e4b
0.18.0 published wheel · OLMoE-1B-7B-0924, seq 512, 24 steps/arm, 4 dropped ·
$2.42 billed.** Reduced by
[`reduce_capturability_gate.py`](../../reduce_capturability_gate.py); receipts in
[`capturability_gate/`](capturability_gate/).

Three sweeps on ONE card in one session — `pub1` (published wheel) → `cap` (the
same wheel with only `nf4_grouped.py` and `nf4_qlora.py` replaced) → `pub2`
(published restored). The swap is asserted in both directions off a symbol that
exists only in the changed file, so the A/B cannot silently compare the wheel
against itself. **`pub2/pub1` is the instrument's own self-pair** and it spans
the *whole* run, where `cap/pub1` spans half — so it is a conservative bound on
drift, and it is checked before the gate is read.

### VERDICT: UNUSABLE — this run cannot grade the change

Not a pass and not a fail. Three things stacked up:

1. **`pub2` offload=0 was truncated** by the puller's 140-minute deadline
   mid-`random`, so the instrument comparison covers 6 of 8 cells.
2. **Every `text` cell drifted, in all three sweeps**, blowing the e2e prereg's
   own G1 band (1.00 ± 0.05) — and *monotonically worsening* across the run:

   | sweep | text, offload=1 | text, offload=0 |
   |---|---:|---:|
   | pub1 | 0.9564 | 0.9047 |
   | cap  | 0.9353 | 0.9271 |
   | pub2 | **0.8926** | **0.8778** |

   `random` — which runs *second* in every sweep — is clean throughout
   (1.0010–1.0453). That position dependence is this program's own
   **clock-recovery law**: the first timed cell after a fixture build reads the
   card boosting back up. `--warmup 4` drops steps per *arm*; there is no
   wall-clock warm-up before the *first* arm. Those cells are excluded by a rule
   that already existed, not by one invented here.
3. That leaves **2 clean instrument-vouched arm-cells** (`random`, offload=1),
   where the instrument resolves ±1.5% (0.9869 / 1.0157) but the two arms
   disagree by **8%** (`fast_train` 1.0594, `fast_train_dgrad` 0.9801). A
   uniform call-path change moves both arms the same way. Two cells that
   disagree by more than the instrument's noise do not grade anything.

**Direction of the unclosed evidence, stated because withholding it would be
selective:** across the four non-drifted cells the gate median is **1.0424** —
`cap` reads *faster*, and no cell reads slower than 0.9801. Nothing here
suggests a regression. **That is not a claim of a speedup**: the band exists
precisely to stop a directional read from four cells with an incomplete
instrument, and a fix that "looks fine" is not a fix that passed.

### What DID pass, cleanly

**Frozen-storage integrity, on every resident cell of every sweep including
`cap`: 32 tensors hashed, `frozen_changed = 0`.** The check is applicable there
(`integrity_applicable: true`, non-vacuous), so this is a real result: **the
change does not mutate frozen quantized bytes.** Under offload the driver marks
the check vacuous and the count reads a constant 2 in `pub1`, `cap` and `pub2`
alike — a property of offload staging, not of any arm.

**The leg replicates.** `pub1` text/resident `fast_train_dgrad` reads **4.582**
against the published 4.504 (4090) and 4.746 (H100) — a third architecture and
two e4b minor versions later.

### Two errors of mine this run caught

1. **The reducer read `frozen_changed` as a boolean when it is a COUNT**, and
   flagged 30 healthy offload cells as integrity failures. The receipt carries
   `integrity_applicable` for exactly this reason and the first version ignored
   it. A gate that cries wolf on healthy cells is as bad as one that passes sick
   ones.
2. **The GPU-busy probe's cost was never budgeted, and it cost this run its last
   sweep.** At `m=8` it added `2+2m = 18` steps per arm — ~1080 extra steps
   across three sweeps, roughly 23 minutes — on a run that then missed its
   deadline by less than that. The fractions are stable to the percent, so the
   default is now `m=3` with `--busy-steps` to raise it. **C4 is not free, and
   the leg that folds it in has to pay for it in its wall-clock budget.**

## 8 · Run 2 — complete, and VOID. The instrument cannot resolve the question.

**RTX L40S again, 2026-08-14, 82 min, $1.35.** All six sweeps completed this
time (`--busy-steps 3` recovered ~60 minutes), all twelve cells present.
Receipts in [`capturability_gate_run2/`](capturability_gate_run2/).

### The warm-up did not work, and the data says so plainly

Per-sweep `text` self-pairs, run 2 (with a 1.5 s wall-clock GPU warm-up before
the first arm of every mode, confirmed firing in the log at 1.57–1.58 s):

| | offload=1 | offload=0 |
|---|---:|---:|
| pub1 | 0.8681 | 0.8411 |
| cap  | 0.8992 | 0.8793 |
| pub2 | 0.8795 | 0.8766 |

Run 1, *without* the warm-up, read 0.9564 / 0.9047 on the same two cells.
**The warm-up made it worse, not better** — so "clock recovery" was the wrong
diagnosis. What the numbers actually show is a highly reproducible ~12–16%
slowdown of the reference arm between the first and last arm of the `text`
mode, in every sweep. Reproducible at that tightness is a mechanism, not noise,
and warming the *start* of the mode widens the gap rather than closing it.

### Why the gate is VOID, and it is the instrument's own verdict

With `text` excluded, four `random` cells remain — and the instrument self-pair
on those cells, **running the identical published code twice**, reads:

| cell | pub2/pub1 (same code) | cap/pub1 (the change) |
|---|---:|---:|
| random, offload=0, `fast_train` | **1.0379** | 0.9422 |
| random, offload=0, `fast_train_dgrad` | 0.9915 | 0.9946 |
| random, offload=1, `fast_train` | 1.0313 | 1.0164 |
| random, offload=1, `fast_train_dgrad` | **1.1118** | 1.1080 |
| median | 1.0346 | 1.0055 |

**The two columns have the same spread.** Identical code varies by up to 11.2%;
the change varies by up to 11.1%. A band of ±3.3% cannot be adjudicated by an
instrument whose own noise is ±11%, and the run is VOID on the pre-committed
rule before `cap/pub1` is even read.

Note what is *not* being claimed. The gate median is 1.0055 and nothing suggests
harm — but the instrument median is 1.0346, i.e. **the same-code comparison
drifted further from 1.0 than the change did.** Reading a result out of these
cells in either direction would be reading noise.

### The cause, and it is C4's finding pointing back at C2

These arms run at **12–25% GPU-busy** (§7): the e2e step is host-bound, which is
exactly what the measurement-class label says. A host-bound ratio on a *shared
rented pod* tracks that host's CPU contention, and contention varies over an
80-minute run. The per-sweep self-pairs are clean for `random` (0.9904–1.0374)
because a sweep is short; the cross-sweep comparison spans the whole run and is
not.

**So the quantity this gate is registered on is the one least able to survive
the hardware it can be measured on.** That is a property of the gate's design,
not of the change under test, and no number of re-runs on the same class of box
fixes it.

### What would actually close it — three options, none of them "run it again"

1. **A quiet box.** Bare metal with dedicated CPU (Latitude.sh), where a
   host-bound ratio is not competing with a neighbour. Costlier per hour, and
   the honest fix for a host-bound measurement.
2. **Register the band from the instrument, pre-data.** A band is supposed to
   come from the instrument's measured noise floor; ±3.3% was inherited from
   pods that happened to be quieter. Measuring the self-pair first and
   registering the band from it is correct methodology — but it must be stamped
   **before** the next run's data, or it is a bar moved to fit a result.
3. **Grade the regression where the instrument can see.** The change removes
   *host* work, and the microbench cells at `tokbudget_2048` run both arms at
   96–99% GPU-busy, where the self-pair resolves to ~1%. That measures a
   different thing than "the e2e numbers must not move" and would need its own
   registration — but it is the cell where a call-path change is actually
   resolvable.

Until one of those runs, this change stays **capture-verified and
correctness-verified, not throughput-verified**, and does not merge.

**Spend across both attempts: ~$3.80** (run 1 $2.42, run 2 $1.35, plus wedged
pods that never started a container and were destroyed 404-verified).

---

## 9 · The GPU-bound gate — the instrument works, and the change is FASTER

**RTX A6000 (sm_86), SECURE, 2026-08-14, 8.5 min, $0.08.** Grades
[`kernel/prereg_capturability_gate_tokbudget.json`](../../../../kernel/prereg_capturability_gate_tokbudget.json)
(OTS-stamped **before** this data existed). Receipts in
[`kernel_gate/`](kernel_gate/). `pub` is this branch's merge-base, not current
`origin/main` — main has since moved 8 commits ahead, and using it would have
folded unrelated upstream work into the A/B.

### P1 CONFIRMED — the cell is GPU-bound

All four cells `measurement_class = kernel`, minimum busy fraction 83–97% across
every sweep. `tokbudget_4096` is the kernel-bound cell it was chosen to be.

### P2 CONFIRMED — and this is the methodological result

The instrument self-pair, **identical code run twice**:

| cell | pub2/pub1 |
|---|---:|
| OLMoE `down` | 1.0100 |
| OLMoE `gate_up` | 1.0230 |
| Qwen3-30B `down` | 1.0057 |
| Qwen3-30B `gate_up` | 1.0008 |
| median | **1.0079** |

All four inside 0.967–1.032, worst deviation 2.3%. **The same rented-pod class
that produced 11.2% noise on the host-bound e2e leg produces 2.3% here.** The
diagnosis in §8 was right: the problem was never the hardware, it was grading a
host-bound quantity on a shared host. A GPU-bound cell carries the band.

### P3 FAILED — and the direction is the opposite of harm

| cell | cap/pub1 of `ms.g_a` | reading |
|---|---:|---|
| OLMoE `down` | **0.9382** | 6.2% faster |
| OLMoE `gate_up` | 0.9857 | 1.4% faster |
| Qwen3-30B `down` | **0.8786** | 12.1% faster |
| Qwen3-30B `gate_up` | **0.9313** | 6.9% faster |
| median | **0.9347** | **6.5% faster** |

Ratio is cap-time over pub-time, so **below 1.0 means the changed path is
faster**. Three of four cells are outside the band, every one of them on the
fast side, and no cell is slower. The secondary `d_over_g` view agrees
independently (1.0045–1.1169, median 1.047 — the fused arm gaining on the
baseline).

The effect is far outside the instrument's 2.3% noise, so it is real. And it is
**~20× larger than I predicted**: the prereg estimated 0.1–0.3%, reasoning that
a few small transfers are negligible against a ~150 ms step. That reasoning
counted bytes and ignored the mechanism. Each pageable transfer is
`cudaMemcpyAsync` **plus `cudaStreamSynchronize`** — at these shapes the step
carries roughly eight of them (three tile tensors × two call sites, plus the
`expert_ids` conversions), and each one drains the pipeline and stops CPU
run-ahead. Removing the syncs, not the bytes, is what bought the time.

### The verdict stands as registered, and the rule was mine to get wrong

**P3 registered a TWO-SIDED band with no predicted direction, so a 6.5% median
improvement reads as FAILED, and the stop rule says revert.** That is what the
reducer printed and it is written here verbatim, as the stop rule requires.

**The rule is mis-specified, and that is my error, not a result to be
reinterpreted after the fact.** The purpose it serves — from the scope prereg,
*"a capturability fix that costs throughput is not a fix"* — is one-sided: it
exists to catch harm. I registered it symmetrically anyway. Reading a FAILED
verdict as a pass because the sign is convenient is exactly the move this
program's rules exist to prevent, so it is not made here.

**What this means concretely:** the letter of a stamped gate says revert; the
measured facts say the change is bitwise-identical in output, fixes
capturability, and is 6.5% faster at a kernel-bound size on a clean instrument.
Resolving that is an operator decision, and closing it cleanly needs a one-sided
gate (*"must not be slower than the band"*) stamped pre-data and graded on a
fresh run — not this run re-read under a rule written after seeing it.

That gate is §10, and it was run.

## 10 · The one-sided re-registration — GATE PASSED on a fresh run

**RTX 4090 (sm_89), SECURE, 2026-08-14, 8.5 min, ~$0.10.** Grades
[`kernel/prereg_capturability_gate_oneside.json`](../../../../kernel/prereg_capturability_gate_oneside.json)
— written and OTS-stamped **before this run's data existed** (commit `ef0a8ba`),
with the reducer's `--harm-bound-only` flag verified beforehand to reproduce
§9's two-sided verdict unchanged without it. Receipts in
[`kernel_gate_oneside/`](kernel_gate_oneside/). Design identical to §9; P3
becomes the bound its purpose always was: **`cap/pub1` of the fused arm's own
time ≤ 1.032 on every graded cell, no lower bound.** §9's `GATE FAILED` under
the two-sided band is *not* re-graded and stands in the record beside this.

Two things make this run stronger than a formality:

* **`pub` is current `origin/main`** — the branch was rebased first, so the A/B
  tests exactly the diff the PR merges, including upstream's `_triton_shim` on
  both sides.
* **It ran on a third card class for this gate family** (4090/sm_89, where §9
  was A6000/sm_86), and the within-run design carries across unchanged.

| | P1 (kernel-class) | P2 instrument (two-sided) | P3 harm bound (≤ 1.032) |
|---|---|---|---|
| OLMoE `gate_up` | ✓ | 1.0083 | 0.9327 |
| OLMoE `down` | ✓ | 1.0035 | 0.8375 |
| Qwen3-30B `gate_up` | ✓ | 0.9928 | 0.8568 |
| Qwen3-30B `down` | ✓ | 1.0005 | **0.7328** |
| median | — | **1.0020** | **0.8471** |

**VERDICT: GATE PASSED** — every graded cell far under the harm bound, on an
instrument whose same-code self-pair is clean to 0.2% at the median. The
observed direction: `cap` is **15.3% faster at the median** on this card
(**27% on Qwen3 `down`**), larger than §9's 6.5% on the A6000 — consistent with
the sync-removal mechanism, since the faster card finishes its GPU work sooner
and the ~8 removed `cudaStreamSynchronize` calls were a larger share of its
step. The secondary `d_over_g` view agrees (median 1.105).

**Per the registration, that speed is reported as observed, NOT claimed**: this
gate can only certify absence of harm, and a registered speedup claim would need
its own prereg with a predicted band. The e2e gate
([`prereg_capturability_scope.json`](../../../../kernel/prereg_capturability_scope.json))
stays OPEN — measured twice to be unadjudicable on shared rented pods — and
nothing here closes it.

**With §10 green, the throughput question for this branch is closed under a rule
whose sidedness matches its registered purpose**: the change does not harm
throughput at a kernel-bound size, on two card classes, with clean instruments,
and every output bitwise identical.

## What this does not license

* **Not a speedup, and capture success may not be reported as one.** What this
  removes is an asymmetry: the fused path is now in the same class as a baseline
  that could already be captured. It is not ahead of it.
* **Not a restoration of the leg-4 headline.** The graphed race's verdict stands:
  at training shape the shipped fused path lost to a graphed baseline on both
  devices, and a small-batch speed claim against dequant-on-forward is not
  supported. Replacing it needs a new registered experiment, not this one.
* **Not a usable graph.** MoE routing changes group sizes and the hit-expert set
  every step; a replayed graph re-reads the staging buffer and replays the
  metadata it was captured with. Making a captured fused graph *usable* needs a
  padding or bucketing scheme, with its own prereg, its own padding-waste
  accounting, and its own fidelity gate — none of which is in scope here.
* **No timing claim from this hardware.** The A2000 is a shared production box.
  Which side of a bar a cell falls on is structural and survives contention; a
  ratio measured there does not.

## 11 · The e2e gate on a whole machine — the instrument works, and the gate FAILS

**Vast.ai whole-machine RTX 3060 Ti (24/24 cores, Xeon E5-2650 v4 class), 2026-08-15,
~3.3 h, ~$0.30.** Grades
[`kernel/prereg_capturability_scope_amendment1.json`](../../../../kernel/prereg_capturability_scope_amendment1.json)
(OTS-stamped pre-data at `9b952a1`): hardware rule = every core of the host
belongs to this instance, asserted by the launcher from the API readback; band =
the one-sided harm floor **cap/pub1 of `speedup_vs_reference` ≥ 0.967 on every
clean cell**. Wheels pinned (`gnf4==0.12.0` = the exact merge-base,
`e4b==0.19.1`). Receipts in [`e2e_baremetal/`](e2e_baremetal/).

### Finding 1 — the amendment's bet CONFIRMED: dedicated CPU makes this leg measurable

P2, the instrument self-pair (identical published code, twice, spanning the whole
run) on the clean cells:

| cell | pub2/pub1 |
|---|---:|
| random, off=0, `fast_train` | 0.9976 |
| random, off=0, `fast_train_dgrad` | 0.9928 |
| random, off=1, `fast_train` | 1.0126 |
| random, off=1, `fast_train_dgrad` | 0.9783 |
| median | **0.9952** |

Worst deviation **2.2%**, where the same quantity on shared L40S pods read
**11.2%**. A host-bound step ratio is reproducible when the host is wholly
yours. The two shared-pod verdicts (§§7–8) were about the hardware, exactly as
diagnosed.

### Finding 2 — the first-mode drift is a DRIVER property; the contention hypothesis is dead

The registered side-question resolves against contention: `text` — always the
first data mode — blew G1 in **all six sweeps on a machine with no neighbor**
(reference self-pairs 0.859–0.902), at the same ~10–14% magnitude as on shared
pods, while `random`, always second, stayed clean (0.981–1.026). Per the
amendment's pre-registered reading: the harness needs a **first-mode discard**
(or a mode-order rotation); more warm-up is already falsified. The `text` cells
are excluded by the standing G1 rule, as on every prior run.

### Finding 3 — the GATE FAILS, and the verdict is written as registered

Gate, cap/pub1 of `speedup_vs_reference` on the clean cells:

| cell | cap/pub1 | floor 0.967 |
|---|---:|---|
| random, off=0, `fast_train` | **0.9601** | **BREACH** |
| random, off=0, `fast_train_dgrad` | 0.9842 | ok |
| random, off=1, `fast_train` | 0.9829 | ok |
| random, off=1, `fast_train_dgrad` | 0.9816 | ok |
| median | **0.9822** | — |

> VERDICT: GATE FAILED — harm bound. 1/4 cells with cap/pub1 speedup BELOW
> 0.967 (worst 0.9601). The change costs e2e throughput; per the registered
> stop rule it is reverted.

All four cells sit below 1.0 — the change reads **~1.8% slower at the median**
on this box, and the breaching cell is 4% slower against an instrument that is
clean to 2.2%. This is small, but it is not noise-shaped: four of four below
1.0 is directional.

### The mechanism, and why both gate families are right at once

The same change is **6.5–15% faster where the GPU pipeline is busy** (§§9–10)
and **~2–4% slower where it is idle**. That is one mechanism, not a
contradiction: removing `cudaStreamSynchronize` pays only when there is a
pipeline to keep full. At 12–24% GPU-busy the syncs the old path paid were
nearly free — the GPU was idle anyway — while the new path's extra HOST work
per call (pinned-staging writes, event queries, arena bookkeeping) is pure
added cost on a step that is host-bound by definition, and this box's
E5-2650-v4-class cores price that work high. The capture-legal path was made
**unconditional**, so the e2e leg pays for a property (graph-legality) it never
uses outside capture.

That reading also names the repair, for a FUTURE change with its own gates:
route index transfers through the arena **only when
`torch.cuda.is_current_stream_capturing()`** (with the arena pre-allocated
outside capture), and keep the old pageable path otherwise — the e2e path then
returns to the pre-change code by construction while capturability is
preserved. Not implemented here; the stop rule bars bolting it onto this run.

### Status

* This section records a **FAILED registered gate against code that is already
  on `main`** (merged via PR #85 under the kernel-bound gate and the recorded
  operator adjudication). The registered stop rule says the change is reverted;
  whether to revert `main`, accept the measured trade explicitly, or land the
  capture-conditional repair under new gates is an **operator decision**, and
  this document does not make it.
* Not affected by this verdict: capturability (6/6), bitwise identity (26/26),
  frozen-storage integrity, the kernel-bound gates (§§9–10), and the memory
  results. The trade is real and now measured on both of its sides.

## 12 · The capture-conditional repair — G1–G3 GREEN, and what it costs

Grades
[`kernel/prereg_capture_conditional_repair.json`](../../../../kernel/prereg_capture_conditional_repair.json)
(OTS-stamped before the throughput data existed). The change:
`to_device_i32` takes the pinned-arena path **only when
`torch.cuda.is_current_stream_capturing()`**; otherwise it performs the
pre-change pageable build. The arena is touched on every CUDA call so it exists
before any capture; an oversized capture refuses by name.

### G1 + G2 — capture and values (A2000)

Capture ladder **6/6** with the arena guard refusing by name; bitwise A/B
**26/26** vs `origin/main` (0.13.0); `test_eids_forms` **4/4**; 76 neighbours.
The conditional reintroduced no hazard: the capture discipline's uncaptured
warm-up is what sizes the arena, and `G_stack` exercises exactly that.

### G3 — the e2e triple on a whole machine, PASSED

Two attempts, both instructive:

* **BM2 (3060 Ti 8 GB, ~$0.28): UNUSABLE — instrument casualty.** The
  `batched` arm's ~6.95 GB peak fragmented the 8 GB card (6.43 GiB allocated +
  628 MiB reserved-unallocated, OOM at 504 MiB), and the cascade corrupted
  e4b's enable/disable pairing — caught by the driver's own `patched 0
  modules` guard — and produced off0 artifacts the committed reducer refused.
  The **clean off1 half** already read parity (gate 1.0161/0.9950 vs
  instrument 1.0010/0.9789). Two harness notes recorded: Vast's create-body
  `env` does not reach ssh-launched processes (the launcher now injects
  `PYTORCH_CUDA_ALLOC_CONF` on the launch line itself), and the arm-set's peak
  must fit the card with fragmentation margin.
* **BM3 (A4000 16 GB, 16/16 cores, ~$0.12): PASSED.**

| cell (random) | P2 instrument pub2/pub1 | P3 gate cap/pub1 |
|---|---:|---:|
| off=0, `fast_train` | 1.0196 | 0.9920 |
| off=0, `fast_train_dgrad` | 1.0007 | 0.9997 |
| off=1, `fast_train` | 1.0057 | 1.0084 |
| off=1, `fast_train_dgrad` | 1.0257 | 1.0182 |
| median | 1.0127 | **1.0040** |

> VERDICT: GATE PASSED — all 4 clean cells at or above the 0.967 harm floor
> (median 1.0040), instrument clean on all 4. Cells above 1.032 are reported
> as observed, NOT as a registered speedup claim.

The gate ratios sit **inside the instrument's own spread**: the repaired
uncaptured path is the pre-change path, measured. The `text` first-mode drift
appeared in all six sweeps on this third host too (0.886–0.924) — ten
consecutive drifted first modes across three machines; the first-mode-discard
harness fix is queued as follow-up work.

**A reported nuance:** on the A4000's modern cores the offload=1 cells grade
`measurement_class = kernel` (51–58% busy) where every Xeon-class host read
12–25%. The leg's host-boundness is itself host-speed-dependent; the label
now says so per cell, which is exactly what C4 exists for.

### The registered price, paid knowingly — the kernel-bound claim re-scopes

The spot-check (reported, pre-registered) on **both** cards:

| card | cap/pub (`g_a`) | pub2/pub |
|---|---|---|
| 3060 Ti | 1.0020 / 1.0043 | 1.0018 / 1.0021 |
| A4000 | 1.0033 / 1.0233 | 1.0083 / 1.0250 |

Uncaptured, the repaired path is the pre-change path **everywhere** — which
means **the +6.5–15% uncaptured kernel-bound win of §§9–10 is forfeited by
this repair**, exactly as its mechanism requires. Per the registration, the
kernel-gate result is hereby **re-scoped to captured execution**: under
capture and in graph replay the arena path still runs and §§9–10's mechanism
applies there; outside capture, 0.13.1 behaves as 0.12.0 did. The operator
chose this trade explicitly with the forfeit on the table: e2e integrity as
the default, the win where graphs actually run.

### Net position at 0.13.1

* Capturability: **intact** (6/6, arena guard named).
* Values: **bitwise identical** on every path and every `expert_ids` form.
* e2e throughput: **parity with pre-change, gate-verified** on a whole machine
  (median 1.0040 against a 1.27%-noise instrument).
* Kernel-bound speed: **re-scoped to captured execution**; uncaptured =
  pre-change.
* §11's FAILED verdict stands in the record; this section is its registered
  constructive close.
