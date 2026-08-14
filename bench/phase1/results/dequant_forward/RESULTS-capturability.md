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
> This change does not merge on the strength of §§1–6 alone.

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
