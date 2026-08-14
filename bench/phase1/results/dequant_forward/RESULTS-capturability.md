# RESULTS — why the fused training path could not be CUDA-graphed, and what it took to fix

**2026-08-14 · RTX A2000 12GB (sm_86) · torch 2.8.0+cu128 · triton 3.4.0 ·
bitsandbytes 0.50.0 · report-only.**
Scope registered pre-data in
[`kernel/prereg_capturability_scope.json`](../../../../kernel/prereg_capturability_scope.json).

> **CAPTURABILITY IS A PRECONDITION, NOT A SPEEDUP.** Nothing in this document
> is a performance claim, and capture success must not be reported as one. See
> [What this does not license](#what-this-does-not-license).

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
| capture, shipped path, no probe patches | **5 / 5** — `G_base` and `G_full`, both `expert_ids` forms, and `D_base` still captures |
| `pytest` compiled path | **145 passed** |
| `pytest` interpreter contract (separate process) | **18 passed** |
| bitwise A/B, pre-change vs post-change | **26 / 26 tensors bit-identical** |

The A/B ([`ab_capturability_bitexact.py`](../../ab_capturability_bitexact.py))
covers both `expert_ids` forms across `gemm_4bit_grouped`, `dgrad_4bit_grouped`,
`lora_delta_grouped`, `fused_grouped_lora` forward and input gradient, the
`dgrad_kernel=False` fallback loop, `build_group_tiles` at four `block_m`
values, and the `_PAD_WASTE_LIMIT` skew fallback. `torch.equal`, not a
tolerance: this change may move no value, so a tolerance would hide exactly the
mistake it could make.

## 7 · Still open — the throughput regression gate

**The registered hard gate is NOT yet met.** `fast_train` and `fast_train_dgrad`
must reproduce their measured e2e ratios within the registered self-pair band
**0.967–1.032** on at least one device before this merges. That is a *timing*
measurement, and the A2000 is a correctness testbed on which no timing claim can
be made, so it needs a rented card. Until it runs, this work is
**capture-verified and correctness-verified, not throughput-verified.**

The change should be neutral-to-positive — it strictly removes transfers and
syncs — but "should be" is not a measurement, and a capturability fix that costs
throughput is not a fix.

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
