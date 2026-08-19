# Tribrid CPU/GPU/NVMe execution engine — architecture notes (gnf4 side)

Pre-work map for Stage 3: where the deadline-aware cold path hooks into
this repo, which seams already exist, and which of the directive's
mechanisms turn out to be *bookkeeping* rather than new machinery.
Runtime-side notes (scheduler policy, destination selection, placement
solver) live in `experts4bit-qlora/docs/hybrid/ARCHITECTURE-NOTES.md` per
the standing placement rule: kernels + tier mechanics → gnf4, router /
solver / executor → e4b.

Lineage: `docs/cold-engine/ARCHITECTURE-NOTES.md` (Stage 1, G0–G5) and
`experts4bit-qlora/bench/hybrid-g9/HANDOFF.md` (Stage 2 scoreboard —
G8 B=8 CLOSED at 0.978 on reference silicon, B=16 OPEN at 0.698).

## The three concepts Stage 3 must keep apart

Stage 2 could conflate them because a tier assignment implied both a
location and an executor. Stage 3 cannot:

| concept | question | owner |
|---|---|---|
| **placement** | where should this expert live over a long horizon? | `placement.solve_placement` (e4b) |
| **residency** | where do valid packed bytes exist *right now*? | `ColdTier` + the VRAM slot map (gnf4) |
| **execution destination** | which processor runs *this* invocation? | `_HybridTier._cold_contrib` (e4b) |

An expert may be NVMe-placed, DRAM-resident, and GPU-executed in the same
step. Today the code cannot express that sentence.

## What the objective becomes

Stage 2B's objective is `T_cpu ≈ T_gpu` over *resident* work, and it is
implemented literally: `placement.py:222-232` accumulates `t_gpu` /
`t_cpu` completion-time proxies and the greedy assigns each expert to
whichever side finishes sooner. **NVMe-placed experts are assigned in the
same loop's `else` branch and contribute to neither proxy** — cold mass is
invisible to the balance objective. That is the hole Stage 3 fills, and it
is why "use NVMe more" is the wrong framing: the scheduler currently has
no term for cold work at all, so it cannot trade against it.

Stage 3 target:

    T_layer = max(T_gpu_resident, T_cpu_resident, T_cold_contributions)

with the first-class metric

    hide_ratio = 1 - (cold time exposed on critical path / cold isolated time)

Both `hide_ratio` and raw cold latency get reported; a run that moves more
bytes to expose fewer is the expected shape of a win.

## Seam map — verified against the tree at this commit

| directive workstream | seam that exists | what is missing |
|---|---|---|
| 1. dual-destination cold I/O | `_HybridTier._cold_contrib` (e4b `hybrid.py:413`) already splits routed rows into `nr` (NVMe) / `dr` (DRAM) and already picks a destination for `dr`; **`ColdCpuView` (this repo) now supplies the CPU destination's kernel-shaped bytes** | the e4b dispatch still sends `nr` to GPU unconditionally — the deadline branch is the next PR |
| 2. cold CPU cache | `ColdTier` (`nvme_residency.py:52`) IS a bounded packed-byte DRAM row cache with LFU+LRU, demand-window protection, publish-after-fill; **reclaimable residency + slot generations landed** | NUMA placement; direct NVMe→stack scatter landing |
| 3. GPU cold staging | `ArenaExpertSource._staging` / `_scatter_views` / `_to_device` (`arena_experts.py:211-298`) is NVMe→pinned→H2D today | `_to_device` allocates per fetch — needs a bounded runtime-owned staging pool |
| 4. deadline-aware scheduler | `offload_rows` / `dram_thin` thresholds in `_cold_contrib` are a *static* destination rule | no time-to-contribution estimate, no queue depth, no lead-time issue |
| 5. timing model | `placement.py` `cpu_us_fixed` / `cpu_us_per_row` — the B=16 row-scaling law already in the cost model | no storage / H2D / backlog terms |
| 6. promotion / demotion | `ColdTier` eviction; `Mxfp4NvmeResidency._invalidate` (`mxfp4_residency.py:532`) | no DRAM→VRAM path that skips the arena |
| 7. instrumentation | `ColdTier` already separates `demand_fill_ns` / `demand_wait_ns` / `spec_fill_ns` — critical-path vs overlapped is already a distinction this code makes | no per-stage timestamps, no counterfactuals |

## Three findings that change the implementation estimate

### 1. Dual-destination dispatch already exists — for the wrong tier

`_cold_contrib` computes `gpu_route` for DRAM-resident experts from
`_gpu_only`, `dram_thin`, and an `offload_rows` rows-per-unique-expert
threshold, then calls either `_dram_on_gpu` or `_dram_contrib`. That is
structurally the destination decision the directive asks for, one tier
early. Stage 3 is therefore not "add a second destination"; it is
**extend the existing branch to cold rows and replace the threshold with
a deadline estimate.**

The thresholds are also a standing receipt for prediction 4 (the optimal
destination flips with batch): the serving playbook ships
`offload_thin_uniq=4` with "8 helps B=16" — the flip point already moves
with batch, measured, before any deadline model exists.

### 2. Reclaimable DRAM residency is *bookkeeping*, not a rewrite — what was
missing is the event, not the bytes

`ColdTier._claim_slot` evicts like this:

    old = self._key_of[slot]
    if old is not None:
        del self._slot_of[old]
        self._key_of[slot] = None         # unpublish before refilling
        self.evictions += 1

It **never zeroes the row**, and it runs *inside the claim* — i.e. what the
tier called "eviction" was already the directive's **physical overwrite**,
not its logical eviction. The directive's logical eviction had no
counterpart at all: every mapped row was equally resident, protected or
not, and a hit on an unprotected row was already free. So the reclaimable
*behaviour* was partly there and entirely unmeasurable — there was no
moment to timestamp, so `P(reuse before overwrite | logical eviction)` had
neither a numerator nor a denominator.

What shipped therefore adds the event and the accounting, not a new
allocator:

- `protected_rows ≤ hot_rows` — a capacity-ownership budget *within* the
  pool. Rows beyond it are RECLAIMABLE: mapped, readable, and first in
  line to be overwritten (`_victim`'s new leading rank term).
- `_reclaimable[key] = clock` at demotion — membership is the state, and
  the tick is what makes eviction→overwrite a measurable interval.
- resurrection on any hit that removes a key from that map: metadata only,
  no I/O, counted separately for demand and speculative callers.
- `_gen[slot]`, bumped on every claim, so a held `(slot, generation)` can
  be checked by `validate()` rather than trusted. An expert id alone never
  proved a slot's contents; now nothing has to pretend it does.

Defaulting `protected_rows` to `hot_rows` leaves the reclaimable set
permanently empty and every path identical to the pre-Stage-3 tier — R5
(`T_reclaimable ≤ T_hard_eviction`) holds by construction at the default
and costs one dict lookup away from it.

### 3. The VRAM side already has both halves the directive says to build

The directive warns against implementing VRAM resurrection over
`del tensor; torch.cuda.empty_cache()`. This repo never did:

- `Mxfp4NvmeResidency` (`mxfp4_residency.py:341`) owns a fixed `k_slots`
  device slot arena and carries `_invalidate` (`:532`) precisely because
  *a slot's address does not identify its contents* — the address-vs-
  contents lesson is already load-bearing code. Reclaimable VRAM is the
  same fact read the other way: record which expert those bytes still
  are, and a re-route before overwrite is a zero-copy hit.
- `RowPool.demote_head` / `settle` (`row_pool.py:150,191`) is already the
  RETIRING→RECLAIMABLE state machine: the device row remains the source
  of truth until a recorded CUDA event is observed complete by a
  *non-blocking* query, then ownership flips. The directive's "becomes
  reclaimable only after the relevant CUDA event completes" is this
  method, verbatim.

So the VRAM state machine is an extension of two existing mechanisms, not
a new allocator.

## Format neutrality — where the boundary already is

`ColdTier` is format-blind by construction: it moves opaque `row_bytes`
keyed `(layer, expert)` and never interprets them; `segment_into` /
`segment_tensor` (`nvme_residency.py:523,600`) are the format-aware layer
above it. **The cold scheduler must stay above that line** — it schedules
packed row objects; NF4 vs MXFP4 lives below, in `cpu_grouped` and the
GPU kernels. NF4 first (the CPU path is mature); MXFP4 immediately after
via the segment swap already named as open work in the Stage-2 handoff.

## Correctness — the five equivalences and what already oracles them

No new numerics. A cold expert executes the same packed bytes through the
same kernel as the resident path; relocation must never reinterpret.

| equivalence | oracle |
|---|---|
| NVMe→CPU vs resident CPU | `cpu_grouped.ref_gemv_grouped` / `ref_ffn_grouped` — exact equality, locked summation order |
| NVMe→GPU vs resident GPU | `nf4_grouped.dequant_ref` + the existing grouped-GEMM tests |
| cold-cache reuse vs fresh read | byte equality of the row against `ArenaReader.read_row` |
| CPU-cache→VRAM promotion vs arena→VRAM | byte equality of the staged row; the arena manifest is authoritative |
| dynamic destination vs fixed destination | logits equality per step under a fixed routing trace |

Determinism rule is unchanged and now carries a sharper edge: the
*destination* may vary with load, so per-backend reduction order must
make CPU and GPU contributions combine identically regardless of which
path supplied a given expert. The deterministic per-token combine
(e4b #151, unique `(token, slot)` writes + one fixed-order sum) is what
makes that hold; Stage 3 must not add a combine that depends on arrival
order.

## Open decisions (resolve in the first PR)

1. **Where the reclaimable state lives.** Extending `ColdTier` in place
   keeps one vocabulary (the invariant-8 argument that produced
   `RowPool`); a wrapper keeps the read-only tier simple. Leaning
   in-place, because `_victim`/`_claim_slot` is exactly where the
   decoupling has to happen and a wrapper cannot see it.
2. **Generation counters on `ColdTier` slots** — needed for reclaimable,
   harmless without it. Ship them with the state machine, not before.
3. **Staging pool ownership.** `ArenaExpertSource` allocates device
   tensors per fetch (`arena_experts.py:278`); the bounded staging pool
   could be a `RowPool` partition instead of a third allocator. Prefer
   `RowPool` unless its append-only/head-demoting contract fights the
   random-slot access a staging pool needs — it probably does, which is
   the Phase-10 "random free / reuse" extension that class already names
   as deliberately unshipped.
4. **Which repo owns the deadline estimator.** The measured constants are
   gnf4's (calibration blob); the policy is e4b's. Split at "gnf4 exports
   a cost function, e4b decides" so the estimator stays testable without
   a model.

## Stop conditions (inherited, unchanged)

Any invariant requires violation · determinism unachievable in a phase ·
a dependency forces a weight-format change · a cold path would have to
reinterpret or requantize packed bytes to work. Halt and report; do not
improvise.
