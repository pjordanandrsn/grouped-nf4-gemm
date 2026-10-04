# K29 — what does a pinned host byte cost a cgroup v2 container? The release gate for #457's pinned-tier model (registered 2026-10-04, before any K29 data)

Lane number claimed by `prereg/k29` (pushed 2026-10-04T15:09:54Z). Issue: #71. Follows `kernel/receipts-71/` and #457.

## Why this lane

#457 replaces `PINNED_ROW_FACTOR = 1.9` with an exact model: a pinned tier costs `pow2ceil(hot_rows × stride + 4096)`,
because PyTorch's caching host allocator rounds each pinned request up to a power of two. On cgroup v1 (driver
575.64.05, torch 2.8.0+cu128, RTX A2000) the model holds to 0.5 %: pinned 340 / 700 / 1359 / 2100 / 3000 MB were
charged 514.5 / 1028.7 / 2057.1 / 4113.8 / 4113.8 MB, and power-of-two pinned requests and every pageable size cost
1.0043 / 1.002 per byte. That agrees with #71's cap ladder (1.004).

The model hands back **more** rows than 1.9 at most budgets. That is safe only if a page-locked byte costs about one
byte in the cgroup regime the tier actually runs in. Rented boxes are cgroup **v2**: both of lane P55's Vast 5090s
read `/sys/fs/cgroup/memory.max`. v2's per-byte charge has never been measured. #71's blocker was that a cap ladder
needs a delegated cgroup. This lane does not need one: it reads the container's own `memory.current`, the instrument
calibrated against the cap ladder on v1 in `kernel/receipts-71/`.

**#457 lands on main unreleased. Its release waits for this reading.**

## Subject and method

- **One rented box:** an RTX 5090 (Vast verified/secure, the launcher's policy class), image
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8.0+cu128). Nothing is installed; the probe needs only torch.
- **The probe:** `kernel/receipts-71/pinned_charge_probe.py`, staged byte-identical as
  `experts4bit-qlora bench/k29/pinned_charge_probe.py` (sha256 in that lane's `staged.sha256`).
  - Each (mode, size) runs in a fresh process: build the CUDA context, read the cgroup charge, allocate N MB and touch
    every byte, then read the charge again.
  - Pinned allocations use `torch.empty(N, uint8, pin_memory=True)`; pageable ones use the same call without pinning.
- **Sizes (MB):** 0, 340, 512, 700, 1024, 1359, 2048, 2100, 3000, 4096. Two repetitions, both modes.
- **Forensics:** the cgroup mounts, `/proc/self/cgroup`, `memory.max`, the driver, GPU, kernel and torch.

## The rule (`experts4bit-qlora bench/k29/k29_reduce.py`)

For every pinned row with N > 0, `r = charged / pow2ceil(N)`. The probe asks for exactly N, so no pad applies.

The verdict is the first of these that applies:

1. **VOID** if any of:
   - the box is not cgroup v2 (no `cgroup2` mount, or no readable `memory.current`); this is STOP-1, and the class
     was not drawn;
   - any child process failed;
   - a pinned row reports `is_pinned` False;
   - the **instrument control** fails: the pageable fit's slope is outside [0.95, 1.10], or its R² is below 0.99.
2. **CONFIRMED** if every pinned row's `r` is in **[0.97, 1.06]**. The power-of-two model holds on v2.
3. **PREMIUM** if the median pinned `r` is above 1.06. v2 charges page-locked memory more than its size.
4. **MIXED** otherwise.

**Why these bars.** On v1 the rows read r = 1.004–1.005. 1.06 allows ~6 % for allocator metadata and noise on another
kernel and driver. 0.97 tolerates a charge read a little early. Below that, the instrument is suspect, which the
pageable control would also show.

## Consequences, registered now

- **CONFIRMED:**
  - grouped-nf4-gemm's next release ships #457's model;
  - `kernel/receipts-71/` and the docstring add the v2 reading;
  - #71 closes.
- **PREMIUM:** #457's default is withdrawn before any release. `capacity_for_bytes(pinned=True)` divides the rounded
  model's budget by the measured v2 median `r`, or the old 1.9 returns, whichever is smaller. #71 records v2 as
  stack-dependent.
- **MIXED or VOID:** nothing is released. A rerun is allowed, inside the ceiling, only if the cause is a harness
  fault or STOP-1.

## Predictions

Written after `kernel/receipts-71/` (v1). Not blind to v1; blind to v2.

| # | prediction |
|---|---|
| Q1 | the drawn box is cgroup v2 |
| Q2 | the pageable slope reads 1.00 ± 0.01 with R² ≥ 0.999 |
| Q3 | every pinned `r` is in [1.00, 1.01] |
| Q4 | the verdict is CONFIRMED (about 85 %), MIXED (10 %) or PREMIUM (5 %) |

## Budget and STOP rules

- **Run.** One box, a 0.5 h guard, at the RTX 5090 policy rate of $0.85/h. The launcher's estimate is about $1.53
  including its default download allowance. The real cost should be a few minutes of GPU time, because nothing is
  downloaded.
- **Ceiling.** The lane ceiling is **$3.00** (the reading and one rerun), inside the owner's standing $15 no-ask tier.
  No proving run is needed: the guard is ≤ 1 h.
- **Stops.**
  - **STOP-1:** a box that is not cgroup v2 is recorded, and the verdict is VOID.
  - **STOP-2:** a VOID or MIXED reading is not retried inside the same launch.
  - **STOP-3:** the driver refuses a dirty tree, or a staged file that differs from its pin.
- **Rehearsal.** Before any rental, the box script runs end to end on the QNAP A2000. That host is cgroup v1, so
  STOP-1 must fire there. The rehearsal proves the harness, the probe, the markers and the reducer's VOID path.

## What this lane cannot say

- Nothing about cgroup v2 on other kernels or drivers beyond the drawn box.
- Nothing about allocators other than torch 2.8's caching host allocator.
- Nothing about the tier's speed.
