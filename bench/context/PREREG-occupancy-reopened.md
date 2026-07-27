# PREREG — occupancy and ILP on the post-#43 GEMV (a deliberate re-test)

**This re-tests a hypothesis I already falsified.** #41 swept
`BLOCK_N × warps × split_k` (64 configs) on the decode GEMV and found **8× the
occupancy bought 16%** (1.162× against a 1.2× abandon line), then declared
config tuning abandoned under its stopping rule.

Re-opening is justified **only** because the measured bottleneck has changed,
and #45 documents the change with a profile rather than an argument:

| | old kernel (#41's subject) | post-#43 kernel (this subject) |
|---|---|---|
| sectors/request | 12.08 | **1.95** |
| binding resource | **L1TEX sector throughput** | **L1TEX latency** (Long Scoreboard) |
| SM throughput | 15.80 % | 66.17 % |

More warps cannot help a kernel whose *transactions* are the wall — that is why
#41's sweep found nothing, and #41 was right about its own subject. Latency, by
contrast, is exactly what occupancy and in-flight loads hide. **If this sweep
also finds nothing, occupancy is falsified twice and the remaining ~2.1× is
structural, not configurational.**

## What ncu says is available (#45)

| lever | ncu est. local speedup |
|---|---:|
| remove Long-Scoreboard stalls | **50.95 %** |
| close achieved (51.44 %) vs theoretical (66.67 %) occupancy | 22.84 % |
| raise theoretical occupancy (8 warps/sched vs hw 12) | 33.33 % |

## Levers, and the one #41 did not sweep

1. `num_warps` — warps per block; more warps, more latency hiding.
2. `BLOCK_N` — work per block, hence blocks/SM.
3. `split_k` — extra parallelism for starved grids.
4. **`num_stages`** — Triton's software pipelining depth, currently **3**.
   **#41 did not sweep this.** It is the direct ILP lever: more independent
   global loads in flight before the dependent consume, which is precisely what
   a Long-Scoreboard stall is.

## Pre-committed predictions

Best config vs the shipped default, isolated GEMV, flagship shape, median of 7,
interleaved in one process.

| arm | prediction |
|---|---|
| `num_warps` sweep alone | **1.10–1.35×** |
| `num_stages` sweep alone | **1.10–1.40×** — expected the *stronger* lever |
| best combined | **≥ 1.25×** |

## Decision rules, fixed now

- **≥ 1.25× on BOTH cards** → land it, which means changing `_decode_plan`.
  That plan is tuned and four registered result-sets depend on it, so landing
  additionally requires the census shapes to show **no regression on any shape**
  through the real API.
- **1.10–1.25×** → real but below the bar. **Do not touch `_decode_plan`.**
  Record as a measured partial; the risk to existing results exceeds the gain.
- **< 1.10×** → **occupancy falsified a second time, on a kernel where the
  profiler says it should have worked.** That is the strongest possible signal
  that the remaining headroom is structural (shared-memory staging, or a layout
  change), and config tuning is closed permanently for this kernel.

## Rules carried from today's two failures

1. **Nothing ships on one card's evidence** (#43 shipped a 2.2× regression;
   #44 shipped-nothing but was bit-identical-here / wrong-there). A2000 screens,
   A100 decides.
2. **Screen free, rent only on a pass.** If the A2000 shows < 1.25× the rule
   above already says it does not land, so **do not rent**.
3. The A2000 is shared and noisy (same kernel/shape: 0.687 vs 1.519 ms across
   runs). The 1.25× bar sits deliberately outside that spread; anything inside
   it is not a result.
4. Correctness gate unchanged: agreement with the shipped kernel at the bf16
   floor, on every config that is reported.
