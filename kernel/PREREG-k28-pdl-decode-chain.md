# K28 — does programmatic dependent launch shorten the served B=1 decode layer's gnf4 kernels in a CUDA graph, bit-identically? 48 layers at Qwen3-30B-A3B's shapes, `GNF4_PDL` off against on, on one RTX 5090 (registered 2026-10-04, before any 5090 run)

Lane number claimed by `prereg/k28` (pushed 2026-10-04T04:57Z). Issue: experts4bit-qlora#1015. The switch: #448.

## Why now

**What SC1b found.** experts4bit-qlora's SC1b census (#846, `bench/h2h-2026-10-02/sc1b/README.md`, run `sc1d-5090-3`)
read e4b's B=1 decode step on an RTX 5090 against llama.cpp's:

| engine | summed kernel time | busy time (union) | overlap | consecutive pairs overlapping, one stream | graph span |
|---|---|---|---|---|---|
| e4b | 4.087 ms | 4.087 ms | 0.000 | 0 % of 1,549 | 4.075 ms |
| llama.cpp | 4.135 ms | 2.754 ms | **1.380** | **95.5 %** of 1,109 | 2.711 ms |

The two do the same kernel work, but llama.cpp starts each kernel before the previous one ends, on one stream. That is
programmatic dependent launch (PDL). It hides about 1.38 ms a step, more than the whole B=1 gap. SC1b named PDL as the
lever and did not test it.

**Whose kernels.** 913 of the 1,550 kernels in e4b's B=1 step are this repository's decode-row kernels (the same run's
node census):
- the int4 trio `_quant_x_rows` → `_gemv_int4_b32` → `_reduce_partials`, 192 each;
- `_rope_norm_heads` 96;
- `_rmsnorm_rows` 49;
- `_rmsnorm_resid_rows`, `_router_epilogue`, `_swiglu_rows` and `_combine_rows`, 48 each.

**What the switch does** (#448, `GNF4_PDL=1`, opt-in). On an sm_90+ NVIDIA card those kernels, and three more
that other paths use, launch with Triton's `launch_pdl`. Each kernel's first statement waits for the previous kernel on
the stream to complete, then lets the next one launch. Nothing reads or writes memory before the wait, so what PDL can
hide is launch latency, never ordering.

## The bench (`kernel/k28_bench.py`)

**The layer** follows the served step's order from SC1b's node census, leaving out e4b's attention kernels and the
cuBLAS router GEMM, which are not gnf4's:

1. `rmsnorm_rows`;
2. the trio for qkv (N 5120, K 2048, one row);
3. `rope_norm_heads` on q (32 heads) and on k (4 heads), D 128;
4. the trio for o (N 2048, K 4096);
5. `rmsnorm_resid_rows`, then `router_epilogue` (top-8 of 16);
6. an ATen `index_select` for the expert rows, as served;
7. the trio for gate_up (N 1536, K 2048, 8 rows), then `swiglu_rows`;
8. the trio for down (N 2048, K 768, 8 rows), then `combine_rows`;
9. an ATen residual add, as served.

That is 19 gnf4 kernels and 2 ATen kernels a layer, over 48 layers: **912 gnf4 kernels**. Each layer has its own
synthetic int4 stores (random bytes, seeded), with 16 experts a layer, of which 8 are routed. Each layer reads the one
before, so a stale read would change the final residual stream.

**Graphs.** Each is captured once from its own build:
- `chain_off`, `chain_on` and `chain_off2`: the layer as above. `chain_off2` repeats `chain_off`, as the instrument.
- `glued_off` and `glued_on`: the same plus two ATen kernels a layer where the served step leaves gnf4. One is an
  attention stand-in (a copy); the other is the router GEMM (`torch.mm`), whose logits the epilogue reads. These are
  reported, not ruled on.

**Timing.** 20 warm rounds, then 200 rounds. Each round replays all five graphs once, in an order that reverses every
round. Each replay is timed by CUDA events and synchronised. The bench takes medians per graph.

**Checks:**
- **Bitwise.** Every recorded output of `chain_on` and `chain_off2` must equal `chain_off`'s, and `glued_on` must equal
  `glued_off`. The recorded outputs are each layer's residual stream and k, the router's three outputs, and the glued
  router logits. The eager chain must equal itself with the switch off and on, and must equal `chain_off`.
- **Launch accounting.** A Triton launch hook counts each capture's gnf4 launches, and how many carried `launch_pdl`.
- **Engagement probe.** A 20 µs spin kernel releases its dependents at once and stamps the global clock last. A
  dependent stamps the clock before and after its wait. There are five graph replays with PDL and five without.

## The rule (`verdict`, 11-case self-test)

The verdict is the first of these that applies:

1. **VOID.** Any of:
   - a graph's timing is missing;
   - the card is below sm_90;
   - a capture launched other than 912 gnf4 kernels, an on capture launched any of them without PDL, or an off capture
     launched any with it;
   - in some PDL replay, the probe's dependent did not start at least 10 µs (half the spin) before the spin ended;
   - without PDL, the dependent started early in some replay.
2. **FUNCTION_FAIL.** A bitwise check failed, or a probe dependent left its wait before its primary's last stamp.
3. **NOISY.** `chain_off2 / chain_off` falls outside [0.98, 1.02].
4. **LEVER.** The chain saves at least **0.25 µs per gnf4 kernel**, where the saving is `(chain_off − chain_on) / 912`.
5. **NO_LEVER** otherwise.

**Reported:** both ratios, the glued saving per kernel, and every graph's median, quartiles and replay times.

**Why 0.25 µs.** At 0.25 µs, the 913 gnf4 kernels of the served B=1 step would save 0.23 ms, 5.6 % of SC1b's 4.075 ms
span, if the saving transferred whole. The served step interleaves about 640 kernels that are not gnf4's, and each one
breaks the overlap on both its edges. At half the transfer the gain is about 2.8 %. That is about the smallest a served
lane reads above its ±1–2 % self-pairs (P111's read 0.997–1.002).

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | the probe engages: with PDL the dependent starts at least 15 µs before the 20 µs spin ends in every replay, and without PDL at or after it |
| Q2 | every bitwise check holds |
| Q3 | the accounting reads 912 gnf4 launches per capture: all 912 with PDL in the on captures, none in the off ones |
| Q4 | the chain saves 0.3–1.2 µs per gnf4 kernel. llama.cpp hides about 1.24 µs a pair (1.38 ms over 1,109), and e4b's in-graph kernels average 2.6 µs (4.075 ms over 1,550), so a dependent's launch and ramp are a large share |
| Q5 | the glued graph saves less per gnf4 kernel than the chain does, because an ATen kernel is not a programmatic dependent and costs the overlap on both its edges |
| Q6 | `chain_off` replays in 1.5–3.0 ms (1,008 kernels; SC1b's 1,550-kernel step spans 4.08 ms) |
| Q7 | the verdict is LEVER, at about 65 % |

## Consequence, registered now

- **LEVER.**
  - experts4bit-qlora registers a served lane: `GNF4_PDL=1` on its default decode step, read for tokens (bitwise) and
    decode tok/s at B=1 and B=16.
  - That lane also decides whether e4b's own decode kernels (the KV append and the paged decode) get the same preamble.
  - `GNF4_PDL` stays off by default here until the served lane reads.
  - The CHANGELOG records K28's number as a microbenchmark, not a served speed.
- **NO_LEVER.**
  - `GNF4_PDL` stays opt-in, and the CHANGELOG records the reading.
  - SC1b's overlap gap is then not explained by PDL on gnf4's kernels as they stand. The next look is how llama.cpp
    places its triggers, not this switch.
- **FUNCTION_FAIL.** The switch is withdrawn until the difference is explained, and nothing else moves.
- **VOID or NOISY.** No consequence; an amendment or a rerun inside the ceiling.

## What was seen before this page (stated, not hidden)

Everything below ran on the NAS RTX A2000 (sm_86), in a throwaway `pytorch:2.8.0-cuda12.8` container with torch 2.8.0
and triton 3.4.0, the rented boxes' pair. The A2000 is correctness-only and cannot run PDL, so none of it is a reading.

- **The switch's checks** (#448's description):
  - the off path's PTX is identical to main's on sm_86 and, cross-compiled, on sm_120;
  - with the switch on, every kernel's sm_120 PTX has `griddepcontrol.wait` ahead of every global load and store;
  - a mutation moving the wait below a load is caught twice.
- **A bug they found.** Triton 3.4.0's own `gdc_wait` / `gdc_launch_dependents` wrappers do not compile, so the switch
  emits the PTX itself.
- **This bench, rehearsed on the A2000** (gnf4 at #448's head `4c49120`). Its 11-case self-test passed. It then read
  **VOID** ("compute capability [8, 6] is below sm_90"), as designed. Up to that verdict, everything but the PDL arms
  ran:
  - every capture launched 912 gnf4 kernels, none with PDL, and no other Triton kernel;
  - every bitwise check held;
  - without PDL the probe's dependent started at or after the spin's last stamp (−1024, 0, 0, 0, 0 ns), and every wait
    was correct;
  - the medians were `chain_off` 9.4809, `chain_on` 9.4822, `chain_off2` 9.4884, `glued_off` 9.7872 and `glued_on`
    9.7884 ms, which puts the instrument at 1.0008;
  - the residual stream's mean |x| after 48 layers was 1049, finite;

  Those timings are a shared card's and say nothing about the 5090.

The 5090 has not run any of this.

## Box and cost

- **`k28-5090-<n>`:** one RTX 5090. **Guard 0.5 h at ≤ $0.75/h (≤ $0.375)**, so there is no proving rental.
- **Driven from** experts4bit-qlora `bench/k28/` (K27's runner, with the bench, the tripwire and the premise replaced),
  with no model:
  - the tripwire proves the installed gnf4 carries the switch, Triton can launch with PDL, the card is sm_90+, and gnf4
    reads the switch as active there (rc 9);
  - the premise is `kernel/test_pdl.py` compiled on the card: **23 passed, none skipped**. Its sm_90+ tests are the
    on-card bitwise and engagement checks (rc 23).
- **Lane ceiling $0.75; hard stop $1.00.**

## What this lane cannot say

- **No served speed.** There is no e4b here, none of its attention kernels, and no router GEMM except the glued stand-in.
- **Nothing about the NF4 server.** It decodes through other GEMVs, which do not take the switch.
- **Nothing about B=16 or other cards.** B=16 runs other kernels; sm_90 and sm_100 are other cards.

## Receipts

`kernel/receipts-k28/5090/`: `k28.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `logs/` (the PDL contract and
the bench), the teardown proof and `SHA256SUMS`. `RESULTS-k28-pdl-decode-chain.md` is written from those files.

Amendments, dated, go below this line before any 5090 data is read.
