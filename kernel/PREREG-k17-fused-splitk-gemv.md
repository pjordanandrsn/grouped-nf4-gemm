# Pre-registration — lane K17: fold the split-K reduce into the int4-b32 GEMV's own launch (registered 2026-09-21, before any kernel code)

Owner directive (Jordan, 2026-09-21): *"main thing needed is throughput work."* Record: experts4bit-qlora#652.

## What the census says, read from receipts (not a guess)

Lane P54 (experts4bit-qlora `bench/p54/RESULTS-p54.md`, one RTX 5090, receipts in-tree) censused Qwen3-30B-A3B's int4 serving stack with `--replay-profile-out`. In the fused-q/k/v arms — the stack that ships at B=1 from 0.36.4 — the second-largest *separable* row after the expert GEMV itself is the GEMV's own split-K reduce:

| batch | `_gemv_int4_b32` | `_reduce_partials` | reduce share of step | reduce calls/step |
|---|---|---|---|---|
| B=1 (3.693 ms) | 1.402 ms, 192 calls | **0.300 ms**, 192 calls | **8.1 %** | one per GEMV call |
| B=16 (11.197 ms) | 6.386 ms, 96 calls | **0.324 ms**, 96 calls | **2.9 %** | one per GEMV call |

`gemv_int4_b32` (`kernel/int4_b32.py`) launches two kernels per call: `_gemv_int4_b32` stores fp32 partials at `part[(sk*R + e)*N + n]` over a grid `(cdiv(N,128), R, SK)`, then `_reduce_partials` sums the `SK` partials in `tl.static_range` order and casts to bf16. On the 5090 (170 SMs) the planner picks **SK = 16 on the expert gate/up (N=1536, K=2048), SK = 6 on the expert down (N=2048, K=768), SK = 8 on `q_proj` / fused qkv (N=4096 / 5120, K=2048) and SK = 16 on `k_proj` / `v_proj` (N=512) and `o_proj` (N=2048, K=4096)** — and the R term does not collapse it on this SM class — so the reduce is never a no-op launch: every call pays it.

**The fix already exists in this package.** `int4_smallm._gemm_int4_b32_smallm` (K16) writes the same kind of fp32 partial, then the LAST-arriving program of each column block (`tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")`, `prev == SK - 1`) sums the `SK` partials **in split order**, stores bf16, and re-arms the counter with `tl.atomic_xchg(cnt_ptr + pid_n, 0)` — one launch, deterministic, capture-legal with a preallocated `cnt`. K17 ports that epilogue to `_gemv_int4_b32`.

## Design (what K17 builds)

- `_gemv_int4_b32` gains a `cnt_ptr` and an `out_ptr` beside `part_ptr`, and `FUSED_REDUCE: tl.constexpr`. With `FUSED_REDUCE=0` the kernel is byte-for-byte today's (partials only). With `FUSED_REDUCE=1`: store the fp32 partial exactly as today, `tl.debug_barrier()`, `prev = tl.atomic_add(cnt_ptr + (e * cdiv(N,BN) + pid), 1, sem="acq_rel")`; if `prev == SK - 1` the program sums `part[(s*R + e)*N + n]` for `s in range(0, SK)` — **the same split order `_reduce_partials` uses** — casts to bf16 once, stores `out[e*N + n]`, and `tl.atomic_xchg`es its counter back to 0. One counter per (row, column block): `cnt [R * cdiv(N, BLOCK_N)] int32`, zeroed once at allocation.
- `gemv_int4_b32(..., part=None, cnt=None, out=None, fused_reduce=None)`: `fused_reduce` defaults to the env `GNF4_GEMV_FUSED_REDUCE` (`1`/`0`; unset = the decision rule's outcome, initially **`0` = today's path** until the lane reads). The fused path allocates `cnt`/`out` when not given (the graph private pool, the certified capture pattern the consumer already relies on for `part=None`); callers that preallocate `part` under capture preallocate `cnt` and `out` the same way. `reduce_partials` stays exported for the unfused path and any external caller.
- **Not in K17-A:** folding the activation quantise (`_quant_x_rows`, 0.153 / 0.318 ms at B=1 / B=16) into the GEMV's K loop. The per-32-block scale is a local max the kernel could compute inline (`s = max|x|/127 + 1e-12; q = floor(x/s + 0.5)`), which would remove one more launch per call and be bit-exact by construction — but it multiplies the quantise work by the number of column blocks (32 programs per row on N=4096) and changes the `xq/xs` contract every caller shares. It is **K17-B**, registered here only as a candidate: it runs only if K17-A's P1 and P2 hold, with its own prediction added by amendment before it is measured.

## Registered predictions (falsifiable)

- **P1 — bitwise identity.** For every shape in the table and every `R ∈ {1, 8, 16, 128}`, `gemv_int4_b32(..., fused_reduce=True)` returns **`torch.equal`** to `fused_reduce=False`: the same fp32 partials are summed in the same static order and cast to bf16 once, so nothing can differ. Under `TRITON_INTERPRET=1` on CPU **and** compiled on the 5090. *Refuted by* a single differing element — which would mean the epilogue is not summing in split order, and the kernel is refused until it is.
- **P2 — per-call time falls by about one launch.** Microbench (K14's instrument: CUDA-graph replay medians, the measured launch floor beside every row) at `R = 1`: the fused path is faster than the two-launch path by **3–6 µs per call** on each of the six shapes (the 5090's launch floor read 4.61 µs in K16; the `_reduce_partials` launch is launch-bound at R=1). *Refuted by* a saving under 2 µs on the majority of shapes (the reduce was not the cost) or by the fused path being slower on any shape (the atomic epilogue costs more than the launch it removes — possible on the SK=16 shapes, where the last arriver reads 16 partials serially).
- **P3 — at R = 128 (the B=16 expert call) the fused path is not slower.** Band: −0.10× to +0.15× of the two-launch time on `gate_up` and `down` — the last arriver now reads `SK` partials per column block that the separate reduce read with a wider grid. *Refuted by* the fused path slower by more than 15 % at R=128 on either expert shape → the decision rule routes `fused_reduce` by R (on at R ≤ 16, off above), never as a blanket default.
- **P4 (model level, the phase-rule requirement)** — with the gnf4 cut installed and `GNF4_GEMV_FUSED_REDUCE=1`, P54's harness on the same box shows the **`_reduce_partials` row at 0 calls/step** at both batches, the timed step falling **≥ 0.15 ms at B=1** (of the 0.300 ms row) and **≥ 0.10 ms at B=16** (of 0.324), and — because of P1 — the generated tokens **identical** to the unfused control's on every sequence at both batches. *Refuted by* any token difference (P1 failed in the wild), or a B=1 saving under 0.08 ms (the launches were overlapped in the graph, as P54 found at B=16 for the K16 row — a kernel-row saving is an upper bound on a step saving).

## Correctness contract (before any perf number)

- P1's `torch.equal` on the six shapes × four row counts, on CPU under the interpreter (`kernel/test_int4_b32_fused_reduce_interp.py`, **added to the explicit test list in `.github/workflows/ci.yml` line 98** — a test file the workflow does not name is inert, and inert and untested are the same fact) and compiled on the lane's card.
- The counter is re-armed after every launch (a second call on the same workspace returns the same bits — K16's `test_counter_rearmed_and_workspace_reuse`, mirrored) and the kernel refuses a `cnt` whose length is not `R * cdiv(N, BLOCK_N)`.
- `FUSED_REDUCE=0` is byte-for-byte today's kernel: the existing `test_int4_b32.py` suite passes unchanged with the flag off, and with it on.

## Decision rule

P1 ∧ P2 ∧ P3 → `GNF4_GEMV_FUSED_REDUCE` defaults **on** in the next gnf4 release, the consumer (experts4bit-qlora) preallocates `cnt`/`out` in `Int4Linear._install` and the hot-residency decode route, and P4 is read in the consumer before any position moves. P1 ∧ P2 ∧ ¬P3 → on for `R ≤ 16`, off above, by the planner, and P4 is read at B=1 only. ¬P1 → **the kernel is refused**; the lane records the differing element and stops. ¬P2 (with P1) → ship as opt-in (it costs nothing and is exact) and record that the reduce launch was not the cost.

## Budget and STOP rules

Development and P1 on CPU (interpreter) cost nothing; the A2000 (sm_86) on the QNAP gives the first compiled read for free. One 5090 lane for the registered rows: `k17-5090`, ceiling $0.65/h, guard 1 h, estimate ≤ $0.65; no proving run needed (guard < 1 h). P4 rides the **next consumer lane that already rents this model** as two extra arms (`GNF4_GEMV_FUSED_REDUCE=0/1` at each batch), not a box of its own. STOP: a card not of the 5090 class; the instrument's launch floor failing to reproduce K16's 4.61 µs within 30 % (the box is not comparable to the K14–K16 rows); any P1 failure on the box before the perf sweep runs.

**Dry-run rule (P54's lesson):** every arm in this pre-registration was checked against the harness's own argument checks before registration — the K14 instrument takes `fn` closures and needs no new flags; P4's arms are P54's `speed_arm` with one extra env var the runner already forwards through `env "$@"`. Nothing here asks a harness for a combination it refuses.

## Receipts

`kernel/receipts-k17/5090/` (rows json, bench log, tests log, versions, forensics, summary) and `kernel/RESULTS-k17-fused-splitk-gemv.md` quoting only the rows; the consumer's P4 rows in the lane that carries them. Amendments dated below, before the data they touch.

## Amendments

(none yet)
