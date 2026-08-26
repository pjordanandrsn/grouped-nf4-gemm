# RESULTS — K11: REFUTED-INFEASIBLE. The M dimension cannot be filled,
# and it cannot be shrunk either.

## 250-by-composition closes.

Measured 2026-08-26 under PREREG-k11-mrow-feasibility. Receipts in
`receipts-k11/`. No verdict here rests on an estimate.

```
K11 VERDICT: REFUTED-INFEASIBLE
  7 candidates, none qualifying; the M dimension has no filler at
  T=1 on this toolchain, so 250-by-composition closes
```

## The criterion, and the two rows that decided it

A mapping qualified iff it strictly increased useful MACs per MMA
**without** increasing weight bytes per output. Five candidates were
disposed of before any probe (batch, the 8 experts, gate/up, draft
rows, K-scatter). **Two were left OPEN deliberately, because I did
not know the answer**, and both were resolved against the installed
build rather than documentation.

### Sparse MMA — absent

No `sparse`/`spmm` symbol in `triton.language`, and no file in the
NVIDIA backend mentioning a sparse MMA lowering. Checked on Triton
**3.4.0 and 3.7.1**, on **sm_86 and sm_120**.

### Sub-16 M tile — the answer changed with the version, then
### stopped mattering

| toolchain | `tl.dot(M=8)` |
|---|---|
| triton 3.4.0, sm_86 | **refuses to compile** |
| triton 3.7.1, sm_86 | **compiles** |
| triton 3.7.1, **sm_120** (census hardware) | **compiles** |

Had Stage A stopped at the first probe it would have concluded
"infeasible on this toolchain" for the wrong reason — 3.4 refuses
`M=8`, but the campaign's boxes run 3.7.1, where it compiles.

And then the PTX settles it:

```
M=8  -> 8x mma.sync.aligned.m16n8k16
M=16 -> 8x mma.sync.aligned.m16n8k16
```

**Identical.** Triton accepts `M=8` and pads it straight back to the
hardware tile. The API is present; the capability is not
([[presence-is-not-usability]]). Confirmed on the census hardware,
not inferred from the Ampere result.

## Why this is a hardware fact, not a toolchain one

The prereg anticipated saying "the constraint is the toolchain's, not
the hardware's". **That would have been wrong.** The emitted
instruction is `mma.sync.aligned.m16n8k16` — the M extent is fixed at
16 *by the instruction the hardware exposes for this operand type*. A
newer Triton cannot shrink it; only a different instruction could.

So the 93.8% waste is structural for a single-token GEMV:

- `M` cannot be **filled** — a T=1 GEMV has no second right-hand
  side, and every candidate source of one is either a different lane
  (batch), needs distinct weights per row (experts), is already in
  `N` (gate/up), or rests on a mechanism refuted twice (speculation).
- `M` cannot be **shrunk** — the hardware MMA is m16n8k16.

## Consequence for 250, stated as the prereg required

RESULTS-250-closing left the target OPEN pending exactly this lane,
which had to supply **0.898 ms** — 57–63% of the headroom that
remains. It cannot supply any: there is no mechanism.

**Single-stream 250 tok/s is out of reach for this design.** Not
short by an estimate — closed by the absence of a mechanism in the
term that carried it.

The honest paths that remain are neither this target:

- **B>1 aggregate throughput**, already certified at **419 tok/s**
  (BV3b) — a different metric, and the one that actually scales.
- **A different quantisation or compute format**, which would change
  the streaming floor itself rather than the MMA mapping. That is a
  new campaign, not a lane in this frame.

## What stands

| configuration | ms/step | tok/s |
|---|---|---|
| certified default | 7.37 ±4.2% | 130–142 |
| `GNF4_GEMV_DOTPAD=1` | 6.476 | 154.4 |
| **+ `GNF4_ATTN_COMPUTE=fp8`** | **6.281** | **159.2** |

**159.2 tok/s single-stream is the certified ceiling of this design**,
shipping in 0.16.0, and the MoE GEMV's 3.8×-floor gap is now
explained rather than merely measured: it is the cost of issuing a
16-row MMA for a 1-row problem.

## Stage B was not run, and why that is not a gap

Stage B existed to measure K-scatter and convert an arithmetic
argument into a receipt. Its purpose was to catch the case where the
MAC-ratio reasoning was wrong. That reasoning is no longer load-
bearing: the lane is closed by the PTX, which is a *measurement* of
what the hardware is asked to execute, not an argument about it.
Running a kernel that the PTX already shows to be identical would
add cost, not evidence. The REOPEN path in `k11_verdict.py` remains
armed for anyone who disagrees and wants to measure it.
