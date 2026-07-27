# PREREG — what the #43 kernel fix does to a 235B-shaped decode step

Addendum to `PREREG-gemv-issue-bound.md`, whose confirmation branch requires a
**measured** step-level number rather than #40's arithmetic. Stamped before the
pod is created.

## Design change from the original plan

I intended a real Qwen3-235B run on the e4b offload path. `bench/phase3/
offload_decode_235b.py` is better value: it builds the **exact flagship
geometry** (94 layers, E=128, top_k=8, hidden 4096, inter 1536) with
**synthetic** NF4 bytes in pinned host RAM, real bf16 GQA attention, and the
real fused kernel — so it needs **no 120 GB download** and costs a fraction.

It is explicitly a **different mechanism** from the e4b offload path (#29): it
double-buffers, so transfer is already overlapped. That difference is the point
of the experiment, not a flaw in it.

## Four cells

`{--no-stream, streamed} x {original kernel, #43 kernel}`, same pod, same
process count, median of the harness's own repeats.

## Pre-committed predictions

**The central one:** phase3 is built to hide MoE compute under the copy stream.
If it succeeds at that, a 2× faster expert GEMM buys **little at the step
level** even though the kernel really is 2× faster.

| cell | prediction |
|---|---|
| `--no-stream` (resident, compute exposed) | **1.4–2.0×** — near the kernel's own 2.006× geomean, diluted by attention/norms/router |
| streamed (transfer overlapped) | **1.00–1.20×** — compute is hidden; little survives |

**This is a prediction that the fix does almost nothing on this harness**, and
I am registering it because the opposite outcome is the more interesting one.

- **Streamed ≥ 1.20×** → compute was NOT fully hidden on phase3; the fix helps
  even an overlapped pipeline, and #40's 2.19× becomes more plausible.
- **Streamed < 1.20× while `--no-stream` ≥ 1.4×** → confirms the kernel is
  genuinely faster *and* that phase3 hides it. The step-level prize then lives
  on the **synchronous** e4b path, not here, and #40's 2.19× stays **unmeasured**
  rather than being reported as delivered.
- **`--no-stream` < 1.4×** → the isolated 2.006× does not survive into a full
  94-layer step at all, and #43's landed claim must be narrowed to the kernel
  microbenchmark.

## What this run may NOT claim

It cannot confirm or refute **#40's 2.19×**. That figure came from a decomposition
of the *e4b* step where experts are 71.3% of wall, and this harness is phase3.
Whatever this measures, the honest statement about #40 remains "unmeasured on the
mechanism it was derived from".

## Cost rails

Hard cap **$8**. 2×A100-SXM at ~$2.78/hr; expected ~40 min. Teardown is
evidence-gated and session-independent: the pod is deleted as soon as the JSON
payloads are retrieved, and the delete is verified by a follow-up API read
returning the pod absent. **No bulk deletes** — concurrent Claude sessions share
this account, and one was destroyed that way before.
