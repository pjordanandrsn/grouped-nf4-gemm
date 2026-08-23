# RESULTS — the low-G item split: REFUTED as scored, reverted — and the box could not have scored it

Registered in [PREREG-lowg-split.md](PREREG-lowg-split.md) (#236). Run
2026-08-23 on an EPYC 9V74, A/B/A, staged harness, clean rebuilds,
bit-exact passing on the armed split path before any timing. Receipts:
[p4-receipts/](p4-receipts/) (`lg-oldA/newB/oldA2`).

**Scored verdict: REFUTED — the change is reverted in this PR** (the P4
precedent: no unproven churn in the hottest kernel). B1: the N=256
starved cell delivered 617 → 342 µs (ratio 0.554 vs the 0.55 bar — a
hair over even on the favorable min-of-old baseline); the N=768 cell
read 1.18. B2: 23 of the well-fed cells left the ±10% window.

## What the receipts actually show: the environment failed the bars first

The two OLD arms — the same binary, same harness, minutes apart — 
disagree by **up to 55%**, median **15%** across the grid:

| cell | oldA | oldA2 | spread |
|---|---|---|---|
| G=32, rows=32, N=768 | 240.2 | 155.1 | 55% |
| G=2, rows=128, N=256 | 430.6 | 633.9 | 47% |
| G=1, rows=128, N=768 | 685.1 | 1001.6 | 46% |

Most B2 "violations" are this noise wearing a verdict: a 34% *speedup*
appeared at (32, 32, 256), a cell whose machine code is bit-identical
(`rsplit = 1`). Against a ±10% bar and a 0.55 bar, a host with 15%
median A/A spread cannot score the experiment in either direction. The
registered protocol computed REFUTED and the hard stop executes it; the
scientific state is *unproven*, with one likely-real signal (the 1.8×
at the N=256 starved cell exceeds any observed noise) that the rules
correctly refuse to bank.

## The instrument law this buys (the collection's next entry)

**Same-arm gate before any A/B is read**: run the A arm twice first; if
the A/A spread at the gated cells exceeds half the tightest bar, the box
cannot host the experiment — destroy and re-hunt before the B arm ever
runs. (The G3'/G4' burst-and-blip gates screened GPU pathologies; this
is the CPU-microbench sibling. Cost of learning it here: one $0.12 box
and a reverted-but-recorded kernel change.)

## Disposition

Reverted; the prereg, receipts, and the retiling/bit-exact guard remain.
A future attempt re-registers with the A/A gate and — given the N=256
signal — a bar set from measured same-box noise rather than round
numbers. The starvation finding itself (G=1 slower than G=64) is
untouched: it reproduced on both hunt hosts and still names a real
kernel gap; only this fix's certification failed.

Box destroyed; zero instances; ~$0.12; program total ~$1.93.
