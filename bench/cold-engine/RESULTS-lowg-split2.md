# RESULTS — the low-G split re-certification: UNRUNNABLE (3 gate hosts exhausted)

Registered in [PREREG-lowg-split2.md](PREREG-lowg-split2.md). Verdicts computed
by the committed [lowg_cert_verdict.py](lowg_cert_verdict.py); receipts in
[p4-receipts/lowgcert/](p4-receipts/lowgcert/). Total cycle cost ≈ $0.39.

## Verdict

**UNRUNNABLE as registered.** The A/A gate examined three hosts (the
registered cap): one passed the gate and then VOIDed on the A3 drift
check; two were rejected before arm B ever ran. Zero scoreable B arms
were produced. Per the prereg's hard stop, the kernel change is
**reverted** (this PR) and the line closes — a future attempt needs
perf-counter isolation on owned bare metal, not another rented box.

## Host trail (5 rentals, 3 gate hosts)

| # | host | role | outcome |
|---|------|------|---------|
| 1 | EPYC 9654 (CN, fractional, $0.136/hr) | provisioning | image never loaded after 15 min — destroyed, no timing |
| 2 | EPYC 9655 (CA, whole-48) | provisioning | B3 harness defect (missing PYTHONPATH made pytest SKIP; the `1 passed` guard refused the skip) — destroyed. Runner orchestration fixed and disclosed; no measurement logic changed |
| 3 | EPYC 9655 (CA, whole-48, same machine) | **gate host 1** | gate **ACCEPT** (B1 noise 2.2%/6.1%; B2 median 1.8%) → B ran → **A3 VOID**: drift at 5/10 scored cells |
| 4 | EPYC 9654 (CA, fractional) | **gate host 2** | gate **REJECT**: B1 noise 14.8%/18.5% (bar ≤10%), B2 median 13.0% (bar ≤5%) |
| 5 | EPYC 9554 (TH, whole-64) | **gate host 3** | gate **REJECT**: triad 41.9 GB/s (broken/misconfigured memory), absolutes 10× other hosts, same-binary noise up to 116.7%, median 83.5% |

B3 (bit-exactness of the split path, 32 threads) **passed on all three
hosts that reached it** — the correctness property is machine-checked
and not in question. What is unproven is the performance claim.

## The VOID, and why it is the right call (gate host 1)

The 9655 passed the gate quiet (B2 median 1.8%) and then shifted regime
mid-run. Two independent lines of evidence in the same receipts:

1. **Impossible-by-construction "kernel effects."** Every B2 sentinel
   cell has `items ≥ 2×threads`, so arm B runs the same work as arm A
   (`rsplit = 1`). Six of eight B2 cells nonetheless showed B slower by
   191–260 µs (e.g. (64,128,768): 333.1 → 593.9 µs). A code change that
   is a no-op at those shapes cannot do that; the box changed.
2. **A3 confirmed directly**: 5/10 cells outside their allowance,
   bidirectional — most cells 11–20% *faster* than mean(A), while
   (1,128,768) was 65% *slower*. The machine was not stationary in
   either direction.

Had the registered design lacked the A3 confirm, this run would have
scored as a spectacular false REFUTATION — "the split harms well-fed
cells by 60–80%" — on cells where the binary behavior is identical.
The triad drift between two rentals of this same machine 20 minutes
apart (252.7 → 197.5 GB/s) says co-tenant load was moving underneath us.

## What survives as suggestive (never certified)

At the starved cell (G=1, rows=128, N=256), the split has now shown a
large same-direction improvement on two different machines, in two
differently-shaped attempts, both times in runs that could not be
scored:

* first cert (9V74, rsplit=4): 617 → 342 µs (ratio 0.554) — box
  unscoreable (A/A spread median 15%, worst 55%)
* this cert (9655, rsplit=8, pre-drift): 428.2 → 294.6 µs (31.2%
  improvement, its ≥30% bar) — run voided by A3

Nothing about N=768 survives: the one gate-accepted box drifted
hardest at exactly that cell (+65% in A3).

## Instrument findings (the real product of this cycle)

* **The A/A gate works and is cheap.** ~4 minutes of box time decides
  scoreability before the expensive arms. It accepted the quietest box
  seen in either cert (median 1.8%) and rejected the two loud ones with
  margin to spare.
* **The gate is necessary but not sufficient — the A3 confirm is what
  catches the rest.** A box can be quiet for the gate's four minutes and
  shift during the next ten. Gate + confirm together bracket the whole
  measurement; neither alone is enough.
* **"Whole machine" does not mean quiet.** Host 5 was a whole 64-core
  socket with no co-tenants and produced 83.5% median same-binary noise
  from its own memory system (41.9 GB/s triad on a Zen-4 socket).
  Host-quality priors (family, whole-vs-fractional, triad) predicted
  scoreability poorly; only the gate itself separated them.
* Rented-market conclusion: at bars of ±10%/5-of-median, **0 of 4
  timing-capable vast hosts across both certs were scoreable end-to-end.**
  Micro-benchmark certification at this precision wants owned hardware.

## Disposition

The kernel revert in this PR restores `gnf4_native/cpu_kernels.c` to its
pre-split state (`426b4f2` content, `rsplit` count 0). The prereg,
harness, verdict calculator, and receipts stay. The line is CLOSED
pending a fundamentally different instrument — perf-counter isolation on
owned AVX-512 bare metal — per the prereg's own hard stop. The
suggestive N=256 signal is recorded above for that future instrument.
