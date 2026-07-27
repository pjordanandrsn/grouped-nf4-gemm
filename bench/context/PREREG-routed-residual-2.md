# PREREG — routed-residual 2: R4 re-registered on bare metal, honest instrument

**Tier: CONFIRMATORY on P1–P3. Status: DRAFT until OTS-stamped; must be stamped
before any qualifying host is provisioned.**

## Why a fresh registration and not an amendment

The first campaign (`ba7a29ce` + `2397612e`, results
`RESULTS-routed-residual.md`) left R4 unadjudicated: its registered ceiling
instrument reads pinned below pageable on containerized A100 hosts —
reproducibly, settled or not — so the denominator was unusable. The obvious move,
"amend the ceiling to best-of-N and rerun," is **not available honestly**: the
authors have seen 36 arms of data and know what verdict each candidate
instrument produces (decode-only routed ≈ 22.0 GB/s; candidate ceilings 22.21 /
24.3 / 26.0 → fractions 0.85–0.99, all falsifying). Choosing the instrument now
is choosing the answer. Per the registration ladder this drops to exploratory
and re-registers out-of-sample: **new host, new data, instrument fixed before
seeing it.**

**Contamination disclosure (bias direction):** the authors expect R4 to be
falsified — transfers already near-efficient, coalescer not worth building.
This prereg is written knowing that, so its risk is confirmation bias toward
falsification. The guards: the instrument and bars below are fixed before any
qualifying data exists; the falsification direction requires *nothing* — it is
the null; and the interesting/costly outcome (R4 holds → build the coalescer)
remains fully specified and would be believed.

## Fixture

- **Host: root bare-metal H100 (Latitude.sh `g3.h100.small`, the knee-session
  class) — first choice.** Fallback if Latitude stock does not return within
  14 days: DigitalOcean `gpu-h100x1-base` droplet (accepting its container-free
  but virtualized caveat, recorded as such). **Not RunPod**: the anomaly under
  test is suspected container/NUMA behaviour, so the re-test must be on a host
  where the pinned path can be inspected (`numactl`, `drop_caches`,
  `/proc/buddyinfo` readable).
- **This is a host-class change from campaign 1 (2×A100-SXM → 1×H100)** and
  is stated, not smuggled: single-GPU is acceptable because R4 is a claim about
  the *link*, not about TP; the model still fits via offload (expert homes in
  host RAM; the g3 class carries 188 GB — pin only the experts' ~123 GB).
  c_box and absolute s/token do NOT carry across from campaign 1 and will not
  be compared.
- Qwen3-235B-A22B, NF4 experts pinned, routed staging on, spec staging and
  expert cache OFF, `prefetch=False`, natural prompt, greedy, ctx 512, 12 new
  tokens + 2 warmup, **reps 6, ABBA**, one process, `E4B_OFFLOAD_STATS=1`.
- Code: harness `5cc34a6`-or-later (decode-only stats + settled probe + full
  by_policy receipt), verdicts `97ee239`-or-later (plausibility, prefill,
  power, balance gates all active).

## The instrument, fixed here

**Ceiling = best-of-N direct pinned H2D probe**, N=10 per size, sizes
{64, 256, 1024} MB, each timed individually (CUDA events), taken on the settled
host **before the model loads** AND re-taken after load; the receipt carries
every reading. The registered denominator is **max over all pinned readings,
pre-load**. Rationale (fixed in advance): contention and NUMA misplacement only
ever *slow* a transfer, so the max is the least-corrupted reading of the link's
capability; a mean averages corruption in.

- **I1 (instrument sanity, gate):** pre-load, pinned best ≥ pageable best.
  *If this fails on bare metal, the anomaly is not container-caused; STOP —
  R4 is unmeasurable at this altitude and the finding is the anomaly itself.*
- **I2 (stability, gate):** pre-load pinned best-of-10 spread (max−min)/max
  ≤ 0.15 at 256 MB. Wider ⇒ the box is not settled; re-settle or abandon the
  session — no verdict from an unstable instrument.

## Predictions

- **P1 — GATE, identity:** greedy ids and logits bit-identical across arms
  (C vs T1), as in campaign 1. *Any divergence is a STOP.*
- **P2 — THE RE-ASKED R4:** decode-only routed-policy implied GB/s ≤ **0.70 ×**
  the registered ceiling (above). **Expected outcome, disclosed: FALSIFIED**
  (fraction lands 0.80–1.00). *R4-holds would mean the A100 numbers were
  host-artifact and the coalescer question reopens on its merits.*
- **P3 — DECISION RULE (binding, both directions):**
  - P2 falsified (fraction > 0.70): **the expert-major coalescer is not built.**
    The 5.2 ms/layer residual is adjudicated host-side; the lane moves to the
    per-layer stall (trainer-sync class, cf. the H-D decomposition).
  - P2 holds (fraction ≤ 0.70): the coalescer is built, ceiling = the measured
    gap, and a paired prereg prices it before any further claim.
- **P4 — EXPLORATORY (not registered):** post-load pinned best vs pre-load
  pinned best, as a direct measurement of "does pinning 123 GB of homes degrade
  subsequent pinned transfers" — the suspected mechanism behind campaign 1's
  anomaly. Reported either way; quotable only as exploratory.

## Cost & teardown

One box, ≤ 2.5 h, ≤ $10 (Latitude ~$3/h class). Session-independent
watchdog + hard bill-cap per standing discipline (`lat-billcap` pattern:
deadline file, teardown sentinel, delete-verify, evidence rsynced continuously
— the puller earned it this campaign, it is not optional). KEEP-flag rules
apply if RunPod fallback is ever used.

## Not claimed

Nothing about K3; nothing about tok/s; nothing transfers from campaign 1's
host class; T1/R6 are settled (no measurable effect) and are not re-asked.
