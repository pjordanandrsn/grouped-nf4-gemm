# NVMe tier ceilings — N0 device microbench

**The number the design uses: `S ≈ 3.45 GB/s` sustained achieved read
bandwidth at expert-sized requests (13–50 MiB, QD ≥ 4) on the first owned
measurement box.** `S` is a per-box constant, exactly like the transfer law's
`L` and `c_box`: it is measured by `bench/nvme/nvme_microbench.py` on the box
that will serve the tier, never carried across hosts.

**Regime statement, unhedged:** at `S` in the low GB/s, a tier serving
cold experts from disk is a **batch / overnight tier**. Qwen3-235B-A22B
streams 7.98 GB/token when fully cold — from disk that is ~2.3 s/token
*before* `c_box`; a K3-class model at ~25.8 GB/token is ~7.5 s/token.
Seconds per token, by construction. Interactive use at this scale is not the
claim, and no number in this lane may be quoted without that scope.

## Box

| | |
|---|---|
| host | NAS-class appliance, kernel 6.6.32, Xeon W-1250 |
| device | Seagate FireCuda 530 2 TB (`/dev/nvme1n1`), lbs=pbs=512 |
| negotiated link | **PCIe Gen3 x4** (8.0 GT/s ×4) — drive is Gen4-capable (16 GT/s ×4) |
| rated (drive) | 7.3 GB/s sequential read (Gen4 figure) |
| link practical ceiling | ≈ 3.5 GB/s (Gen3 x4) |
| access | raw block device, `O_RDONLY|O_DIRECT`, random offsets over the full 2 TB span |
| caveat | shared production box (both NVMe are live zpool1 mirror members); contention during the run was measured and negligible — see accounting below |

## The curve (2026-07-27 evening run, 8 s/cell sustained, random aligned)

Achieved GB/s, threaded-pread instrument (`fio`/libaio agreed within ~3 % on
every cell — the two instruments cross-validate; full grids in
`bench/nvme/receipts/receipt-qnap-evening-20260727.json`):

| request | QD 1 | QD 4 | QD 16 | QD 64 |
|---|---|---|---|---|
| 4 MiB | 2.41 | 3.11 | 3.30 | 3.26 |
| 13 MiB | 2.83 | 3.30 | 3.28 | 3.35 |
| 25 MiB | 2.91 | 3.23 | 3.43 | 3.46 |
| 50 MiB | 3.26 | 3.34 | 3.35 | 3.26 |

Per-request p50 latency at the operating region (the prefetch lead-time
constants): 13 MiB = 4.7 ms @ QD1 / 16.4 ms @ QD4; 25 MiB = 8.6 ms @ QD1 /
31.5 ms @ QD4. p99 grows superlinearly with QD (queueing): QD64 p99 runs
0.4–2.6 s — deep queues buy the last ~4 % of bandwidth at latencies the
decode loop cannot tolerate. **Operating point: request = the arena row size
(13–25 MiB class), QD 2–4.**

Validation on the same receipt:

- **self-pair 1.002** at the best cell (25 MiB QD64) — the instrument
  reproduces itself;
- **sustained 30 s = 3.454 GB/s** at that cell (vs 3.464 for 8 s) — no
  thermal/cache cliff; random over a 2 TB span defeats any DRAM/SLC help;
- **sequential control 3.42 GB/s** — random at ≥13 MiB requests costs
  nothing vs sequential on this drive (seek-free NAND at these sizes);
- **cross-instrument** (fio 2.2.10 libaio vs threaded `os.preadv`) ≤ ~3 %
  disagreement per cell, both directions.

## Gate verdict: PASS, with the shortfall fully attributed

The gate asks whether measured `S` falls under ~half the device's rated
sequential figure with no recovering configuration. Measured best = 3.47 GB/s
= **99 % of the negotiated Gen3 x4 link** and 47.5 % of the drive's Gen4
rating. The binding constraint is the *slot generation*, not the drive, the
access pattern, or the design: every request size ≥ 13 MiB at QD ≥ 4 reaches
the link. On a Gen4 x4 port the same part is rated ~2× this figure —
**projection, unmeasured**; the harness re-derives `S` wherever it lands.
Design proceeds with `S = 3.45 GB/s` for this box.

## Findings that shape N2/N3

1. **io_uring is absent here** — `io_uring_setup` returns ENOSYS on
   6.6.32-qnap (QNAP compiles it out) despite the modern kernel. The
   threaded-pread path is therefore a first-class fallback, not a courtesy,
   and it demonstrably drives the device to the link limit at QD ≥ 4. io_uring
   remains the preferred engine where present (pods, stock distro kernels).
2. **QD1 pays at small rows**: 4 MiB @ QD1 = 2.4 GB/s (69 % of link). Small-
   expert models want read depth ≥ 2 more than big ones; the reader keeps ≥ 2
   requests in flight whenever the plan can name the next row.
3. **Alignment discipline held**: all offsets 4096-aligned (≥ lbs 512),
   request sizes MiB multiples, mmap page-aligned buffers — zero EINVAL over
   35 cells. The arena's `row_stride` is padded to 4096 for exactly this.
4. **O_DIRECT on the raw device was mandatory on this box**: the filesystems
   here are QNAP ZFS, whose O_DIRECT is ARC-buffered (a file-based run would
   have measured RAM). On ext4/xfs (pods, desktops) file-based O_DIRECT is
   the normal target; the harness handles both and *warns loudly* when it
   falls back to buffered reads.
5. **RAID-0 pair leg: not measurable on this box** — both NVMe devices are
   live members of the system pool's mirror; no free second device exists.
   The rented-metal receipt (Latitude g3.h100.small, 2026-07-12) measured a
   2-drive stripe at 2.4× a single drive (fio QD1 3.61 GB/s vs 1.52); striping
   scales this tier where provisioning allows it.
6. **Concurrent-with-PCIe leg: MEASURED 2026-07-28 04:23Z — disk and PCIe
   do not contend on this box.** (Run early on operator directive rather
   than at 01:15; the box was verified quiet first — the 235B download had
   completed, non-harness traffic 8–35 MB/s.) Receipts in
   `bench/nvme/receipts/night/`.

   | | alone | with the other stream live |
   |---|---|---|
   | **H2D (pinned, 1 GiB copies)** | **6.25 GB/s** | **6.25 GB/s** (p90 6.13→6.14) |
   | disk 13 MiB QD1 / QD4 | 2.895 / 3.323 | 2.802 / 3.243 |
   | disk 25 MiB QD1 / QD4 | 2.998 / 3.262 | 3.036 / 3.263 |

   Disk degradation under a saturating H2D stream: **mean 0.989, range
   0.968–1.013** — at or inside run-to-run noise, and the H2D side is
   unchanged to three digits. The two paths are independent at these rates
   (≈9.5 GB/s combined is far under this box's DRAM bandwidth), so the
   tier's overlap assumption — read the next expert off disk *while* the
   current one crosses PCIe — holds on real hardware rather than by
   assertion.

## Contention accounting

Every cell's receipt carries `other_read_mbps` / `other_write_mbps` computed
from `/proc/diskstats` deltas minus harness bytes. During the evening run,
non-harness writes were 1–6 MB/s and non-harness reads ~0 (the ~350 MB/s
figure in the pread cells of this first receipt is the harness's own warmup
traffic — an accounting artifact fixed in the harness the same evening; fio
cells in this receipt subtract nothing and show their own full traffic).
Pre-run baseline: 0.4 MB/s reads, 2.5 MB/s writes. The disk was effectively
idle but the box was not (load ~10): CPU contention affects the *engine*
numbers later, not this device ceiling.

## Publication note

The receipts in this lane were taken on the operator's own hardware. For
publication the NAS **hostname is redacted** in the receipt `probe` blocks
(replaced by an explicit `host_redacted` field rather than silently
removed — a quietly altered receipt is worse than a disclosed one), and
box-specific paths in `night_h2d_leg.sh` are parameterized via
`GNF4_BENCH_DIR` / `DOCKER_BIN`. Every measurement, device identifier,
kernel version and timing figure is unchanged.

## Scope of this document

This is the **device** half of the NVMe-tier work: what the hardware
delivers for this access pattern, measured, with the instruments and
receipts to re-derive it. Placement policy, DRAM/disk economics and the
serving ladders live outside this repository.
