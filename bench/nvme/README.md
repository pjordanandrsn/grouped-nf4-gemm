# NVMe arena: format, provenance, verifier, device microbench

This is the **checkable half** of an NVMe storage tier for MoE expert
weights: the on-disk format, the provenance chain that makes a relocated
copy verifiable, and the instruments that measure what a device actually
delivers for this access pattern. Placement policy and serving economics
are not part of this repository.

## What's here

| | |
|---|---|
| `kernel/nvme_arena.py` | expert-major arena: **relocation bake** (per-expert byte-identity to the vendor's shipped tensors), sidecar index, sha256 manifest, and `verify` (arena self-check, plus `--against-source` for the full shipped→manifest→arena chain) |
| `kernel/nvme_bake_nf4.py` | **quantize-at-bake** for bf16 checkpoints; two-hop provenance (arena hash + the source ranges consumed + the quantizer record), honestly weaker than relocation and labeled as such |
| `kernel/nvme_reader.py` | O_DIRECT reads with async submission into caller-owned aligned buffers; loud, never-silent fallback when O_DIRECT is unavailable |
| `bench/nvme/nvme_microbench.py` | device curve: request size × queue depth, two independent instruments (threaded `preadv` and fio), alignment enforced from sysfs, contention accounted from `/proc/diskstats`, self-pair validation |
| `bench/nvme/compare_bake_determinism.py` | adjudicates whether two independent bakes agree, per segment, with a source-hash control |

## Two results worth reading

- **`docs/nvme-ceilings.md`** — a measured device curve, including the
  finding that disk reads and a saturating host-to-device PCIe stream do
  **not** contend on the box measured (H2D unchanged; disk within noise).
- **`docs/RESULTS-nvme-determinism.md`** — two different GPUs (sm_89 and
  sm_86), different hosts, independent downloads of the same checkpoint,
  produced **byte-identical arena rows: 20,480/20,480 segment hashes**.
  The practical consequence is that nobody needs to ship a large arena —
  bake locally from the vendor's release and check your manifest against
  a published one.

## Running the tests

```
python3 -m pytest kernel/test_nvme_arena.py kernel/test_nvme_bake_nf4.py \
                  kernel/test_nvme_bake_experts.py -q
```

24 tests, CPU-only, no GPU and no model required. They cover byte-identity
of relocated data, both verify hops, single-byte corruption being caught
*and named*, index round-trip, block alignment, worker-count invariance,
and a file-descriptor bound (a leak here only appears at scale — it cost a
1.4 TB bake to learn).
