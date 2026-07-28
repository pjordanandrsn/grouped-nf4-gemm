# Cross-box bake determinism: PASS, exactly — 20,480/20,480 segments identical
### 2026-07-28; prereg `docs/PREREG-nvme-bake-determinism.md` (pre-data), receipt `bench/nvme/receipts/determinism-l40s-vs-a2000.json`

**Two different GPUs, two different hosts, two independent downloads, one
answer: the arena is byte-identical.** Every one of 20,480 compared
segment hashes matched — `D1 = 1.000000` against a registered bar of
0.999 — with the source control clean.

| | A | B |
|---|---|---|
| box | rented L40S (Ada, **sm_89**) | owned A2000 (Ampere, **sm_86**) |
| host | rented, 1.4 TB RAM | owned NAS-class appliance, 128 GB RAM, live services |
| source | its own 438 GB download | its own **independent** 438 GB download |
| bake | `nf4-quantize`, bnb 0.50.0 | `nf4-quantize`, bnb 0.50.0 |
| wall | 552 s (94 layers) | 792 s (40 layers) |

| registered outcome | result |
|---|---|
| **D3 source control** (must be clean or D1 is void) | **CLEAN** — 30,720 shared source records, **0 differ** |
| **D1 segment determinism** ≥ 0.999 | **1.000000** — 20,480 / 20,480 |
| per-segment-kind | gate_up_blocks 5120/5120 · gate_up_absmax 5120/5120 · down_blocks 5120/5120 · down_absmax 5120/5120 |

The per-kind breakdown matters: it was registered specifically so a
scales-only or blocks-only divergence could not hide inside an average.
Neither packed nibbles nor fp32 block-absmax moved by a single byte across
architectures.

## Why this is the stronger test

Amendment 1 originally registered a **file transfer** — copy the 128 GB
arena from the bake box to the consumer box. That was abandoned on
measurement: the link sustained **~6 MB/s**, i.e. ~6 hours and ~$6 of idle
GPU to move bytes that, it turns out, the receiving box can simply
*regenerate*. A copy demonstrates transport. This demonstrates that **the
arena is a function of the checkpoint, not of the machine that baked it.**

Practical consequence, which is the point: a user does **not** need to
download a 128 GB (or, for K3, 1.4 TB) arena from anyone. They bake
locally from the vendor's own release and check their manifest against a
published one. Provenance survives the trip because nothing has to make
the trip.

## What it does not license

- **Scope: NF4 via bitsandbytes 0.50.0, sm_86 vs sm_89, same
  blocksize/quant_type.** It is not a claim about every quantizer, every
  bnb version, or every architecture pair. A different bnb release could
  change kernel internals; that is exactly why the manifest records the
  quantizer (`kind`, `quant_type`, `blocksize`, `bnb`, `torch`) alongside
  the hashes — a mismatch there is a legitimate reason for hashes to
  differ, and the record makes that diagnosable instead of mysterious.
- The `relocate-expert-tensors` mode (K3, gpt-oss) never depended on this:
  it copies shipped bytes, so its segment hashes *are* the vendor's.
  Determinism there is trivial; here it was a real question because
  quantization is arithmetic.

## Instrument validation (done before the number existed)

`compare_bake_determinism.py` was checked in both directions first:
positive control — self-compare of the L40S manifest returns exactly
1.000000 across all 48,128 segments; negative control — a single mutated
arena hash is caught and named with its layer/expert/segment, and a single
mutated source hash flips D3 to DIVERGENT and voids the verdict. An
instrument that can only report agreement is not evidence of agreement.
