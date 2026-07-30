# The K3 provenance chain: from Moonshot's published hashes to the multiply

Each link below has its own receipt. This document exists because the links are
only worth something *composed* — the claim is not "we hashed some bytes", it is
that **an unbroken, independently checkable chain runs from the hashes Moonshot
published to the numbers a GEMM actually multiplies**, with no requantization
anywhere along it.

Every step is verifiable by a third party from public artifacts: the released
checkpoint, `compressed-tensors`, and this repository.

## The chain

| # | link | check | result |
|---|---|---|---|
| 1 | publication → local store | each of 96 shards vs Moonshot's own LFS sha256 | **96/96** |
| 2 | store → inventory | per-tensor sweep, cross-verified against an independent pass | **10,752/10,752**, 0 mismatches |
| 3 | store → arena | relocation bake, slice-validated against source byte ranges | **48/48 segments identical**, control fires |
| 4 | arena → tensors | `ArenaExpertSource` byte-identity, byte-flip control | gated, control fires |
| 5 | packed bytes → numbers | our decode vs **compressed-tensors' own** | **33,030,144 elements, max delta 0** |
| 6 | numbers → compute | arena-fed GEMM vs memory-fed | bit-identical |

**Link 1** caught a real failure rather than assuming success: shard 88 died
mid-pull on a transient SSL timeout, the manifest logged the FAIL honestly, and
the refetch's sha256 matched the `hf_linked_etag` captured *before* the retry.
A store that reported 95/96 as success would have poisoned everything below.

**Link 3** is hash-preserving by construction, not by audit: every arena row
segment is one whole source tensor range, copied and hashed in a single pass.
The slice check re-derives the source ranges from the shard headers
independently, because `verify()` is the module auditing itself.

**Link 5** is the one that decides whether any of the rest means anything. Our
decode had only ever been checked against *transformers' gpt-oss* path. K3
declares `quant_method: compressed-tensors`, a different implementation of the
same nominal format — so a one-convention gap in nibble order, e8m0 bias or
group axis would have left every other link green and every number wrong.

## Why the chain is worth more here than on another model

K3's own technical report documents its quantization recipe (§4.1.4): MXFP4
weights and MXFP8 activations under QAT from SFT through RL, with **rollout and
training sharing the same quantization**.

The consequence is unusual. For most released checkpoints, "the shipped bytes"
and "the numerics the model was trained under" are different things, separated
by a post-hoc quantization step. For K3 they are the same object. So computing
on the shipped bytes *bit-identically* is not merely efficient — it is serving
the model **as trained**, and this chain is what makes that statement checkable
rather than asserted.

Requantizing to NF4, or dequantizing to bf16 and back, breaks that property
immediately and silently. Nothing in the arena path does either.

## What this does not claim

- **Not a speed result.** The tier is bounded by disk: ~25.83 GB/token routed at
  k=16, which is seconds per token on consumer NVMe. It is a batch tier by
  construction and the ceiling document says so unhedged.
- **Not full-model inference.** The chain covers routed expert weights. The
  ~79.5 GB of always-active BF16 tensors are outside it, and K3's quantization
  config explicitly `ignore`s them — so provenance scope is *expert bytes*, which
  is what links 2–6 measure and nothing more.
- **Link 5 is one expert of one layer**, chosen for portability rather than
  sampled. The geometry is uniform across the release, but that is an argument,
  not a measurement.
- **Two implementations agreeing is not proof either is right.** It excludes
  independent-convention error — the live risk — not a shared misreading of the
  OCP spec.
- **`measured` tier, not `confirmed`.** None of these protocols carries an
  OpenTimestamps anchor. They predate their data in public git history only.

## Receipts

- store + per-tensor sweep: `docs/mxfp4/k3-shard-receipts.json` (private lane)
- slice round-trip: [`RESULTS-k3-slice-roundtrip.md`](RESULTS-k3-slice-roundtrip.md)
- decode oracle: [`RESULTS-k3-phase1-oracle.md`](RESULTS-k3-phase1-oracle.md)
- arena → kernel: `kernel/test_arena_experts.py`, `kernel/test_arena_equivalence.py`
- model wiring: `kernel/arena_moe_patch.py`, `kernel/test_arena_moe_patch.py`
- device ceiling and regime: [`nvme-ceilings.md`](nvme-ceilings.md)
