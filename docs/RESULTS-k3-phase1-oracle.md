# Phase-1 oracle: our MXFP4 decode is bit-identical to compressed-tensors', on real K3 bytes
### 2026-07-30 · `moonshotai/Kimi-K3` layer 1 expert 0, extracted verbatim · compressed-tensors 0.17.1

The open gate every K3 number rested on. Our decode had been validated
bit-exactly against **transformers' gpt-oss path**
(`convert_moe_packed_tensors`) — a different implementation of the same nominal
format. But K3's config declares `quant_method: "compressed-tensors"`, format
`mxfp4-pack-quantized`, so *their* dequant is the reference K3 actually names.
A one-convention gap in nibble order, e8m0 bias or group axis would have left
every existing gate green and every K3 number wrong.

## Result

| projection | shape | exact | max abs delta | mismatched |
|---|---|---|---|---|
| `w1` (gate) | 3072 × 3584 | **True** | **0** | 0 / 11,010,048 |
| `w3` (up) | 3072 × 3584 | **True** | **0** | 0 / 11,010,048 |
| `w2` (down) | 3584 × 3072 | **True** | **0** | 0 / 11,010,048 |

**33,030,144 elements, zero differences.** Not "within tolerance" — identical.

## Method

Real released bytes, not a fixture: one expert lifted verbatim out of the
byte-verified 1.56 TB store (its shards individually checked against Moonshot's
own LFS hashes), carrying its shipped `weight_packed` / `weight_scale` pairs.

The reference is composed from compressed-tensors' own code rather than
reimplemented. `MXFP4PackedCompressor` extends `NVFP4PackedCompressor`, so the
decode is their nibble unpack times their scale decode:

```python
theirs_q = unpack_fp4_from_uint8(blocks, rows, K)      # compressed_tensors.compressors.nvfp4.helpers
theirs_s = decompress_mx_scale(scales)                 # compressed_tensors.compressors.mx_utils
theirs   = (theirs_q.reshape(rows, K//32, 32) * theirs_s.reshape(rows, K//32, 1))
```

An earlier revision of the test probed four candidate entry points and skipped
if none worked. That skip was honest but proved nothing; the entry points are
now resolved and pinned, so the gate either runs or fails.

## What this licenses, and what it does not

**Licensed:** the packed→dense arithmetic. Our reading of K3's e2m1 nibbles and
e8m0 exponents agrees exactly with the library the checkpoint names, so numbers
computed from these bytes are not resting on a format guess.

**Not licensed:**

- **Not a kernel claim.** This gates the reference decode
  (`mxfp4_pack_ref.dequant_mxfp4`), which is what the arena path's equivalence
  test compares *against*. The Triton kernel is gated separately.
- **One expert, one layer.** Chosen for portability, not sampled. The geometry
  is uniform across the release, so a per-expert divergence is implausible —
  but it is untested.
- **Two implementations agreeing is not proof either is right.** It excludes
  independent-convention error, which was the live risk; it does not exclude a
  shared misreading of the OCP spec.

Reproduce: `K3_EXPERT=<expert.safetensors> pytest kernel/test_k3_oracle.py`.
