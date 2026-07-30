# Real-bytes arena round-trip on released Kimi-K3: 48/48 segments identical
### 2026-07-30 · slice of the real 1.56 TB store (2 layers × 4 experts) · torch-free bake path

Before committing to a **1446.5 GB** bake, this checks the arena against the
bytes Moonshot actually shipped. The toy fixtures in
`kernel/test_arena_experts.py` cannot do that: they are built from the same
constants they test, so a wrong `name_template`, a wrong `kinds` order, or a
misread segment geometry would pass them and fail on the real checkpoint.

## What ran

`bake_expert_tensors` over `moonshotai/Kimi-K3` restricted to layers 1–2 and the
first 4 experts — 8 rows, 0.1 GB, ~1 s at 4 workers.

| | |
|---|---|
| snapshot | `/share/ZFS532_DATA/hf-models/moonshotai_Kimi-K3` (96 shards, byte-verified against Moonshot's LFS hashes) |
| template | `language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{kind}` |
| kinds | `w1/w3/w2 × {weight_packed, weight_scale}` |

## Geometry, derived from the release rather than assumed

| segment | offset | length | shape | dtype |
|---|---|---|---|---|
| `w1.weight_packed` | 0 | 5,505,024 | [3072, 1792] | U8 |
| `w1.weight_scale` | 5,505,024 | 344,064 | [3072, 112] | U8 |
| `w3.weight_packed` | 5,849,088 | 5,505,024 | [3072, 1792] | U8 |
| `w3.weight_scale` | 11,354,112 | 344,064 | [3072, 112] | U8 |
| `w2.weight_packed` | 11,698,176 | 5,505,024 | [3584, 1536] | U8 |
| `w2.weight_scale` | 17,203,200 | 344,064 | [3584, 96] | U8 |

**Row = 17,547,264 B**, which reproduces this program's independently recorded
per-expert figure (17.55 MB) from nothing but the release's own headers.

Two geometry facts fall out and are worth keeping:

- **`row_stride == row_bytes`.** 17,547,264 = 4284 × 4096 exactly, so at
  `align=4096` K3's arena carries **zero padding overhead** — useful bytes and
  billed bytes coincide, which is not true in general.
- The experts contract over **3584**, the `routed_expert_hidden_size`, not the
  7168 `hidden_size`: gate/up are `[moe_inter 3072, 3584/2]` and down is
  `[3584, moe_inter/2]`. "Stable LatentMoE" down-projects *outside* the experts.
  This is why the MoE patch attaches to `moe_infer` and not `forward`.

## Results

| check | result |
|---|---|
| `verify()` — re-hash every segment against the manifest | **8/8 rows OK**, 0 failures, 0.4 s |
| **independent** — arena row bytes vs safetensors byte ranges, headers re-read from source | **48/48 segments identical** |
| negative control — flip one arena byte | **detected** (differs from source: `True`) |

The independent check exists because `verify()` is the module auditing itself: a
defect shared by bake and verify would pass it. Re-deriving the source ranges
from the shard headers and comparing raw bytes closes that. The negative control
exists because a byte-identity check with no demonstrated failure mode is a
constant function.

## Scope

- **Slice, not the store.** 8 of 82,432 rows. It validates naming, segment
  geometry, offsets and relocation fidelity — not the full bake's endurance.
- **Bytes, not compute.** This is the arena and reader path, which is torch-free.
  `ArenaExpertSource`'s tensor reconstruction is gated separately in
  `kernel/test_arena_experts.py` (19/19 on an RTX A5000, including a byte-flip
  control and an arena-fed-vs-memory-fed GEMM equality).
- **No throughput claim.** The 0.1 GB/s observed here is 8 rows on a NAS with
  live services and says nothing about the full bake.

## What it de-risks

A wrong `name_template` or `kinds` order would have produced a clean-looking
arena of the wrong bytes, discovered only after ~1.4 TB of IO. That failure is
now excluded for ~1 second of work.
