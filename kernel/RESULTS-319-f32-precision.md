# RESULTS — #319: the f32 arms were never running an f32 kernel

## The finding

`kernel/test_fp8_paged_attn.py` builds its shape arms in `_modes()`. The
two f32 arms were:

```python
modes = [("split", {}), ("packed", {"pack_heads": True})]
```

No `compute` key. That was right while an unset `compute` meant f32. It
stopped being right at **RESULTS-m3-default-on**, which made the default
capability-conditional: `_compute_default` returns **fp8 on sm_89+**. So
on every Ada, Hopper and Blackwell card since that cycle, the `split` and
`packed` arms have been running the **fp8** kernel — and `_close` has
been judging them at the **f32** tolerance, because it reads the
tolerance off the arm's *name*:

```python
tol = 1.5e-1 if mode in ("f8dot", "pf8") else 2e-2
```

The fp8 path's own error envelope is 1.5e-1. Checked at 2e-2, it fails.
That is the whole of #319.

## The measurement that settles it

`p319b-where-1`, RTX 5090 (sm_120), torch 2.8.0+cu128 / triton 3.4.0.
One call per arm on identical inputs, reading the kernel's own compute
tally:

```
compute_counts after four calls: {'f32': 0, 'fp8': 4}

  split  identical_to_split=True   max|o-split|=0.000000
  packed identical_to_split=False  max|o-split|=0.090210
  f8dot  identical_to_split=True   max|o-split|=0.000000
  pf8    identical_to_split=False  max|o-split|=0.090210
  packed vs pf8 identical: True
```

Four calls, four fp8, zero f32. `split` and `f8dot` return **byte-identical**
tensors; so do `packed` and `pf8`. Within each geometry they were the same
call. The arms differed by tile shape, never by compute mode.

And the error is the fp8 path's own, exactly as that path has always
described itself:

| quantity | p319b (5090) | the fp8 path's own record |
|---|---|---|
| relative Frobenius | 0.049213 | 4–5%, "flat in head dim" (gnf4#324) |
| T=1 | 0.000000 | "probed exact at T=1 (p == 1 → 448, representable)" |
| T=16 | 0.117188 | between the 0.087 at T=2 and 0.034 at T=33 |
| error shape | mean 0.0086, p50 0.0059, p99 0.041, max 0.074 | a rounding cloud, not a structural outlier |

## The fix, verified on the card that reported it

`p319c-verify-1`, same RTX 5090 class, one box, both commits:

| arm | what ran | result |
|---|---|---|
| A | release `9206352`, untouched | **27 failed**, 57 passed |
| B | branch `3ad0dba`, every arm naming its compute mode | **93 passed** |
| D | the f32 half under `GNF4_ATTN_F32_PRECISION=tf32x3` | 34 passed |
| D | the f32 half under `GNF4_ATTN_F32_PRECISION=ieee` | 34 passed |

Arm B is the claim: with the arms pinned, the f32 modes run f32 on sm_120
and meet **their own** 2e-2 tolerance, on the card that reported 27
failures against them. Nothing in the kernels changed between A and B
except that the arms now name what they run.

And with arm C finally measuring what it says, the sm_120 f32 errors are
the **same bf16 output ULPs the A2000 reports**:

| shape | max abs err (sm_120) | max abs err (sm_86) |
|---|---|---|
| permuted_tables | 0.003906 | 0.003906 |
| partial_tails | 0.007812 | 0.007812 |
| grouped_k_scales | 0.007812 | 0.007812 |
| small_gqa_group | 0.003906 | 0.003906 |
| single_token | 0.015625 | 0.015625 |
| head_dim_256 | 0.007812 | 0.007812 |

Two architectures, one set of numbers, 0.00% over tolerance on both, and
exactly 0.000000 under `tf32x3` on both. There was never an
architecture-dependent numerical problem to explain.

## Why Ampere looked healthy

RTX A2000 (sm_86), **the identical wheels the issue names**: the suite
passes, 35/35 non-skipped. Not because Ampere is luckier — because there
is no fp8 MMA to default to, so on sm_86 the `split` and `packed` arms
genuinely run f32. The f32 kernels were fine all along; on sm_89+ they
were simply never being tested.

Their residual there is one ULP of the bf16 output and nothing more:

| shape | mode | max abs err | over tolerance |
|---|---|---|---|
| permuted_tables | split | 0.003906 | 0.00% |
| partial_tails | split | 0.007812 | 0.00% |
| grouped_k_scales | split | 0.007812 | 0.00% |
| small_gqa_group | split | 0.003906 | 0.00% |
| single_token | split | 0.015625 | 0.00% |
| head_dim_256 | split | 0.007812 | 0.00% |

0.015625 is 2⁻⁶, one bf16 step at |out| ≈ 3.8. Worst element diff
0.003906, **median exactly 0.000000**, relative Frobenius 0.002344 — a
different world from the 0.049 the fp8 path shows.

## What "triton 3.4" was doing in the title

Nothing. The issue's two suspects were the toolchain and the dot's
default `input_precision`, and both are refuted:

* **Toolchain.** sm_86 passes on torch 2.8.0+cu128 / triton 3.4.0, the
  exact pair named.
* **Dot precision.** The f32 split path's two dots took fp32 operands and
  passed no `input_precision`, so the arm was whatever the compiler
  picked. It is now named (`GNF4_ATTN_F32_PRECISION`, default `tf32` =
  the inherited behaviour) and measured on sm_86:

  | precision | worst split err | B=25 T=4096 H=32/8 D=128 | vs tf32 | PTX |
  |---|---|---|---|---|
  | `tf32` | 0.015625 (1 bf16 ULP) | 77.9 GB/s | 1.00× | 32 `mma.sync` |
  | `tf32x3` | 0.000000 | 39.5 GB/s | 0.51× | 96 `mma.sync` |
  | `ieee` | 0.000000 | 4.5 GB/s | 0.06× | **0** `mma.sync`, 1045 `fma.rn.f32` |

  So exactness is available and priced, and the issue's suggested remedy
  ("pin `input_precision="ieee"`") would have cost 94% of the path's
  throughput — landing it *below* the 4.8 GB/s occupancy-starved first
  version this kernel was written to replace — while fixing nothing,
  because the f32 kernel was not the one failing.

`tf32x3` is genuinely exact against the fp32 oracle on four of six shapes
and is the honest opt-in for anyone who needs it. It is not a fix for
#319 and is not offered as one.

## A correction to my own round 1

`p319-f32prec-1`'s sm_120 **accuracy** rows carried the same defect as
the suite: its `split` and `packed` rows passed no `compute` and so
measured the fp8 kernel. They are **withdrawn**. What made the mistake
visible was that the error did not move across `tf32` / `tf32x3` / `ieee`
while throughput moved 24× — which I first read as "the dot precision is
refuted" and which actually meant "the f32 kernel the knob governs is not
running". The throughput rows stand; they passed `compute="f32"`
explicitly, which is precisely why they moved.

**The lesson is the fix**: `test_modes_run_the_kernel_they_name` asserts
the compute tally per arm. Both paths return a plausible tensor, so the
tally is the only witness that an arm ran what it claims. Here the
mislabel fell the strict way and manufactured 27 failures; the same slip
on a card without fp8 would fall the lax way and hide a real defect
behind a 7.5× tolerance.

## #324's f32 half, closed

#324 asked for `head_dim` 256 in **every** mode; only the fp8 half got a
test. The f32 packed kernel has no pre-launch fit model — only the
`OutOfResources` catch — and that catch had never been exercised. The new
case reproduces #324's numbers on the A2000, to the byte:

```
RuntimeWarning: packed f32 paged attention does not fit shared memory at
head_dim 256, 8 kv heads, 16 tokens/block (out of resource: shared
memory, Required: 148480, Hardware limit: 101376); falling back to the
split f32 kernel
```

One warning, the geometry remembered in `_PACKED_UNFIT`, a result that
matches the oracle. The 5090 shows the fp8 half of the same fallback
firing pre-launch from the calibrated model. Both cards carry a 101376-byte
limit, so this is the same wall the issue hit.
