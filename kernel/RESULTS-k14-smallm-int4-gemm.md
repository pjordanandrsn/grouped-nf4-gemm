# RESULTS — K14: REFUTED. At M=16 no shipped int4 arm beats dequant-then-GEMM,
# and the registered mix saving is exactly zero.

Measured 2026-09-11 under `PREREG-k14-smallm-int4-gemm.md` including Amendment 1.
Receipts in `receipts-k14/`. RTX 5090, 170 SMs, torch 2.8.0+cu128, triton 3.4.0.
Launch floor **3.57 µs**; measured streaming ceiling **1528 GB/s**. No verdict
here rests on an estimate.

```
K14 VERDICT: REFUTED
  mix saving over 48 layers = 0.000 ms/step against a registered bar of 0.5;
  every shape's best arm is bf16
```

## The table

At M=16, the best of twelve swept block configs for the `gemm` arm:

| shape | `bf16` | GB/s | % of ceiling | × launch floor | `gemm` | `gemv` |
|---|---|---|---|---|---|---|
| `q_proj` | **10.35 µs** | 1621 | **106 %** | 2.90 | 12.47 (1.205×) | 24.81 (2.397×) |
| `k_proj` | **6.22** | 337 | 22 | **1.74** | 12.47 (2.004×) | 10.33 (1.660×) |
| `v_proj` | **6.22** | 337 | 22 | **1.74** | 12.46 (2.003×) | 10.35 (1.665×) |
| `o_proj` | **18.39** | 912 | 60 | 5.16 | 20.66 (1.123×) | 26.91 (1.463×) |

`q_proj` reads 16.78 MB in 10.35 µs — above the measured copy ceiling, which a
read-only stream is entitled to be, since a copy pays for its write. **The bf16
path on the large shape is at the memory roofline and M is free**: 10.3 µs at
M=1 and 10.3 µs at M=16.

## The sweep is why this refutation is worth something

At the shipped `bn64/w8` the GEMM ran **2.4× to 6.3×** bf16. At its best config
it runs **1.12× to 2.00×**. It still loses.

A single projection is one M-tile, so the grid is `1 × cdiv(N, block_n)` — the
expert path this kernel was swept for gets its parallelism from the expert
count and a projection has none of it. `bn16/w2` and `bn32/w4` won everywhere,
lifting program counts from 64 to 128–512. Without the sweep this lane would
have reported a bad default rather than the kernel.

## The registered predictions

1. **`gemm` wins at M=16 on `q_proj` and `o_proj`.** Measured **1.205×** and
   **1.123×**. **REFUTED.** The A2000 dry-run in Amendment 1 read 0.898 and
   1.035 on those cells, and the census hardware does not agree — one more
   instance of the rule P39 established, that a number fitted on sm_86 does not
   transfer to sm_120.
2. **`gemv` loses at M=16 everywhere.** 2.397 / 1.660 / 1.665 / 1.463.
   **CONFIRMED** — `GEMV_ROWS_MAX = 1` is correct, and now measured rather than
   inherited.
3. **`k_proj`/`v_proj` are launch-bound.** 6.22 µs against a 3.57 µs floor,
   **1.74×**, inside the registered 2×; and no int4 arm improves them (1.66×
   and 2.00× *worse*). **CONFIRMED.**

## Why the int4 arms lose, stated as a bound rather than a story

The `gemm` arm reads a quarter of the bytes and takes longer, so it is not
bandwidth-limited: 378 GB/s on `q_proj`, **22 % of ceiling**. The bf16 arm it
must beat is at 106 %.

The prize is therefore real but unclaimed by anything that exists here. An int4
kernel that hit the memory roofline would cost
`max(launch floor, int4 bytes / 1528 GB/s)` per call — which for every one of
these four shapes is the **launch floor**, 3.57 µs, because 4.19 MB streams in
2.74 µs:

| | measured bf16 | int4 at the roofline | saved |
|---|---|---|---|
| per layer (q + k + v + o) | 41.18 µs | 14.28 µs | 26.90 µs |
| **per step (× 48)** | **1.977 ms** | 0.685 | **1.292 ms** |

So a perfect int4 projection kernel is worth about **1.29 ms/step**, which is
above the 0.5 ms bar. Nothing shipped realises any of it, and the kernel built
for this arithmetic delivers 22 % of the bandwidth it would need. **That is a
kernel lane, and it is not this one.**

## Cross-check against P42

Per layer the four projections sum to **41.18 µs**, so **1.977 ms/step**.
P42's census attributed a **~2.07 ms** proportional share to the 192 attention
projections inside its 240-call GEMM row. Two independent measurements, taken
by different instruments on different boxes, agreeing within 5 %.

## What this lane found that it did not go looking for

**`o_proj` is not at the roofline.** It reads the same 16.78 MB as `q_proj` and
takes 18.39 µs against 10.35 — 60 % of ceiling against 106 %. Same bytes, same
device, different orientation (N=2048 K=4096 against N=4096 K=2048). If
`o_proj` reached `q_proj`'s efficiency it would cost ~10.4 µs, worth
**0.38 ms/step** — on the **bf16** side, with no int4 kernel involved.

**q/k/v were measured unfused.** The arms that produced both this lane and
P42's census pass `--no-fuse-qkv`, so each layer issues four projection
launches. `k_proj` and `v_proj` spend 57 % of their time being launched at all.
Replacing three launches with one, which `--fuse-qkv` already implements, would
merge 20.98 MB into a single stream; **at the measured 1528 GB/s ceiling** that
is 13.73 µs against the measured 22.79, worth about **0.44 ms/step**. (At
`q_proj`'s own 1621 GB/s it would be 12.94 µs and worth 0.47.)

Both are bounds computed from this lane's own numbers under a stated
assumption, not measurements. Both are larger than any int4 arm managed, and
neither needs a new kernel.

## Cost

Two ledger rows, **$0.0653**, against a $1.50 lane ceiling — `k14-stagea`
($0.0289) died on a harness bug I wrote from memory, and `k14-stagea-2`
($0.0364) ran the lane in 90 seconds. The A2000 dry-run that produced
Amendment 1 was free and is the reason the second box measured the kernel
rather than a default.
