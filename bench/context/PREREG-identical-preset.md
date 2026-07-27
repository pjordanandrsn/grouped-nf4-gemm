# PREREG — the identical-fidelity preset's speedup

`enable_decode_stack(fidelity="identical")` = routed staging + speculative
staging, **without** the fused kernel. Its docstring currently declines to quote
a number because the published ladder is **cumulative** (routed → +fast → +spec),
so this combination has never been measured. This measures it.

## Why it is not just arithmetic

From #49 on Qwen3-235B-A22B: bulk **5.7974** → routed **0.9223 (6.29×)** →
+fast **0.6800 (8.53×)** → +spec **0.5764 (10.06×)**. Speculation contributed
**1.18×** *on top of the fused kernel*. It does not follow that it contributes
1.18× without it: speculation hides transfer behind compute, and removing the
kernel makes the compute half **slower**, which gives speculation *more* to hide
behind. The naive product could under- or over-state it.

## Prediction

**routed+spec = 7.0–8.0× vs bulk** (i.e. 0.72–0.82 s/token), registered before
the run. Point estimate **7.4×** from the naive product 6.29 × 1.18.

- **Below 7.0×** → speculation is worth less without the kernel than with it.
- **Above 8.0×** → it is worth *more*, i.e. a slower compute half gives the
  prefetch more to hide behind. That would be the interesting outcome and would
  mean the two mechanisms interact, not merely compose.

## Protocol

Same harness as #49 (`bench/context/e4b_ladder.py`), `--rungs bulk,routed,spec`,
same model and pod, **run only after the routed-residual arms have completed and
its process has exited**.

> This is deliberately **not** an arm of `PREREG-routed-residual`. That prereg
> states *"speculative staging OFF … in every arm … leaving either on would
> confound every comparison against the 0.936 s baseline"*. Adding a spec-on arm
> there would violate its design. This is a separate run sharing only the pod and
> the already-downloaded checkpoint.

## Gates

- Both non-bulk rungs must be **bit-identical** to bulk
  (`max|Δlogit| = 0.000e+00`). This is the preset's entire claim; a nonzero
  delta on either falsifies the `"identical"` label and is a **STOP**.
- Absolute times do not transfer between machines; only the within-run ratio is
  claimed.
