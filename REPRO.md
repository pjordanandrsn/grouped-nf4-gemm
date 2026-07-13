# Reproducing the Gate-2 decode result

Everything below runs from the frozen tree (`ad2bef0`) with no configuration:
the kernel's decode dispatch table and defaults are in-code, the benchmark
declares its own shapes, and the reduction applies the pre-registered
criteria mechanically.

## Requirements

- NVIDIA GPU, sm_86 (the registered claim's scope; other archs run but the
  numbers are not the registered ones)
- CUDA driver ≥ 570 (CUDA 12.8 runtime)
- Python ≥ 3.10 with: `torch >= 2.8` (ships triton ≥ 3.4), `bitsandbytes`,
  `pynvml`, `pytest`
- A C compiler on PATH (`gcc` is enough — triton builds its launcher stubs at
  runtime; a bare `pytorch/pytorch:*-runtime` docker image will fail without
  it)

```sh
pip install torch bitsandbytes pynvml pytest
```

## 1. Property suite (correctness, ~2.5 min)

```sh
python -m pytest kernel/test_nf4_grouped.py -q
```

Expected: **35 passed**. This asserts bit-exact NF4 decode vs bitsandbytes at
bf16 output precision, the P-fid bound (fused error ≤ dequant-path error vs
an fp64 reference on fp32-decoded values), B-rel ≤ 2×, adversarial absmax,
and tiling boundaries.

## 2. Benchmark (one rep, ~10–20 min depending on card)

```sh
python bench/phase1/harness.py \
  --models OLMoE Qwen3-30B gemma-4 gpt-oss \
  --extra-shapes bench/phase1/heldout_shapes.json \
  --regimes decode_bs1 \
  --backends dequant_grouped gemv_4bit fused_nf4 \
  --energy-window 5.0 \
  --out rep1.json
```

The confirmatory protocol (`kernel/prereg_gate2_confirmatory.json`, OTS-stamped
before any confirmatory data existed) runs this **three times per device,
fresh process each** (`rep1.json rep2.json rep3.json`).

## 3. Reduction + verdict (mechanical)

```sh
python bench/phase1/reduce_confirmatory.py \
  --device <NAME> rep1.json rep2.json rep3.json \
  --suite <NAME>:35/35
```

Per cell it takes the **worst rep** — `min` over reps of
`dequant_ms / fused_ms` and `max` over reps of `fused_J / dequant_J` — and
prints the C1–C5 verdicts exactly as registered (exit 0 iff all pass). For
the two-device confirmatory, pass both `--device` groups in one invocation.

## Provenance chain

1. `kernel/prereg_gate2_confirmatory.json` + `.ots` — protocol and pass/fail
   criteria stamped (OpenTimestamps → Bitcoin) **before** the confirmatory
   ran, at frozen commit `ad2bef0`.
2. Rep JSONs — committed as produced (each embeds `env`: GPU, driver, torch,
   triton versions).
3. `kernel/RESULTS-gate2-confirmatory.md` — reduction output and verdicts,
   stamped after.

Deviations, aborted attempts, and anything that failed are reported in the
results doc at full volume — the registered no-tune clause forbids re-running
until green.
