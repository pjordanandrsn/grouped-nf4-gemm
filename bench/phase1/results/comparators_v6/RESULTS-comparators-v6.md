# Comparator census on the v6 kernel — fused beats the grouped-GEMM class (unsloth's MoE path) on EVERY cell, decode and prefill

**2026-07-14 · Host:** fresh SECURE A5000 (fifth distinct instance of the
day), frozen v6 tree (`6aeb718` lineage, post-`3cedcd7` verdict), suite
44/44 on-pod. Same-run, same stacks, `--iters 20 --no-energy`. Exploratory
census (not a blind confirmatory); receipts + SHA256SUMS in this directory.

## Backends

- `fused_nf4` — this repo's kernel (v6 register-LUT mainloop, confirmed).
- `unsloth` — `grouped_gemm.ops.gmm` (tgale96), **the grouped-GEMM kernel
  unsloth's MoE backend rides**: dequantize the active experts to bf16,
  then one grouped bf16 tensor-core GEMM. Dequant + layout happen inside
  the timed region — the honest end-to-end cost when weights are STORED
  4-bit (a resident-bf16 deployment is a different regime at 4× the VRAM,
  and is not claimed against). Zero failed cells.
- `dequant_grouped` — the bnb-dequant loop, same-run anchor.

## Result (ratio = comparator ms / fused ms; >1 means fused is faster)

| regime | cell | unsloth/fused | bnb-dequant/fused |
|---|---|---|---|
| decode_bs1 | OLMoE gate_up · down | 4.72 · 4.23 | 2.19 · 3.08 |
| decode_bs1 | Qwen3-30B gate_up · down | 4.61 · 4.45 | 2.39 · 3.33 |
| decode_bs1 | gemma-4 gate_up · down | 4.60 · 5.70 | 1.92 · 4.82 |
| decode_bs1 | gpt-oss gate_up · down | 8.44 · 6.06 | 1.95 · 1.57 |
| prefill_s2048 | OLMoE gate_up · down | 1.65 · 1.54 | 0.58 · 1.08 |
| prefill_s2048 | Qwen3-30B gate_up · down | 2.95 · 3.06 | 1.39 · 2.27 |
| prefill_s2048 | gemma-4 gate_up · down | 2.99 · 3.34 | 1.13 · 2.57 |
| prefill_s2048 | gpt-oss gate_up · down | 5.74 · 5.88 | 1.27 · 1.40 |

**vs unsloth's grouped path: decode median 4.67× (4.23–8.44), prefill
median 3.02× (1.54–5.88) — every cell, both regimes.** The v1-era receipts
(`p2_first.json`, 2026-07-13) had this class AHEAD of our prefill by
1.3–1.8×; the v4 config pass + the v6 mainloop flipped all of it, including
OLMoE — the cell class that remains a known loser against bnb-dequant
(0.58) still beats the grouped path (1.65), because the grouped path pays
dequant + materialization on top of the same GEMM work.

Same-run anchors are consistent with the registered claims: bnb-dequant
decode median 2.29× (census band 1.16–2.73), prefill pattern matches the
v6 confirmatory (OLMoE gate_up below parity, everything else above).

## Marlin: not runnable on this stack (env fact, not a result)

The marlin backend needs vLLM, and `pip install vllm` on the
torch-2.8.0+cu128 pod image dragged in torch 2.11.0+cu130, which cannot
even see the GPU there (`torch.cuda.is_available()` False) — the leg was
fenced to run AFTER the unsloth receipts were banked, and it failed
cleanly (`comp_marlin.log`). The standing marlin datapoint remains the
v1-era survey (`RESULTS-backends-and-a2000.md`): ~dequant-loop speed at
MoE decode (no grouped mode → per-expert launch storm), excellent fidelity
(2.07e-4), and a different quant format (GPTQ, not NF4) — a speed
comparator, not an alternative for bnb-format checkpoints.

## Context for the ecosystem question

Axolotl and the HF PEFT stack have no 4-bit kernel of their own — their
QLoRA forward IS bitsandbytes `Linear4bit`, so the flagship bnb-CUDA
baseline verdict (fused 2.33×/2.21× at offload scale) is the comparison
against that entire stack. This census adds the remaining named
alternative: the grouped-bf16-GEMM class loses to the fused kernel on
every census cell at both decode and prefill, because for 4-bit-stored
experts it pays the dequant round-trip the fused kernel exists to delete.
