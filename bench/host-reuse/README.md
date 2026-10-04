# GNF4_HOST_REUSE: host time of one fused MoE training layer

`moe_host.py` builds one experts4bit-qlora `ExpertsLoRA` layer at Qwen3-30B-A3B's per-layer shape: E=128, hidden 2048,
moe intermediate 768, top-8, r=16, alpha=16, bf16 adapters. It patches the layer with `enable_fast_train(dgrad=True)` and
routes 380 tokens through a skewed router. It then times the forward and the backward with no profiler. Each phase starts
with the GPU idle, so the forward's one routing read waits only for the routing kernels, and each phase's wall time is
close to its host work.

- `--check FILE` saves the output and every gradient.
- `cmp_check.py A B` compares two such files with `torch.equal`.
- `--deterministic` turns on `torch.use_deterministic_algorithms`. Without it, the input gradient varies from run to run
  through the caller's atomic token gather, whatever the flag says.
- `--cprofile` prints the forward's top functions by own time.

```sh
for R in 0 1 0 1 0 1; do GNF4_HOST_REUSE=$R PYTHONPATH=<e4b>:<gnf4>/kernel python moe_host.py --reps 300 | grep fwd; done
GNF4_HOST_REUSE=0 python moe_host.py --reps 3 --deterministic --check off.pt
GNF4_HOST_REUSE=1 python moe_host.py --reps 3 --deterministic --check on.pt
python cmp_check.py off.pt on.pt
```

## Results

RTX A2000, torch 2.8.0+cu128, Triton 3.4.0, experts4bit-qlora 52940233, measured 2026-10-04. The host was shared with
other jobs. Three interleaved off/on pairs of 300 repetitions:

| phase | off, median host ms | on, median host ms |
|---|---|---|
| forward | 3.193 / 3.150 / 2.947 | 2.784 / 2.822 / 2.906 |
| backward | 3.003 / 2.871 / 2.720 | 2.401 / 2.422 / 2.508 |
| backward device span | 18.93 / 19.16 / 19.38 | 18.77 / 19.01 / 19.17 |

**Transfers per layer forward + backward:** 8 → 4. The plan memo hits once per pass, on the down delta.

**Values:**
- With deterministic mode on, two flag-off runs, and a flag-off run against a flag-on run, are `torch.equal` on all six
  tensors: the output, the input gradient, and four adapter gradients.
- With it off, the output and the adapter gradients are equal across the flag, while the input gradient varies between two
  flag-off runs as well.

An earlier draft gathered the adapters with `index_select`. Its backward, an atomic `index_add_`, cost about 0.4 ms of
device time per layer backward on the A2000. That is why the shipped route is `_GatherRows`, whose backward is a scatter.
