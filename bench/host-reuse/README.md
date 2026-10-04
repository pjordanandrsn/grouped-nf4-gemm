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

## NF4_QLORA_COMPACT_DELTA and the checkpoint policy it enables

**`moe_host.py`.** The routing weights now carry grad, as they do in training.

**`moe_saved.py [bf16|fp32]`.** Tallies what autograd saves for one fused MoE layer forward: every saved tensor, deduplicated by
storage, excluding parameters and inputs. With the skewed router at 380 tokens:

| adapters | `NF4_QLORA_COMPACT_DELTA=0` | `=1` |
|---|---|---|
| bf16 | 133.2 MB | 54.2 MB |
| fp32 | 229.1 MB | 54.9 MB |

**`ckpt_ab.py --mode layer|attn|none`.** Trains a 4-layer Qwen3-30B-A3B slice (TC1 token rows, mb2 x accum 4, fp32 attention LoRA)
under three checkpoint policies:
- `layer`: Hugging Face's per-decoder-layer checkpointing;
- `attn`: only `self_attn` is checkpointed, and the MoE activations are kept;
- `none`.

`--check` saves every trainable gradient after one micro-batch under deterministic mode. RTX A2000, `GNF4_HOST_REUSE=1`, 2026-10-04:

| policy | compact | step s (median of 6, two runs) | peak allocated GB |
|---|---|---|---|
| layer | off, no reuse | 1.278 | 4.335 |
| layer | on | 1.286 / 1.297 | 4.335 |
| attn | off | 0.969 / 0.976 | 5.72 |
| attn | on | 0.975 / 0.981 | 4.904 |
| none | off | 0.931 / 0.944 | 5.98 |

All 48 trainable gradients are `torch.equal` across `layer`, `attn` and `none`, and with the flag off and on.
