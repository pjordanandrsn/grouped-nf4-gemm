"""What autograd saves for ONE fused e4b MoE layer forward (ExpertsLoRA, Qwen3-30B-A3B shape, 380 tokens, skewed router):
every saved tensor's shape/dtype/bytes via saved_tensors_hooks, deduplicated by storage, largest first."""
import collections, torch
from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_fast_train
dev = "cuda"; E, H, I, K, T = 128, 2048, 768, 8, 380
torch.manual_seed(0)
base = Experts4bit.from_float(torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=dev) * 0.02,
                              torch.randn(E, H, I, dtype=torch.bfloat16, device=dev) * 0.02, quant_type="nf4", compute_dtype=torch.bfloat16)
import sys
adt = torch.float32 if "fp32" in sys.argv else torch.bfloat16
mod = ExpertsLoRA(base, r=16, alpha=16, dtype=adt).to(dev).train()
assert enable_fast_train(mod, dgrad=True) == 1
g = torch.Generator(device=dev).manual_seed(5)
hs = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g).requires_grad_(True)
logits = torch.randn(T, E, device=dev, generator=g) + torch.linspace(1.5, -1.5, E, device=dev)
w, idx = logits.softmax(-1).topk(K, dim=1)
w = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16).requires_grad_(True)
seen = {}
def pack(t):
    key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
    if key not in seen and not isinstance(t, torch.nn.Parameter) and t.data_ptr() not in (hs.data_ptr(), w.data_ptr()):
        seen[key] = (tuple(t.shape), str(t.dtype).replace("torch.", ""), t.untyped_storage().nbytes())
    return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    out = mod(hs, idx, w)
tot = sum(v[2] for v in seen.values())
print("adapter dtype", adt, "| saved tensors (unique storages, excl. params/inputs):", len(seen), "total MB %.1f" % (tot / 2**20))
for k, (shape, dt, nb) in sorted(seen.items(), key=lambda kv: -kv[1][2])[:14]:
    print("  %8.2f MB  %-8s %s" % (nb / 2**20, dt, shape))
