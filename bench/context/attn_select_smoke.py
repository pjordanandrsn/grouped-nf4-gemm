# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Pre-flight for Experiment A: check the machinery before spending the run.

Three things can silently ruin the measurement, and none of them show up as an
error: attentions coming back None (H2O then selects on nothing), the keep-set
collapsing to recency (H2O then *is* the control), and eviction breaking the
chunked-prefill invariant (which is how #11's first attempt scored 330).
"""
import sys

import torch

sys.path.insert(0, "/root/g/kernel")
sys.path.insert(0, "/root/g/bench/context")
from attn_select import H2O, Recency, run  # noqa: E402
from experts4bit_qlora import NF4KVCache  # noqa: E402
from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

M = "HuggingFaceTB/SmolLM2-135M"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).cuda().eval()
m.set_attn_implementation("eager")
n_kv = getattr(m.config, "num_key_value_heads", m.config.num_attention_heads)
print(f"{M}: H_q={m.config.num_attention_heads} H_kv={n_kv} "
      f"attn={m.config._attn_implementation}")

ids = (torch.arange(256, device="cuda") % 20000).unsqueeze(0)

# 1. attentions are actually produced, and have the shape H2O assumes
with torch.no_grad():
    o = m(ids[:, :64], use_cache=False, output_attentions=True)
assert o.attentions is not None and o.attentions[0] is not None, "no attentions"
print("attentions:", len(o.attentions), tuple(o.attentions[0].shape))
assert o.attentions[0].shape[-1] == 64

# 2. no-eviction control still matches a single forward through DynamicCache --
#    the invariant that caught get_query_offset
def chunked(cache, out=64):
    with torch.no_grad():
        return torch.cat([m(ids[:, lo:lo + out], past_key_values=cache,
                            use_cache=True).logits.float()
                          for lo in range(0, 256, out)], dim=1)


mine = chunked(NF4KVCache(quantize_keys=False, quantize_values=False))
theirs = chunked(DynamicCache())
rel = ((mine - theirs).norm() / theirs.norm()).item()
print(f"chunked prefill vs DynamicCache: {rel:.3e}")
assert rel < 1e-6, rel

# 3. H2O selects something that is NOT sink+recent, and holds the budget it claims
h = H2O(4, 16, 16, n_kv)
c = NF4KVCache(quantize_keys=True, quantize_values=True)
for lo in range(0, 256, 64):
    with torch.no_grad():
        out = m(ids[:, lo:lo + 64], past_key_values=c, use_cache=True,
                output_attentions=True)
    h.observe(out.attentions)
    h.step(c)
    print(f"  after chunk @{lo:3d}: held={c.held_length(0)} seen={c.get_seq_length(0)}")
assert c.held_length(0) == 36, c.held_length(0)
assert c.get_seq_length(0) == 256, c.get_seq_length(0)

# what did it keep, and is it different from recency?
acc = h.acc[0]
print(f"  layer0 kept {acc.shape[0]} slots")
r = Recency(4, 32)
c2 = NF4KVCache(quantize_keys=True, quantize_values=True, **r.cache_kwargs())
for lo in range(0, 256, 64):
    with torch.no_grad():
        m(ids[:, lo:lo + 64], past_key_values=c2, use_cache=True)
    r.step(c2)
assert c2.held_length(0) == 36, c2.held_length(0)
k_h2o = c._load(c._k[0], torch.float32)
k_rec = c2._load(c2._k[0], torch.float32)
same = torch.equal(k_h2o, k_rec)
print(f"  H2O keep-set identical to recency at the same budget: {same}")
assert not same, "H2O collapsed to recency -- selection is doing nothing"

# 4. the run() scorer splits first/second copy correctly on an induction fixture
g = torch.Generator().manual_seed(0)
base = torch.randint(1000, 20000, (64,), generator=g)
ind = torch.cat([base, base]).unsqueeze(0).cuda()
import attn_select  # noqa: E402

attn_select.CH = 32
res = run(m, ind, attn_select.Full(), False, split_at=64)
print("induction split:", res["counts"], {k: None if v is None else round(v, 2)
                                          for k, v in res["ppl"].items()})
assert res["counts"]["first"] == 63 and res["counts"]["second"] == 64, res["counts"]
gain = res["ppl"]["first"] / res["ppl"]["second"]
print(f"  SmolLM2 induction gain (first/second ppl): {gain:.2f}x")
print("\nSMOKE OK")
