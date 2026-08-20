# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Can moving experts to NVMe change the arithmetic of experts that did NOT move?

`_HybridTier.dram_thin` is decided at enable time from the layer's TOTAL DRAM
population: `0 < n_dram <= offload_thin_uniq`. `force_cold_mass` shrinks that
population. So a layer that was above the threshold in the control can fall to
or below it in a cold arm, and the DRAM experts that STAYED flip from the CPU
tier (fp32) to the GPU (compute dtype) — a destination change nobody asked for,
on experts the arm did not touch.

Two placements, differing only in whether 3 experts moved dram -> nvme, run at
cold_dest="cpu" so every moved expert stays on the CPU. If the residual is real,
the arms differ even though nothing that moved changed engine.
"""
import json
import os
import struct
import tempfile

import torch
from nvme_arena import bake_expert_tensors

from experts4bit_qlora import Experts4bit
from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS

E, INTER, H, K = 16, 512, 1024, 4
VRAM = [0, 1]
STAY = [2, 3, 4, 5]          # DRAM in BOTH arms — these never move
MOVE = [6, 7, 8]             # DRAM in control, NVMe in the cold arm
NVME = [9, 10, 11, 12, 13, 14, 15]


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = K, E, H
        self.norm_topk_prob = False
        g = torch.Generator().manual_seed(9)
        self.weight = torch.nn.Parameter(torch.randn(E, H, generator=g) * 0.3)


class _Block(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.router, self.experts = _Router(), experts


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _manifest(spec):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for tier, ids in spec.items():
        tiers[tier] += [[0, e] for e in ids]
    return {"schema": "e4b-placement/1", "tiers": tiers,
            "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}


g = torch.Generator().manual_seed(11)
mod = Experts4bit.from_float(
    (torch.randn(E, 2 * INTER, H, generator=g) * (H ** -0.5)).to(torch.bfloat16),
    (torch.randn(E, H, INTER, generator=g) * (INTER ** -0.5)).to(torch.bfloat16),
    has_gate=True, activation=torch.nn.functional.silu,
    quant_type="nf4", compute_dtype=torch.bfloat16)

dt = {torch.uint8: "U8", torch.float32: "F32"}
n1, k1 = mod._gate_up_shape
n2, k2 = mod._down_shape
payload = {"nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
           "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
           "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
           "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float()}
tmp = tempfile.mkdtemp()
snap = os.path.join(tmp, "snap")
os.makedirs(snap)
tensors = {}
for kind, stack in payload.items():
    for e in range(E):
        t = stack[e].contiguous().cpu()
        tensors[f"model.layers.0.mlp.experts.{e}.{kind}"] = (
            tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
with open(os.path.join(snap, "model.safetensors"), "wb") as fh:
    fh.write(_st_bytes(tensors))
arena = os.path.join(tmp, "m.arena")
bake_expert_tensors(snap, arena,
                    name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
                    kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)

model = torch.nn.ModuleList([_Block(mod.to("cuda"))])
T = 8
torch.manual_seed(3)
hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.5
wts = torch.rand(T, K, device="cuda", dtype=torch.bfloat16)
# route only at experts that STAY in DRAM in both arms, plus the movers
pool = STAY + MOVE
idx = torch.stack([torch.tensor([pool[(t * K + s) % len(pool)] for t in range(T)])
                   for s in range(K)], dim=1).cuda()

CONTROL = _manifest({"vram": VRAM, "dram": STAY + MOVE, "nvme": NVME})
COLD    = _manifest({"vram": VRAM, "dram": STAY,        "nvme": NVME + MOVE})


def run(man, thin):
    hy.enable_hybrid_tier(model, arena, man, hot_rows=E, cold_dest="cpu",
                          offload_thin_uniq=thin)
    try:
        st = model[0].experts._hot_residency
        flags = (int(st.is_dram.sum()), bool(st.dram_thin))
        with torch.no_grad():
            y = model[0].experts(hidden, idx, wts).float().clone()
    finally:
        hy.disable_hybrid_tier(model)
    return y, flags


print(f"routed experts: {sorted(set(idx.flatten().tolist()))}")
print(f"STAY (dram in both arms) = {STAY}   MOVE (dram -> nvme) = {MOVE}\n")
print(f"{'offload_thin_uniq':>18}  {'control n_dram/thin':>20}  {'cold n_dram/thin':>18}"
      f"  {'rel RMS':>10}  bitwise")
for thin in (None, 4, 8):
    yc, fc = run(CONTROL, thin)
    yk, fk = run(COLD, thin)
    rel = ((yc - yk).pow(2).mean().sqrt() / yc.pow(2).mean().sqrt()).item()
    print(f"{str(thin):>18}  {str(fc):>20}  {str(fk):>18}  {rel:10.3e}"
          f"  {torch.equal(yc, yk)}")
