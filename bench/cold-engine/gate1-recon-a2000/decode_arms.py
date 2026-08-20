# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Gate 1's equivalence clause, end to end, on a greedy decode.

The layer-level probes (`destination_gap.py`, `thin_flip.py`) answer what one
layer does. This answers what a GENERATION does, which is the form the clause
is written in: a stack of MoE layers, a fixed prompt, greedy argmax, token
sequences compared against a reference.

Two questions, each an arm pair rather than an argument:

  (1) Does `force_cold_mass(source=...)` decide which cold destination matches?
      Under `source="dram"` the movers leave the CPU tier, so `cold_dest="cpu"`
      holds their arithmetic and `"gpu"` does not. Under `source="vram"` they
      leave the GPU, and THE ARMS SHOULD SWAP.

  (2) Does `offload_thin_uniq` break an otherwise-matched arm? Moving experts
      out of DRAM shrinks the population `dram_thin` is decided from, so
      experts that did NOT move can change destination.

Synthetic weights and a synthetic head: the mechanism is a property of the
engine, not of a checkpoint, and a synthetic stack runs in seconds on any GPU
instead of needing a 7B model resident. What this therefore does NOT give is
OLMoE's own divergence indices — that is still owed to a re-run of the sweep.
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

L = int(os.environ.get("PROBE_LAYERS", "8"))       # MoE layers
E, INTER, H, K = 32, 256, 512, 4
VOCAB, PROMPT, STEPS = 512, 16, int(os.environ.get("PROBE_STEPS", "128"))


class _Router(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = K, E, H
        self.norm_topk_prob = False
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(E, H, generator=g) * 0.3)


class _Block(torch.nn.Module):
    def __init__(self, experts, seed):
        super().__init__()
        self.router, self.experts = _Router(seed).to("cuda"), experts

    def forward(self, x):
        logits = torch.nn.functional.linear(x.float(), self.router.weight.float())
        w, idx = torch.topk(torch.softmax(logits, -1), k=K, dim=-1)
        return x + self.experts(x, idx, w.to(x.dtype))


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _manifest(per_layer):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for lay, spec in enumerate(per_layer):
        for tier, ids in spec.items():
            tiers[tier] += [[lay, int(e)] for e in ids]
    return {"schema": "e4b-placement/1", "tiers": tiers,
            "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}


# ---------------------------------------------------------------- the model
tmp = tempfile.mkdtemp()
snap = os.path.join(tmp, "snap")
os.makedirs(snap)
dt = {torch.uint8: "U8", torch.float32: "F32"}
tensors, blocks = {}, []
for lay in range(L):
    g = torch.Generator().manual_seed(1000 + lay)
    mod = Experts4bit.from_float(
        (torch.randn(E, 2 * INTER, H, generator=g) * (H ** -0.5)).to(torch.bfloat16),
        (torch.randn(E, H, INTER, generator=g) * (INTER ** -0.5)).to(torch.bfloat16),
        has_gate=True, activation=torch.nn.functional.silu,
        quant_type="nf4", compute_dtype=torch.bfloat16)
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    payload = {"nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
               "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
               "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
               "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float()}
    for kind, stack in payload.items():
        for e in range(E):
            t = stack[e].contiguous().cpu()
            tensors[f"model.layers.{lay}.mlp.experts.{e}.{kind}"] = (
                tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
    blocks.append(_Block(mod.to("cuda"), seed=2000 + lay))
with open(os.path.join(snap, "model.safetensors"), "wb") as fh:
    fh.write(_st_bytes(tensors))
arena = os.path.join(tmp, "m.arena")
bake_expert_tensors(snap, arena,
                    name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
                    kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)
model = torch.nn.ModuleList(blocks)

g = torch.Generator().manual_seed(7)
EMB = (torch.randn(VOCAB, H, generator=g) * 0.5).to("cuda", torch.bfloat16)
HEAD = (torch.randn(VOCAB, H, generator=g) * (H ** -0.5)).to("cuda", torch.bfloat16)
PROMPT_IDS = torch.randint(0, VOCAB, (PROMPT,), generator=g).tolist()


def generate():
    """Greedy decode, one token at a time — the shape gate 1 measures."""
    ids, logits_hist = list(PROMPT_IDS), []
    with torch.no_grad():
        for _ in range(STEPS):
            x = EMB.index_select(0, torch.tensor(ids, device="cuda"))
            for blk in model:
                x = blk(x)
            lg = torch.nn.functional.linear(x[-1].float(), HEAD.float())
            logits_hist.append(lg.clone())
            ids.append(int(lg.argmax()))
    return ids[PROMPT:], torch.stack(logits_hist)


# ------------------------------------------------------------- the arms ----
# The CONTROL HAS AN EMPTY NVMe TIER, which is the shape gate 1 measures
# ("control: all VRAM+DRAM, nvme empty"). This matters more than it looks:
# `cold_dest` applies to EVERY cold expert, so a control that already has an
# NVMe population makes the cold-CPU arm switch those experts' destination too,
# and the arm stops being matched for a reason that has nothing to do with the
# movers. A first version of this probe made exactly that mistake and measured
# its own confound (0.1068) — the same error class as #171, one layer up.
# PROBE_DRAM sizes the DRAM population, which is what decides whether
# `offload_thin_uniq` can engage at all: `dram_thin` is `0 < n_dram <= thin`,
# and `force_cold_mass` shrinks n_dram by the movers. A wide population (the
# default) leaves the knob inert in every arm and tests only `cold_dest`; a
# population that STRADDLES the threshold is what exercises the flip.
_ND = int(os.environ.get("PROBE_DRAM", str(E - 8)))
VRAM = list(range(0, E - _ND))
DRAM = list(range(E - _ND, E))
MOVE_FROM_DRAM = DRAM[:4]
MOVE_FROM_VRAM = VRAM[:4]


def placement(source, moved):
    v, d, n = list(VRAM), list(DRAM), []
    if moved:
        n = MOVE_FROM_DRAM if source == "dram" else MOVE_FROM_VRAM
        d = [e for e in d if e not in n]
        v = [e for e in v if e not in n]
    return _manifest([{"vram": v, "dram": d, "nvme": n}] * L)


def run(man, dest, thin):
    hy.enable_hybrid_tier(model, arena, man, hot_rows=E * L, cold_dest=dest,
                          offload_thin_uniq=thin)
    try:
        toks, lg = generate()
        st = hy.cold_stats(model)
    finally:
        hy.disable_hybrid_tier(model)
    return toks, lg, st


def compare(name, ref, got):
    (rt, rl), (gt, gl) = ref, got
    d = (gl - rl).abs().max().item()
    match = rt == gt
    first = next((i for i, (a, b) in enumerate(zip(rt, gt)) if a != b), None)
    print(f"  {name:<34}{d:>12.4f}{str(match):>9}{str(first):>7}")
    return match


print(f"layers={L} experts={E} k={K} steps={STEPS}  "
      f"movers=4/layer ({4 * L} of {E * L} cells)  "
      f"vram={len(VRAM)} dram={len(DRAM)} -> {len(DRAM) - 4} after the move\n")

for thin in (None, 4):
    print(f"offload_thin_uniq={thin}")
    base = run(placement("dram", False), "gpu", thin)   # nvme empty either way
    ref = (base[0], base[1])
    b2 = run(placement("dram", False), "gpu", thin)
    print(f" {'control self-pair':<35}"
          f"{(b2[1] - ref[1]).abs().max().item():>12.4f}"
          f"{str(b2[0] == ref[0]):>9}\n")
    for source in ("dram", "vram"):
        print(f" force_cold_mass(source={source!r})"
              f"{'max|dlogit|':>21}{'tokens':>9}{'firstdiv':>7}")
        for dest in ("gpu", "cpu"):
            toks, lg, st = run(placement(source, True), dest, thin)
            compare(f"cold_dest={dest!r}  (cold rows "
                    f"{st['cold_rows_gpu']}/{st['cold_rows_cpu']})",
                    ref, (toks, lg))
        print()
