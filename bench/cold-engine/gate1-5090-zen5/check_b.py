"""Check (b) + (c): is cold-CPU's 0.0703 the dram_thin flip, or something else?

The hypothesis in the RESULTS reconciliation: `_HybridTier.dram_thin` is
decided at enable time from a layer's TOTAL DRAM population, and
force_cold_mass shrinks that population — so DRAM experts that STAYED can
flip from the CPU tier to the GPU between control and cold arm, a
destination change on unmoved experts.

This records the engine state per module in BOTH arms (check c) so the
question is answerable from the artifact instead of inferred, then reports
whether cold-CPU reproduces its matched DRAM control bitwise.
"""
import json

import torch
from transformers import AutoTokenizer

from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines.placement import (force_cold_mass,
                                                 load_routing_mass,
                                                 solve_placement)
from experts4bit_qlora.loader import load_moe_4bit_streaming
from nvme_arena import load_index

ARENA, MODEL = "/root/models/olmoe.arena", "/root/models/olmoe"
CALIB = "/root/gnf4/bench/cold-engine/receipts-hybrid-calib-vram-arm.json"
idx = load_index(ARENA)
L, E, rb = idx["n_layers"], idx["n_experts_per_layer"], idx["row_bytes"]
mass, _ = load_routing_mass("/root/olmoe_profile.jsonl", L, E)
base = solve_placement(n_layers=L, n_experts=E, bytes_per_expert=rb,
                       vram_budget_bytes=int(0.25 * L * E) * rb,
                       dram_budget_bytes=L * E * rb, calibration=CALIB,
                       profile_path="/root/olmoe_profile.jsonl",
                       top_k=8, batch=1)
forced = force_cold_mass(base, mass, 0.05, order="tail", source="dram")

PROSE = ("The question of how memory works has occupied philosophers and "
         "scientists for centuries. When we recall an event, we do not "
         "replay a recording; we reconstruct it, and the reconstruction "
         "is shaped by everything we have learned since. This is why "
         "eyewitness testimony is less reliable than juries assume. ")
tk = AutoTokenizer.from_pretrained(MODEL)
ids = tk(PROSE * 4, return_tensors="pt").input_ids[:, :64].to("cuda")

out = {"schema": "e4b-tribrid-checkb/1", "arms": {}}


def run(tag, man, dest):
    model, _ = load_moe_4bit_streaming(MODEL, device="cuda",
                                       dtype=torch.bfloat16, r=8, alpha=16,
                                       quant_type="nf4", arena=ARENA)
    n = hy.enable_hybrid_tier(model, ARENA, man, hot_rows=384, cold_dest=dest)
    assert n == 16, n
    # (c): the engine state that decides where an expert EXECUTES
    st = []
    for nm, m in model.named_modules():
        h = getattr(m, "_hot_residency", None)
        if h is None:
            continue
        st.append({"n_dram": int(h.is_dram.sum()),
                   "n_vram": int(h.hot_ids.numel()),
                   "n_nvme": len(h.nvme_set),
                   "dram_thin": bool(getattr(h, "dram_thin", False)),
                   "offload_rows": getattr(h, "offload_rows", None),
                   "fused_ffn": bool(getattr(h, "fused_ffn", False)),
                   "gpu_only": bool(getattr(h, "_gpu_only", False)),
                   "cold_dest": getattr(h, "_cold_dest", None),
                   "protected_rows": getattr(
                       getattr(m, "_e4b_cold_tier", None), "protected_rows", None)})
    toks = []
    with torch.no_grad():
        for _ in range(3):
            o = model(ids, use_cache=True)
            c, p = o.logits[:, -1:].argmax(-1), o.past_key_values
            for _ in range(4):
                o = model(c, past_key_values=p, use_cache=True)
                c, p = o.logits[:, -1:].argmax(-1), o.past_key_values
        o = model(ids, use_cache=True)
        p, c = o.past_key_values, o.logits[:, -1:].argmax(-1)
        for _ in range(128):
            o = model(c, past_key_values=p, use_cache=True)
            p, c = o.past_key_values, o.logits[:, -1:].argmax(-1)
            toks.append(int(c.item()))
        logits = o.logits[:, -1, :].detach().float().cpu()
    out["arms"][tag] = {"engine": st, "tokens": toks}
    thin = sum(1 for x in st if x["dram_thin"])
    print("%-12s dram_thin layers=%d/16 | n_dram range %d..%d | cold_dest=%s "
          "| offload_rows=%s | fused_ffn=%s" % (
              tag, thin, min(x["n_dram"] for x in st),
              max(x["n_dram"] for x in st), st[0]["cold_dest"],
              st[0]["offload_rows"], st[0]["fused_ffn"]))
    hy.disable_hybrid_tier(model)
    del model
    torch.cuda.empty_cache()
    return logits, toks


ref_l, ref_t = run("control", base, "gpu")
cpu_l, cpu_t = run("cold-cpu", forced, "cpu")
d = (cpu_l - ref_l).abs().max().item()
out["max_abs_logit_diff"] = d
out["tokens_match"] = cpu_t == ref_t
out["bitwise"] = d == 0.0
print("\ncold-CPU vs matched DRAM control: dmax=%.6f  bitwise=%s  tokens_match=%s"
      % (d, d == 0.0, cpu_t == ref_t))
# did any layer's DRAM population cross a thin threshold between arms?
a, b = out["arms"]["control"]["engine"], out["arms"]["cold-cpu"]["engine"]
print("per-layer n_dram control->cold:",
      [(x["n_dram"], y["n_dram"]) for x, y in zip(a, b)][:6], "...")
json.dump(out, open("/root/check_b.json", "w"), indent=2)
