"""Training-step cost attribution. Registered in
PREREG-train-cost-attribution.md (stamped 1c27ab12...).

Asks one question before anything is ported: what fraction of a training
step is expert-WEIGHT MOVEMENT? Everything the Stage-3 cold-path program
optimizes acts on that term.

Method is gate 1's, so the numbers are comparable: disk time is
reads_in_window * row_bytes / B_nvme at the box's MEASURED sequential
ceiling, charged against T_step(arm) - T_step(control). The sequential
ceiling makes the disk share a LOWER bound.
"""
import argparse
import json
import statistics
import time

import torch
from transformers import AutoTokenizer

from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines import hybrid_train as ht
from experts4bit_qlora.engines.placement import (force_cold_mass,
                                                 load_routing_mass,
                                                 solve_placement)
from experts4bit_qlora.loader import load_moe_4bit_streaming


def routed_per_layer(model, ids, E):
    """T2: how much of the arena does one training microbatch touch?"""
    seen = {}

    def hook(lid):
        def f(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            if not torch.is_tensor(t):
                return
            if t.is_floating_point() and t.shape[-1] == E:
                idx = torch.topk(t.reshape(-1, E).float(), 8, -1).indices
            elif not t.is_floating_point():
                idx = t
            else:
                return
            seen.setdefault(lid, set()).update(idx.reshape(-1).tolist())
        return f

    hs = []
    for n, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is not None and w.dim() == 2 and w.shape[0] == E and n.endswith("gate"):
            d = [p for p in n.split(".") if p.isdigit()]
            if d:
                hs.append(m.register_forward_hook(hook(int(d[0]))))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    if not seen:
        return None
    return {"layers": len(seen),
            "mean_routed": sum(len(v) for v in seen.values()) / len(seen),
            "max_routed": max(len(v) for v in seen.values()), "n_experts": E}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/olmoe")
    ap.add_argument("--arena", default="/root/models/olmoe.arena")
    ap.add_argument("--profile", default="/root/olmoe_profile.jsonl")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sweep", default="0.0,0.05,0.20")
    ap.add_argument("--hot-rows", type=int, default=512)
    ap.add_argument("--protected", type=int, default=0,
                    help="0 = default (= hot_rows, nothing reclaimable)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    calib = json.load(open(a.calib))
    nv = (calib["cpu_bench"].get("nvme") or {})
    b_nvme = None
    for v in nv.values():
        if isinstance(v, dict) and "gbs" in v:
            b_nvme = max(b_nvme or 0, v["gbs"])
        if isinstance(v, list):
            for e in v:
                if isinstance(e, dict) and "gbs" in e:
                    b_nvme = max(b_nvme or 0, e["gbs"])
    b_nvme = b_nvme or 5.0

    from nvme_arena import load_index
    idx = load_index(a.arena)
    L, E, rb = idx["n_layers"], idx["n_experts_per_layer"], idx["row_bytes"]
    mass, _ = load_routing_mass(a.profile, L, E)
    base = solve_placement(n_layers=L, n_experts=E, bytes_per_expert=rb,
                           vram_budget_bytes=int(0.25 * L * E) * rb,
                           dram_budget_bytes=L * E * rb, calibration=a.calib,
                           profile_path=a.profile, top_k=8, batch=1)
    assert not base["tiers"]["nvme"], "control must have no cold experts"
    print("B_nvme %.2f GB/s | row %.2f MB | L=%d E=%d" % (
        b_nvme, rb / 1e6, L, E))

    tk = AutoTokenizer.from_pretrained(a.model)
    PROSE = ("The question of how memory works has occupied philosophers "
             "and scientists for centuries. When we recall an event we do "
             "not replay a recording; we reconstruct it. ")
    ids = tk(PROSE * 40, return_tensors="pt").input_ids[:, :a.seq].to("cuda")

    out = {"schema": "e4b-train-attrib/1", "prereg_sha256_prefix": "1c27ab12",
           "config": vars(a), "b_nvme_gbs": b_nvme, "row_bytes": rb,
           "arms": {}}

    for frac in [float(x) for x in a.sweep.split(",")]:
        man = base if frac == 0 else force_cold_mass(
            base, mass, frac, order="tail", source="dram")
        tag = "control" if frac == 0 else "cold-%d" % round(frac * 100)
        model, _ = load_moe_4bit_streaming(
            a.model, device="cuda", dtype=torch.bfloat16, r=8, alpha=16,
            quant_type="nf4", arena=a.arena)
        kw = {"hot_rows": a.hot_rows}
        if a.protected:
            kw["protected_rows"] = a.protected
        n = ht.enable_hybrid_train(model, a.arena, man, **kw)
        if n == 0:
            print("  %-9s NOT-ENGAGED" % tag)
            out["arms"][tag] = {"engaged": 0}
            del model
            torch.cuda.empty_cache()
            continue
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if tag == "control":
            out["routed_per_layer"] = routed_per_layer(model, ids, E)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-5) if params else None
        per = []
        for i in range(a.warmup + a.steps):
            if i == a.warmup:
                torch.cuda.synchronize()
                pre = hy.cold_stats(model)
                pre_t = {}
                for _, mm in model.named_modules():
                    t = getattr(mm, "_e4b_cold_tier", None)
                    if t is not None:
                        pre_t = t.stats()
            t0 = time.perf_counter_ns()
            o = model(ids, labels=ids)
            o.loss.backward()
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= a.warmup:
                per.append(time.perf_counter_ns() - t0)
        post = hy.cold_stats(model)
        rec = {"engaged": n, "median_ns": statistics.median(per),
               "p10_ns": sorted(per)[max(0, len(per) // 10 - 1)],
               "p90_ns": sorted(per)[min(len(per) - 1, 9 * len(per) // 10)],
               "reads_in_window": post.get("disk_reads", 0) - pre.get("disk_reads", 0),
               "trainable_params": sum(p.numel() for p in params)}
        # Read the tier DIRECTLY: cold_stats does not surface hits/misses,
        # and inferring "no reads" from a missing key would be exactly the
        # instrument failure this measurement exists to avoid.
        tier = None
        for _, mm in model.named_modules():
            t = getattr(mm, "_e4b_cold_tier", None)
            if t is not None:
                tier = t
        ts = tier.stats() if tier is not None else {}
        rec["tier_seen"] = tier is not None
        for k in ("hits", "misses", "demand_misses", "evictions",
                  "resurrections", "logical_evictions",
                  "reuse_before_overwrite", "disk_reads", "hot_rows"):
            rec[k] = ts.get(k)
        rec["hits_in_window"] = (ts.get("hits", 0) or 0) - (pre_t.get("hits", 0) or 0)
        rec["misses_in_window"] = (ts.get("misses", 0) or 0) - (pre_t.get("misses", 0) or 0)
        out["arms"][tag] = rec
        print("  %-9s patched=%2d median %8.2f ms | win_reads %6d | "
              "hits/miss in window %7s/%-7s (tier_seen=%s)" % (
                  tag, n, rec["median_ns"] / 1e6, rec["reads_in_window"],
                  rec["hits_in_window"], rec["misses_in_window"],
                  rec["tier_seen"]))
        ht.disable_hybrid_train(model)
        del model, opt
        torch.cuda.empty_cache()
        json.dump(out, open(a.out, "w"), indent=2)

    ctl = out["arms"].get("control", {}).get("median_ns")
    if ctl:
        print()
        for tag, v in out["arms"].items():
            if tag == "control" or not v.get("median_ns"):
                continue
            d_ms = (v["median_ns"] - ctl) / 1e6
            disk_ms = (v["reads_in_window"] * rb / 1e9) / b_nvme * 1000.0 / len(
                [1]) / max(1, a.steps)
            v["delta_ms"] = d_ms
            v["disk_ms_per_step"] = disk_ms
            v["disk_share_of_delta"] = (disk_ms / d_ms) if d_ms > 0 else None
            v["delta_share_of_step"] = d_ms / (ctl / 1e6)
            print("%-9s delta %+7.2f ms/step (%.1f%% of step) | disk %6.3f ms "
                  "-> %s of the delta" % (
                      tag, d_ms, 100 * v["delta_share_of_step"], disk_ms,
                      ("%.1f%%" % (100 * v["disk_share_of_delta"]))
                      if v["disk_share_of_delta"] else "n/a"))
    if out.get("routed_per_layer"):
        r = out["routed_per_layer"]
        print("\nT2 routed/layer: mean %.1f of %d (%.1f%%), max %d" % (
            r["mean_routed"], r["n_experts"],
            100 * r["mean_routed"] / r["n_experts"], r["max_routed"]))
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
