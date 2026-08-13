#!/usr/bin/env python3
"""LEG 3 — the same question as leg 2, on an instrument that pairs at the
iteration instead of the block.

Legs 1 and 2 both died of the same thing and it was not the kernels. Each timed
a whole block of one arm, then a whole block of the other, and divided; that
exposes every ratio to drift on any timescale longer than a block. Leg 1 lost a
device to a fixture build leaving the GPU cold (amendment 1). Leg 2's amendment
2 then lengthened blocks to meet the registered 250 ms target and made things
WORSE -- 32/32 live at 89 ms became 26/32 at 258 ms, median |1 - self-pair|
doubling from 0.0012 to 0.0027 -- while the ratios themselves moved by a median
of 0.999. Longer blocks put the paired blocks further apart, which is the wrong
direction when the noise is between-block drift.

So this leg pairs at the iteration: one call of `G_base`, one call of `D_base`,
ratio, repeat, order alternating. Any drift slower than a single pair cancels
inside the pair. See `interleave.py`, whose tests inject drift and show the
interleaved statistic is exact where the block statistic is badly biased.

EVERY CELL ALSO REPORTS THE BLOCK STATISTIC, computed from the SAME collected
per-call timings. Nothing else in this repo can show what the pairing bought;
here it is per cell, from identical data, at no extra cost.

Arms are leg 2's, unchanged: G_base / D_base (the primary), G_full / D_full
(the bridge to leg 1), the delta alone (the floor). `expert_ids` is the
documented list form. Amendment 1's warm-up is carried.
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import os
import statistics
import sys
from pathlib import Path

import torch

_ROOT = Path(os.environ.get("DQF_REPO", Path.cwd())).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(f"DQF_REPO/cwd={_ROOT} is not the repo root. Set DQF_REPO.")
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402



def _load(name, path):
    s = _iu.spec_from_file_location(name, path)
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


dqf1 = _load("dqf1", _ROOT / "bench" / "phase1" / "train_dequant_forward.py")
ff = _load("ff", _ROOT / "bench" / "phase1" / "train_dequant_forward_floorfree.py")
il = _load("il", _ROOT / "bench" / "phase1" / "interleave.py")
_warm, _DeqCount, _fidelity_sub = dqf1._warm, dqf1._DeqCount, dqf1._fidelity_sub
_energy = dqf1._energy
RANK, SCALING = ff.RANK, ff.SCALING


def _pairs_for(fa, fb, target_ms, lo=30, hi=600):
    """Pairs so the whole interleaved collection spans ~target_ms of device
    work. Unlike leg 2's block ceiling this does NOT trade against pairing
    quality: more pairs means more independent ratios, never a longer gap
    inside one."""
    import time as _t
    for _ in range(3):
        fa()
        fb()
    torch.cuda.synchronize()
    t0 = _t.perf_counter()
    for _ in range(5):
        fa()
        fb()
    torch.cuda.synchronize()
    per = (_t.perf_counter() - t0) * 1000.0 / 5
    return int(min(hi, max(lo, round(target_ms / max(per, 1e-3)))))


def cell(spec, regime, device, args, stack):
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    eids = [int(e) for e, _ in groups]
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()

    row = {"model": spec.model, "proj": spec.proj, "regime": regime,
           "N": spec.N, "K": spec.K, "E": spec.E, "top_k": spec.top_k,
           "groups": len(groups), "rows": a_cat.shape[0], "rank": RANK,
           "pairing": "iteration-level interleaved, order alternating"}

    lora_A = (torch.randn(spec.E, RANK, spec.K, device=device,
                          dtype=torch.bfloat16) * 0.01).requires_grad_(True)
    lora_B = torch.zeros(spec.E, spec.N, RANK, device=device,
                         dtype=torch.bfloat16).requires_grad_(True)

    def _zero():
        for t in (a_cat, lora_A, lora_B):
            t.grad = None

    gb = ff.g_base_arm(a_cat, B_pack, A_scale, sizes, eids)
    db = ff.d_base_arm(stack, a_cat, sizes, eids)
    fwd = {"G_base": gb, "D_base": db,
           "G_full": ff.with_lora(gb, a_cat, sizes, eids, lora_A, lora_B),
           "D_full": ff.with_lora(db, a_cat, sizes, eids, lora_A, lora_B)}
    step = {}
    for k, f in fwd.items():
        def mk(f=f):
            def run():
                _zero()
                f().float().pow(2).mean().backward()
            return run
        step[k] = mk()

    try:
        # ---- wiring gate, before any timing ---------------------------------
        gate = {}
        with torch.no_grad():
            og, od = fwd["G_base"]().detach(), fwd["D_base"]().detach()
        nz = sum(1 for s in sizes if s > 0)
        with _DeqCount(stack) as c:
            with torch.no_grad():
                fwd["D_base"]()
        gate["deq_calls_D"], gate["nonempty_groups"] = c.n, nz
        gate["deq_calls_ok"] = bool(c.n >= nz)
        gate["base_rows_differing"] = int(
            ((og - od).abs().max(dim=1).values > 1e-2 * od.abs().max()).sum())
        gate["base_arms_agree"] = bool(gate["base_rows_differing"] == 0)
        for k in step:
            _zero()
            step[k]()
            g = a_cat.grad
            gate[f"grad_{k}"] = {
                "act_finite": bool(g is not None and torch.isfinite(g).all()),
                "act_nonzero": bool(g is not None and g.abs().sum() > 0)}
        with torch.no_grad():
            lora_B.copy_(torch.randn_like(lora_B) * 0.01)
        _zero()
        step["G_full"]()
        gate["gradA_at_nonzero_B"] = bool(
            lora_A.grad is not None and lora_A.grad.abs().sum() > 0)
        with torch.no_grad():
            lora_B.zero_()
        _zero()
        row["gate"] = gate

        row["fid_rows_per_group"], row["fid_max_groups"] = args.fid_rows, args.fid_groups
        row["b_rel_G"] = _fidelity_sub(stack, groups, og, sizes,
                                       args.fid_rows, args.fid_groups)
        row["b_rel_D"] = _fidelity_sub(stack, groups, od, sizes,
                                       args.fid_rows, args.fid_groups)
        row["b_rel_G_over_D"] = (row["b_rel_G"] / row["b_rel_D"]
                                 if row["b_rel_D"] else None)
        del og, od
        torch.cuda.empty_cache()

        # ---- interleaved timing ---------------------------------------------
        _warm(step["G_base"], args.warm_s)
        pairs = _pairs_for(step["G_base"], step["D_base"], args.target_ms)
        row["pairs"] = pairs
        row["warm_s"] = args.warm_s

        # SELF-PAIR: the same arm against itself, same machinery. This is the
        # instrument floor measured exactly as the primary is measured.
        ta, tb, orders = il.interleaved_pairs(step["G_base"], step["G_base"],
                                              pairs, torch_mod=torch)
        row["selfpair"] = il.pair_stats(ta, tb, orders)
        row["selfpair_block"] = il.block_ratio(ta, tb)

        # PRIMARY
        ta, tb, orders = il.interleaved_pairs(step["G_base"], step["D_base"],
                                              pairs, torch_mod=torch)
        s = il.pair_stats(ta, tb, orders)
        row["dbase_over_gbase"] = s["ratio_median"]
        row["primary"] = s
        # the OLD statistic on the SAME data -- what block pairing would have said
        row["dbase_over_gbase_blockstat"] = il.block_ratio(ta, tb)

        # BRIDGE to leg 1 / leg 2
        ta, tb, orders = il.interleaved_pairs(step["G_full"], step["D_full"],
                                              pairs, torch_mod=torch)
        s2 = il.pair_stats(ta, tb, orders)
        row["dfull_over_gfull"] = s2["ratio_median"]
        row["full"] = s2
        row["dfull_over_gfull_blockstat"] = il.block_ratio(ta, tb)

        if args.energy:
            en = {}
            for k in ("G_base", "D_base"):
                w, j, meth, n = _energy(step[k], args.energy_s)
                en[k] = {"watts": w, "j_per_step": j, "method": meth, "samples": n}
            if en["G_base"]["j_per_step"] and en["D_base"]["j_per_step"]:
                row["j_ratio_dbase_over_gbase"] = (
                    en["D_base"]["j_per_step"] / en["G_base"]["j_per_step"])
            row["energy"] = en
        row["status"] = "ok"
    except Exception as e:  # pragma: no cover
        row.update({"status": "skipped",
                    "reason": f"{type(e).__name__}: {str(e)[:220]}"})
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*",
                    default=["decode_bs1", "decode_m8", "decode_m32",
                             "tokbudget_2048"])
    ap.add_argument("--target-ms", type=float, default=500.0,
                    help="device work per interleaved collection; sets pairs")
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--energy-s", type=float, default=1.5)
    ap.add_argument("--fid-rows", type=int, default=16)
    ap.add_argument("--fid-groups", type=int, default=32)
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="il1")
    args = ap.parse_args()

    out = {"leg": "3 - interleaved pairing",
           "prereg": "kernel/prereg_dequant_forward_interleaved.json",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "rank": RANK,
           "regimes": args.regimes, "target_ms": args.target_ms,
           "warm_s": args.warm_s, "rows": []}
    try:
        import bitsandbytes as bnb
        out["bitsandbytes"] = bnb.__version__
    except Exception:
        pass

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"interleaved_{args.tag}.json"
    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        stack = H.QuantStack(spec, "cuda")
        for regime in args.regimes:
            r = cell(spec, regime, "cuda", args, stack)
            out["rows"].append(r)
            print(json.dumps(r), flush=True)
            art.write_text(json.dumps(out, indent=1, default=str))
        del stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    if ok:
        print("MEDIAN dbase/gbase=%.3f (blockstat %.3f)  selfpair=%.4f  cells=%d" % (
            statistics.median(r["dbase_over_gbase"] for r in ok),
            statistics.median(r["dbase_over_gbase_blockstat"] for r in ok),
            statistics.median(r["selfpair"]["ratio_median"] for r in ok), len(ok)))
    print("INTERLEAVED_DONE")


if __name__ == "__main__":
    main()
