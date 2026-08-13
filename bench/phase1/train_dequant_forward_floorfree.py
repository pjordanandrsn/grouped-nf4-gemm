#!/usr/bin/env python3
"""Small-batch training axis vs dequant-on-forward, WITHOUT the shared floor.

Leg 1 (`train_dequant_forward.py`, prereg `prereg_dequant_forward.json`) found
its own small-batch criterion untrustworthy and said so: at `decode_m8` the
identical `lora_delta_grouped` call both arms make was **54-66% of the fused
arm's measured time**, against cells only 1.2-1.6 ms long. A cost added equally
to both arms pins every ratio near 1.0, so S1 was measuring the harness as much
as either kernel. Leg 1 reported that floor per cell and refused to subtract it,
because arithmetic on ratios after the fact is how instruments get talked into
saying what you wanted. This leg removes it *by construction* instead.

TWO CHANGES FROM LEG 1, both registered in
`kernel/prereg_dequant_forward_floorfree.json` before any data:

1. **`expert_ids` is passed as a python list of ints.** `gemm_4bit_grouped`'s
   documented contract is "``expert_ids [G]`` int32/list" and it builds the
   device tensor itself. Leg 1 handed it a CUDA tensor, which made
   `FusedGroupedNf4.forward`'s `[int(e) for e in expert_ids]` and
   `lora_delta_grouped`'s `enumerate(expert_ids)` do one device sync per group.
   That was the harness's choice, not the shipped path's. Using the documented
   form is a correction, not an arm change, and `--eids-tensor` reproduces leg
   1's form so the cost of the difference is measured rather than asserted.

2. **The primary comparison is BASE-ONLY** — the frozen 4-bit projection's
   forward and backward with no adapter delta on either side. That is the
   quantity the two kernels actually differ in. The full arms (base + identical
   LoRA delta) are still timed, so this leg can be laid beside leg 1, and the
   delta alone is still timed, so the floor stays visible.

Arms per cell, all fwd+bwd, quantized base frozen:
  G_base   `gemm_4bit_grouped_train` — single launch over packed bytes
  D_base   per-routed-expert `dequantize_4bit` + `F.linear` (GenON's core)
  G_full   G_base + `lora_delta_grouped`
  D_full   D_base + the identical `lora_delta_grouped`
  L        the delta alone — the floor, reported, never subtracted

Carries amendment 1's warm-up: the cell that runs first after a per-spec stack
build otherwise measures the GPU clocking back up, which voided a whole device
in leg 1.
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
import torch.nn.functional as Fn

_ROOT = Path(
    os.environ.get("DQF_REPO", os.environ.get("H2H_REPO", Path.cwd()))
).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(f"DQF_REPO/cwd={_ROOT} is not the repo root. Set DQF_REPO.")
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402
from nf4_qlora import (gemm_4bit_grouped_train, lora_delta_grouped)  # noqa: E402

# Leg 1's helpers are imported rather than copied: the timing discipline, the
# warm-up, the dequant counter and the fidelity subsample must be the SAME
# instrument, or the two legs cannot be compared.
_spec = _iu.spec_from_file_location(
    "dqf1", _ROOT / "bench" / "phase1" / "train_dequant_forward.py")
dqf1 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(dqf1)
_timed, _warm, _pilot_iters = dqf1._timed, dqf1._warm, dqf1._pilot_iters
_energy, _DeqCount, _fidelity_sub = dqf1._energy, dqf1._DeqCount, dqf1._fidelity_sub

RANK = 16
SCALING = 1.0


def g_base_arm(a_cat, B_pack, A_scale, sizes, eids):
    def run():
        return gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes, eids,
                                       dgrad_kernel=True)
    return run


def d_base_arm(stack, a_cat, sizes, eids):
    """GenON's core, with nothing else in the timed path: per routed expert,
    `dequantize_4bit` on the packed weight inside the forward, then
    `F.linear`. Same blocksize and quant_type the harness quantizes with.
    Rows arrive group-sorted, so the published module's gather degenerates to a
    contiguous slice — the same op boundary every arm here is measured at."""
    def run():
        outs, row = [], 0
        for g, e in enumerate(eids):
            n = int(sizes[g])
            if n == 0:
                continue
            outs.append(Fn.linear(a_cat[row:row + n], stack.dequant_bf16(int(e))))
            row += n
        return torch.cat(outs)
    return run


def with_lora(base, a_cat, sizes, eids, lora_A, lora_B):
    def run():
        out = base()
        d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, SCALING)
        return out + d.to(out.dtype) if d is not None else out
    return run


def cell(spec, regime, device, args, stack):
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    # THE CHANGE: the documented list form. --eids-tensor restores leg 1's.
    eids_list = [int(e) for e, _ in groups]
    eids = (torch.tensor(eids_list, dtype=torch.int32, device=device)
            if args.eids_tensor else eids_list)
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()

    row = {"model": spec.model, "proj": spec.proj, "regime": regime,
           "N": spec.N, "K": spec.K, "E": spec.E, "top_k": spec.top_k,
           "groups": len(groups), "rows": a_cat.shape[0], "rank": RANK,
           "eids_form": "tensor" if args.eids_tensor else "list"}

    lora_A = (torch.randn(spec.E, RANK, spec.K, device=device,
                          dtype=torch.bfloat16) * 0.01).requires_grad_(True)
    lora_B = torch.zeros(spec.E, spec.N, RANK, device=device,
                         dtype=torch.bfloat16).requires_grad_(True)

    def _zero():
        for t in (a_cat, lora_A, lora_B):
            t.grad = None

    gb = g_base_arm(a_cat, B_pack, A_scale, sizes, eids)
    db = d_base_arm(stack, a_cat, sizes, eids)
    fwd = {"G_base": gb, "D_base": db,
           "G_full": with_lora(gb, a_cat, sizes, eids, lora_A, lora_B),
           "D_full": with_lora(db, a_cat, sizes, eids, lora_A, lora_B)}

    step = {}
    for k, f in fwd.items():
        def mk(f=f):
            def run():
                _zero()
                f().float().pow(2).mean().backward()
            return run
        step[k] = mk()

    def lora_only():
        _zero()
        d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, SCALING)
        d.float().pow(2).mean().backward()

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
        # base arms must agree to the bf16 budget: they compute the same
        # projection by different routes, so a real disagreement is a bug.
        gate["base_max_rel"] = float(
            (og - od).abs().max() / od.abs().max().clamp_min(1e-30))
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
        # lora_A is exactly zero at LoRA init (B zero-init); positive-control it
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

        # ---- timing: warm FIRST (amendment 1) -------------------------------
        _warm(step["G_base"], args.warm_s)
        row["warm_s"] = args.warm_s
        iters = _pilot_iters(step["G_base"], args.block_ms)
        row["iters"] = iters

        t = {}
        t["gb_a"] = _timed(step["G_base"], iters)
        t["gb_b"] = _timed(step["G_base"], iters)      # ADJACENT self-pair
        row["gb_selfpair"] = t["gb_b"] / t["gb_a"]
        t["db_a"] = _timed(step["D_base"], iters)      # adjacent to gb_b
        row["dbase_over_gbase"] = t["db_a"] / t["gb_b"]     # PRIMARY
        t["db_b"] = _timed(step["D_base"], iters)      # ADJACENT self-pair
        row["db_selfpair"] = t["db_b"] / t["db_a"]

        t["gf_a"] = _timed(step["G_full"], iters)
        t["df_a"] = _timed(step["D_full"], iters)
        row["dfull_over_gfull"] = t["df_a"] / t["gf_a"]     # bridge to leg 1
        t["lora"] = _timed(lora_only, iters)
        row["lora_floor_frac_of_gbase"] = t["lora"] / t["gb_a"]
        row["lora_floor_frac_of_gfull"] = t["lora"] / t["gf_a"]

        t["gb_end"] = _timed(step["G_base"], iters)
        row["gb_drift"] = t["gb_end"] / t["gb_a"]
        row["ms"] = t

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
    ap.add_argument("--block-ms", type=float, default=250.0)
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--energy-s", type=float, default=1.5)
    ap.add_argument("--fid-rows", type=int, default=16)
    ap.add_argument("--fid-groups", type=int, default=32)
    ap.add_argument("--eids-tensor", action="store_true",
                    help="pass expert_ids as a CUDA tensor, reproducing leg 1's "
                         "form, so the cost of the documented list form is "
                         "measured rather than asserted")
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="ff1")
    args = ap.parse_args()

    out = {"leg": "small-batch training axis, floor-free",
           "prereg": "kernel/prereg_dequant_forward_floorfree.json",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "rank": RANK, "scaling": SCALING,
           "eids_form": "tensor" if args.eids_tensor else "list",
           "regimes": args.regimes, "block_ms": args.block_ms,
           "warm_s": args.warm_s, "rows": []}
    try:
        import bitsandbytes as bnb
        out["bitsandbytes"] = bnb.__version__
    except Exception:
        pass

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    artifact = dest / f"floorfree_{args.tag}.json"
    specs = H.census_specs(H.REPO / "census" / "shape_census.json", args.models)
    for spec in specs:
        stack = H.QuantStack(spec, "cuda")
        for regime in args.regimes:
            r = cell(spec, regime, "cuda", args, stack)
            out["rows"].append(r)
            print(json.dumps(r), flush=True)
            artifact.write_text(json.dumps(out, indent=1, default=str))
        del stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    if ok:
        print("MEDIAN dbase/gbase=%.3f  dfull/gfull=%.3f  floor_of_gbase=%.3f  "
              "gb_self=%.3f  cells=%d" % (
                  statistics.median(r["dbase_over_gbase"] for r in ok),
                  statistics.median(r["dfull_over_gfull"] for r in ok),
                  statistics.median(r["lora_floor_frac_of_gbase"] for r in ok),
                  statistics.median(r["gb_selfpair"] for r in ok), len(ok)))
    print("FLOORFREE_DONE")


if __name__ == "__main__":
    main()
