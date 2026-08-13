#!/usr/bin/env python3
"""E1 (report-only under prereg_dequant_forward_floorfree.json): what did leg
1's CUDA-tensor `expert_ids` form cost, versus the documented list form?

EXPLORATORY INSTRUMENT, deliberately separate from the stamped leg-2 script so
that script's frozen bytes are untouched.

WHY THIS EXISTS AS A SECOND ATTEMPT. The first ablation ran the two forms as
three separate invocations (list, tensor, list) on one pod. Self-pairs held but
DRIFT did not: 0.83-1.42 across cells, because that pod ran eight ~1.4 ms cells
per pass with a CPU-bound stack build between each and no large cell to keep
the card hot. Every cell voided under the registered rule and E1 was not
adjudicable. The defect is not the card; it is that I compared two forms across
invocations when adjacent pairing is this program's entire discipline for
exactly this reason.

So: both forms are timed ADJACENTLY INSIDE ONE CELL, on the same fixtures, in
the order list, list, tensor, tensor, list. The opening pair is the self-pair;
the tensor timing is taken against the list timing immediately before it; the
closing list timing gives drift across the comparison rather than across a
process boundary. Nothing else differs between the two timed callables -- same
activations, same packed bytes, same sizes, same arm -- so the ratio isolates
the argument form and nothing else.

Both arms are probed, because the tensor form charged BOTH: the fused arm paid
it in `FusedGroupedNf4.forward`'s `[int(e) for e in expert_ids]` and both arms
paid it again in `lora_delta_grouped`'s `enumerate(expert_ids)`.
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

_ROOT = Path(os.environ.get("DQF_REPO", Path.cwd())).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(f"DQF_REPO/cwd={_ROOT} is not the repo root. Set DQF_REPO.")
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402
from nf4_qlora import gemm_4bit_grouped_train, lora_delta_grouped  # noqa: E402

_spec = _iu.spec_from_file_location(
    "dqf1", _ROOT / "bench" / "phase1" / "train_dequant_forward.py")
dqf1 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(dqf1)
_timed, _warm, _pilot_iters = dqf1._timed, dqf1._warm, dqf1._pilot_iters

RANK = 16


def build(spec, regime, device, stack):
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    eids_list = [int(e) for e, _ in groups]
    eids_tens = torch.tensor(eids_list, dtype=torch.int32, device=device)
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()
    lora_A = (torch.randn(spec.E, RANK, spec.K, device=device,
                          dtype=torch.bfloat16) * 0.01).requires_grad_(True)
    lora_B = torch.zeros(spec.E, spec.N, RANK, device=device,
                         dtype=torch.bfloat16).requires_grad_(True)

    def zero():
        for t in (a_cat, lora_A, lora_B):
            t.grad = None

    def g_step(eids):
        def run():
            zero()
            out = gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes, eids,
                                          dgrad_kernel=True)
            d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, 1.0)
            (out + d.to(out.dtype)).float().pow(2).mean().backward()
        return run

    def d_step(eids):
        def run():
            zero()
            outs, row = [], 0
            for g, e in enumerate(eids):
                n = int(sizes[g])
                if n == 0:
                    continue
                outs.append(Fn.linear(a_cat[row:row + n],
                                      stack.dequant_bf16(int(e))))
                row += n
            out = torch.cat(outs)
            d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, 1.0)
            (out + d.to(out.dtype)).float().pow(2).mean().backward()
        return run

    return {"groups": len(groups), "rows": a_cat.shape[0],
            "g_list": g_step(eids_list), "g_tens": g_step(eids_tens),
            "d_list": d_step(eids_list), "d_tens": d_step(eids_tens)}


def cell(spec, regime, device, args, stack):
    f = build(spec, regime, device, stack)
    row = {"model": spec.model, "proj": spec.proj, "regime": regime,
           "E": spec.E, "groups": f["groups"], "rows": f["rows"]}
    try:
        _warm(f["g_list"], args.warm_s)
        iters = _pilot_iters(f["g_list"], args.block_ms)
        row["iters"] = iters
        t = {}
        for arm in ("g", "d"):
            lo, tn = f[f"{arm}_list"], f[f"{arm}_tens"]
            t[f"{arm}_l1"] = _timed(lo, iters)
            t[f"{arm}_l2"] = _timed(lo, iters)          # ADJACENT self-pair
            t[f"{arm}_t1"] = _timed(tn, iters)          # adjacent to l2
            t[f"{arm}_t2"] = _timed(tn, iters)          # ADJACENT self-pair
            t[f"{arm}_l3"] = _timed(lo, iters)          # closes the comparison
            row[f"{arm}_selfpair_list"] = t[f"{arm}_l2"] / t[f"{arm}_l1"]
            row[f"{arm}_selfpair_tens"] = t[f"{arm}_t2"] / t[f"{arm}_t1"]
            row[f"{arm}_drift"] = t[f"{arm}_l3"] / t[f"{arm}_l1"]
            row[f"{arm}_tensor_over_list"] = t[f"{arm}_t1"] / t[f"{arm}_l2"]
        row["ms"] = t
        row["status"] = "ok"
    except Exception as e:  # pragma: no cover
        row.update({"status": "skipped",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}"})
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=["decode_m8", "tokbudget_2048"])
    ap.add_argument("--block-ms", type=float, default=400.0)
    ap.add_argument("--warm-s", type=float, default=3.0)
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="e1")
    args = ap.parse_args()

    out = {"probe": "E1 expert_ids form, adjacent within-cell",
           "tier": "EXPLORATORY / report-only",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "regimes": args.regimes,
           "block_ms": args.block_ms, "warm_s": args.warm_s, "rows": []}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"eids_form_{args.tag}.json"
    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        stack = H.QuantStack(spec, "cuda")
        for regime in args.regimes:
            r = cell(spec, regime, "cuda", args, stack)
            out["rows"].append(r)
            print(json.dumps(r), flush=True)
            art.write_text(json.dumps(out, indent=1, default=str))
        del stack
        torch.cuda.empty_cache()

    def band(r, k):
        return 0.97 <= r[k] <= 1.03
    live = [r for r in out["rows"] if r.get("status") == "ok"
            and band(r, "g_selfpair_list") and band(r, "g_selfpair_tens")
            and 0.95 <= r["g_drift"] <= 1.05]
    print(f"LIVE {len(live)} of {len([r for r in out['rows'] if r.get('status')=='ok'])}")
    if live:
        for arm, label in (("g", "G_base+lora"), ("d", "D_base+lora")):
            v = [r[f"{arm}_tensor_over_list"] for r in live]
            print("%s  tensor/list: median %.3f  range %.3f-%.3f" % (
                label, statistics.median(v), min(v), max(v)))
    print("EIDS_PROBE_DONE")


if __name__ == "__main__":
    main()
