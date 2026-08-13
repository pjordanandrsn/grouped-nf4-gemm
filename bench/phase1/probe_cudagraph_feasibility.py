#!/usr/bin/env python3
"""FEASIBILITY probe: can the training step be captured in a CUDA graph?

Leg 3 measured that at decode-band sizes on an H100 this step is CPU-BOUND --
the GPU finishes in ~0.44 ms inside a ~1.68 ms step. A small-batch ratio there
is substantially a statement about Python, and no pairing scheme fixes it. A
CUDA graph replays the whole launch sequence from one call, which removes the
per-launch CPU cost and lets the kernels be compared on their own terms.

Whether that is POSSIBLE for these arms is an empirical question with several
known ways to fail, so this probe answers it before any card is rented. Each
hazard below is checked separately and reported by name rather than as one
opaque "capture failed":

  H1  host-side sync inside the region. `lora_delta_grouped` iterates
      `expert_ids`; if that is a CUDA tensor, `enumerate` forces one device
      sync per group and capture dies. A python list is host-only.
  H2  `FusedGroupedNf4.forward` does `[int(e) for e in expert_ids]` -- same
      hazard from the other direction.
  H3  `gemm_4bit_grouped` builds `torch.tensor(expert_ids, device=dev)` when
      handed a list. That is a pageable host-to-device copy, which is not
      capturable. So H1/H2 want a list and H3 wants a pre-made device tensor:
      the two pull opposite ways and the probe reports which combination, if
      any, survives.
  H4  Triton JIT and autotune must finish BEFORE capture; a compile inside the
      capture region is a host-side excursion.
  H5  backward needs `.grad` buffers to already exist and to be zeroed
      IN PLACE inside the graph -- `grad = None` allocates on replay.
  H6  bitsandbytes `dequantize_4bit` must itself be capturable.

PROCESS ISOLATION IS MANDATORY HERE, and the first version of this probe got it
wrong. A failed capture leaves the CUDA context poisoned -- every later attempt
in the same process reports "operation failed due to a previous error during
capture" whether or not it would have failed on its own, and the damage leaked
into the next cell as "Offset increment outside graph capture". So each attempt
now runs in its OWN subprocess and only its own verdict is reported.

Report-only. No prereg grades any of this, no arm changes, and the numbers it
prints are feasibility and shape checks, not a measurement of anything.
"""
from __future__ import annotations

import argparse
import json
import os
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

RANK = 16


def _capture(build_step, tag):
    """Warm on a side stream, then capture. Returns (graph|None, note)."""
    step = build_step()
    try:
        # H5: materialise .grad buffers before capture, and never set them to
        # None afterwards -- the graph must zero them in place.
        step()
        torch.cuda.synchronize()

        # H4: warm on a SIDE stream. This is where Triton JIT/autotune, bnb's
        # first-call setup and any allocator growth must happen.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        return g, "captured and replayed"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"


def cell(spec, regime, device, args, stack):
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    eids_list = [int(e) for e, _ in groups]
    eids_dev = torch.tensor(eids_list, dtype=torch.int32, device=device)
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()
    lora_A = (torch.randn(spec.E, RANK, spec.K, device=device,
                          dtype=torch.bfloat16) * 0.01).requires_grad_(True)
    lora_B = torch.zeros(spec.E, spec.N, RANK, device=device,
                         dtype=torch.bfloat16).requires_grad_(True)

    def zero_inplace():
        # H5: in place, never None
        for t in (a_cat, lora_A, lora_B):
            if t.grad is not None:
                t.grad.zero_()

    def g_base(eids):
        def run():
            zero_inplace()
            out = gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes, eids,
                                          dgrad_kernel=True)
            out.float().pow(2).mean().backward()
        return run

    def d_base():
        def run():
            zero_inplace()
            outs, row = [], 0
            for g, e in enumerate(eids_list):
                n = int(sizes[g])
                if n == 0:
                    continue
                outs.append(Fn.linear(a_cat[row:row + n], stack.dequant_bf16(e)))
                row += n
            torch.cat(outs).float().pow(2).mean().backward()
        return run

    def g_full(eids):
        def run():
            zero_inplace()
            out = gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes, eids,
                                          dgrad_kernel=True)
            d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids_list, 1.0)
            (out + d.to(out.dtype)).float().pow(2).mean().backward()
        return run

    row = {"model": spec.model, "proj": spec.proj, "regime": regime,
           "groups": len(groups), "rows": a_cat.shape[0], "attempts": {}}

    # H1/H2/H3 pull opposite ways, so try both eids forms for the fused arm.
    for name, builder in (
        ("G_base(eids=list)", lambda: g_base(eids_list)),
        ("G_base(eids=devtensor)", lambda: g_base(eids_dev)),
        ("D_base(bnb dequant + F.linear)", d_base),
        ("G_full(eids=list, lora)", lambda: g_full(eids_list)),
    ):
        g, note = _capture(builder, name)
        row["attempts"][name] = {"captured": g is not None, "note": note}
        print(f"    {name:34s} {'OK  ' if g else 'FAIL'} {note}", flush=True)
        del g
        torch.cuda.empty_cache()

    row["status"] = "ok"
    return row


ATTEMPTS = ["G_base(eids=list)", "G_base(eids=devtensor)",
            "D_base(bnb dequant + F.linear)", "G_full(eids=list, lora)"]


def run_one(spec_filter, regime, attempt):
    """Child process: build ONE arm, try ONE capture, print one JSON line."""
    specs = H.census_specs(H.REPO / "census" / "shape_census.json", [spec_filter])
    spec = specs[0]
    stack = H.QuantStack(spec, "cuda")
    groups = H.make_activations(spec, regime, "cuda")
    sizes = [a.shape[0] for _, a in groups]
    eids_list = [int(e) for e, _ in groups]
    eids_dev = torch.tensor(eids_list, dtype=torch.int32, device="cuda")
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()
    lora_A = (torch.randn(spec.E, RANK, spec.K, device="cuda",
                          dtype=torch.bfloat16) * 0.01).requires_grad_(True)
    lora_B = torch.zeros(spec.E, spec.N, RANK, device="cuda",
                         dtype=torch.bfloat16).requires_grad_(True)

    def zero_inplace():
        for t in (a_cat, lora_A, lora_B):
            if t.grad is not None:
                t.grad.zero_()

    def g_base(eids):
        def run():
            zero_inplace()
            out = gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes, eids,
                                          dgrad_kernel=True)
            out.float().pow(2).mean().backward()
        return run

    def d_base():
        def run():
            zero_inplace()
            outs, row = [], 0
            for gi, e in enumerate(eids_list):
                n = int(sizes[gi])
                if n == 0:
                    continue
                outs.append(Fn.linear(a_cat[row:row + n], stack.dequant_bf16(e)))
                row += n
            torch.cat(outs).float().pow(2).mean().backward()
        return run

    def g_full():
        def run():
            zero_inplace()
            out = gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes,
                                          eids_list, dgrad_kernel=True)
            d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids_list, 1.0)
            (out + d.to(out.dtype)).float().pow(2).mean().backward()
        return run

    builder = {"G_base(eids=list)": lambda: g_base(eids_list),
               "G_base(eids=devtensor)": lambda: g_base(eids_dev),
               "D_base(bnb dequant + F.linear)": d_base,
               "G_full(eids=list, lora)": g_full}[attempt]
    g, note = _capture(builder, attempt)
    print("RESULT " + json.dumps(
        {"attempt": attempt, "captured": g is not None, "note": note}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=3, default=None,
                    help="internal: <model> <regime> <attempt>")
    ap.add_argument("--models", nargs="*", default=["OLMoE"])
    ap.add_argument("--regimes", nargs="*", default=["decode_m8"])
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="cg1")
    args = ap.parse_args()
    if args.child:
        run_one(*args.child)
        return

    out = {"probe": "CUDA graph capture feasibility for the training step",
           "tier": "EXPLORATORY / feasibility only",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "rows": []}
    try:
        import bitsandbytes as bnb
        out["bitsandbytes"] = bnb.__version__
    except Exception:
        pass
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"cudagraph_feasibility_{args.tag}.json"

    import subprocess
    for model in args.models:
        for regime in args.regimes:
            row = {"model": model, "regime": regime, "attempts": {}}
            print(f"  {model} {regime}", flush=True)
            for attempt in ATTEMPTS:
                # OWN PROCESS: a failed capture poisons the context for
                # everything after it, so a shared process cannot tell a real
                # failure from collateral damage.
                p = subprocess.run(
                    [sys.executable, __file__, "--child", model, regime, attempt],
                    capture_output=True, text=True, timeout=900,
                    env={**os.environ, "DQF_REPO": str(_ROOT)})
                res = None
                for line in p.stdout.splitlines():
                    if line.startswith("RESULT "):
                        res = json.loads(line[7:])
                if res is None:
                    tail = (p.stderr or p.stdout).strip().splitlines()
                    res = {"attempt": attempt, "captured": False,
                           "note": "child died: " + (tail[-1][:200] if tail else
                                                     f"rc={p.returncode}")}
                row["attempts"][attempt] = res
                print("    %-34s %s %s" % (attempt, "OK  " if res["captured"]
                                           else "FAIL", res["note"][:150]),
                      flush=True)
            row["status"] = "ok"
            out["rows"].append(row)
            art.write_text(json.dumps(out, indent=1, default=str))

    print()
    tally = {}
    for r in out["rows"]:
        for k, v in (r.get("attempts") or {}).items():
            tally.setdefault(k, [0, 0])
            tally[k][0 if v["captured"] else 1] += 1
    for k, (ok, bad) in tally.items():
        print("%-34s captured %d, failed %d" % (k, ok, bad))
    print("CUDAGRAPH_FEASIBILITY_DONE")


if __name__ == "__main__":
    main()
