#!/usr/bin/env python3
"""C1 — BISECT the fused path's CUDA graph capture failure, one hazard at a time.

`probe_cudagraph_feasibility.py` established the asymmetry: the dequant-on-forward
baseline captures 4/4 on both devices, the fused training path fails 8/8. CUDA's
error -- "operation failed due to a previous error during capture" -- is
`cudaErrorStreamCaptureInvalidated`, which is what a *later* call reports after an
*earlier* illegal one already killed the capture. It never names the offender. So
the cause was recorded as NOT ESTABLISHED, with two candidates named pre-run.

This probe establishes it. Each hazard is removed on its own and capture is
re-attempted, so every verdict is attributable to one named change.

## The hazards, by name

  HA  `FusedGroupedNf4.forward` does `[int(e) for e in expert_ids]` (and the same
      over `sizes`). With a device tensor that is one D2H sync PER GROUP. Fires
      only in the `eids=devtensor` form. (Candidate 1, named pre-run.)
  HB  `gemm_4bit_grouped` builds `torch.tensor(expert_ids, device=dev)` when
      handed a list -- a pageable H2D, which is `cudaMemcpyAsync` followed by
      `cudaStreamSynchronize` and is not capturable. Fires only in the
      `eids=list` form. (Candidate 2, named pre-run.)
  HC  `build_group_tiles` builds THREE pageable H2D tensors (`t_row0`, `t_rows`,
      `t_group`) from Python lists. It is called by the M-tile forward path AND
      again by `dgrad_4bit_grouped` in the backward, and it takes `sizes`, not
      `expert_ids` -- so it fires in BOTH eids forms and neither named candidate
      touches it. NOT named pre-run.
  HD  `dgrad_4bit_grouped` repeats HB's conversion in the backward. `ctx` always
      hands it a Python list (that is what HA's list-comp produces), so this one
      fires in both eids forms too. NOT named pre-run.

HA and HB pull opposite ways -- HA wants a list, HB wants a tensor -- which is why
neither eids form captured. HC and HD are indifferent to the form.

## Why each attempt gets its own process

A failed capture poisons the CUDA context: every later attempt in the same
process reports the same "previous error during capture" whether or not it would
have failed on its own. That discipline is `probe_cudagraph_feasibility.py`'s and
it is reused here verbatim -- one attempt, one subprocess, one verdict.

## Positive controls, because a probe that only ever prints FAIL proves nothing

  * `D_base` runs through the same `_capture()` and the same process isolation.
    It must succeed. If it does not, this instrument is broken and no FAIL it
    reports means anything.
  * the hazard census (below) is itself positive-controlled: a deliberate
    `torch.tensor(..., device='cuda')` and a deliberate `int(cuda_tensor)` are
    executed and must both be seen. A census that sees zero hazards because it is
    blind looks exactly like a call path that has none.

## What the removals are

Memoised hoists: the offending tensor is built ONCE, keyed on the routing
metadata, during warm-up -- so inside the capture region the call is a dict hit
with no host round trip. This isolates the hazard without touching kernel math or
changing a single value the kernel reads. It is a probe device, not the shipped
fix: the shipped form (C2) hoists to the caller's boundary instead of caching
per routing pattern. What it establishes is which removals are SUFFICIENT.

Report-only. No prereg grades it; C3 registers what capturability does and does
not buy before anything is measured with it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as Fn

_ROOT = Path(os.environ.get("DQF_REPO", Path.cwd())).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(f"DQF_REPO/cwd={_ROOT} is not the repo root. Set DQF_REPO.")
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402

RANK = 16
PATCHES = ("ctx", "gemm", "tiles", "dgrad")


# ------------------------------------------------------------------ removals
def apply_patches(names):
    """Install the memoised hoists named in `names`. Returns what was installed."""
    import nf4_grouped as NG
    import nf4_qlora as NQ

    installed = []
    memo = {}

    def dev_eids(eids, dev):
        if torch.is_tensor(eids):
            return eids
        key = ("eids", tuple(int(e) for e in eids), str(dev))
        if key not in memo:
            memo[key] = torch.tensor(list(key[1]), dtype=torch.int32, device=dev)
        return memo[key]

    if "gemm" in names:
        _orig_gemm = NG.gemm_4bit_grouped

        def gemm_hoisted(a_cat, B, absmax, sizes, expert_ids, *a, **k):
            return _orig_gemm(a_cat, B, absmax, sizes,
                              dev_eids(expert_ids, a_cat.device), *a, **k)

        NG.gemm_4bit_grouped = gemm_hoisted
        installed.append("gemm:eids H2D hoisted")

    if "dgrad" in names:
        _orig_dgrad = NG.dgrad_4bit_grouped

        def dgrad_hoisted(grad_out, B, absmax, sizes, expert_ids, *a, **k):
            return _orig_dgrad(grad_out, B, absmax, sizes,
                               dev_eids(expert_ids, grad_out.device), *a, **k)

        NG.dgrad_4bit_grouped = dgrad_hoisted
        installed.append("dgrad:eids H2D hoisted")

    if "tiles" in names:
        _orig_tiles = NG.build_group_tiles

        def tiles_hoisted(sizes, block_m, device):
            key = ("tiles", tuple(int(s) for s in sizes), int(block_m), str(device))
            if key not in memo:
                memo[key] = _orig_tiles(list(key[1]), block_m, device)
            return memo[key]

        NG.build_group_tiles = tiles_hoisted
        installed.append("tiles:t_row0/t_rows/t_group H2D hoisted (3 per call)")

    if "ctx" in names:
        @staticmethod
        def fwd_no_percall_int(ctx, a_cat, packed, absmax, sizes, expert_ids,
                               weights_fn=None, dgrad_kernel=True):
            from nf4_grouped import gemm_4bit_grouped
            ctx.dgrad_kernel = dgrad_kernel
            out = gemm_4bit_grouped(a_cat, packed, absmax, sizes, expert_ids)
            ctx.weights_fn = weights_fn
            if weights_fn is None:
                ctx.packed, ctx.absmax = packed, absmax
            else:
                ctx.packed = ctx.absmax = None
                ctx.wshape = tuple(packed.shape)
            # THE REMOVAL: keep whatever was handed in. No per-element int(),
            # so a device tensor never round-trips to the host.
            ctx.sizes = sizes
            ctx.expert_ids = expert_ids
            return out

        NQ.FusedGroupedNf4.forward = fwd_no_percall_int
        installed.append("ctx:per-element int() over expert_ids/sizes removed")
    return installed


# ------------------------------------------------------------- hazard census
class Census:
    """Names the hazards that actually FIRE in this cell, with caller file:line.

    Capture pass/fail is the ground truth; this says which of the four named
    sites the step reached, so a hazard that is simply not on this cell's path
    is never blamed for the failure."""

    def __init__(self):
        self.h2d = []          # host->device transfers that are NOT capturable
        self.ok_h2d = []       # host->device transfers that are (pinned + async)
        self.syncs = []        # device->host syncs the debug mode reports
        self._orig = None
        self._w = None

    def __enter__(self):
        import traceback
        self._orig = torch.tensor
        _to, _copy = torch.Tensor.to, torch.Tensor.copy_

        def site():
            for fr in reversed(traceback.extract_stack()[:-2]):
                if Path(fr.filename).name != "probe_capture_bisect.py":
                    return f"{Path(fr.filename).name}:{fr.lineno}"
            return "?"

        def note(src_pinned, non_blocking, where):
            # A host->device copy is capture-legal only if the source is pinned
            # AND the copy is async. Anything else is cudaMemcpyAsync followed by
            # cudaStreamSynchronize, which invalidates a capture.
            (self.ok_h2d if (src_pinned and non_blocking) else self.h2d).append(
                where + ("" if src_pinned else "[pageable]")
                + ("" if non_blocking else "[blocking]"))

        def counting_tensor(data, *a, **k):
            dev = k.get("device")
            if dev is not None and "cuda" in str(dev) and not torch.is_tensor(data):
                note(False, False, site())
            return self._orig(data, *a, **k)

        def counting_to(self_t, *a, **k):
            tgt = k.get("device") or next(
                (x for x in a if isinstance(x, (str, torch.device))), None)
            if (self_t.device.type == "cpu" and tgt is not None
                    and "cuda" in str(tgt)):
                note(self_t.is_pinned(), bool(k.get("non_blocking", False)), site())
            return _to(self_t, *a, **k)

        def counting_copy(self_t, src, *a, **k):
            if (self_t.device.type == "cuda" and torch.is_tensor(src)
                    and src.device.type == "cpu"):
                nb = bool(k.get("non_blocking", a[0] if a else False))
                note(src.is_pinned(), nb, site())
            return _copy(self_t, src, *a, **k)

        torch.tensor = counting_tensor
        torch.Tensor.to = counting_to
        torch.Tensor.copy_ = counting_copy
        self._restore = (_to, _copy)
        self._w = warnings.catch_warnings(record=True)
        self._rec = self._w.__enter__()
        warnings.simplefilter("always")
        try:
            torch.cuda.set_sync_debug_mode("warn")
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        try:
            torch.cuda.set_sync_debug_mode("default")
        except Exception:
            pass
        for w in self._rec:
            m = str(w.message)
            if "synchron" in m.lower():
                self.syncs.append(m[:120])
        self._w.__exit__(*exc)
        torch.tensor = self._orig
        torch.Tensor.to, torch.Tensor.copy_ = self._restore
        return False


def census_selftest():
    """Positive control, one per hook. A census that cannot see a hazard it was
    handed is blind, and a blind census reporting zero looks exactly like a
    clean call path. The first version of this probe had only the `torch.tensor`
    hook and reported "0 H2D" for a call path doing a pinned `.to()` every call.

    Each construct here has a KNOWN verdict, measured on the A2000 in its own
    process (see scratch probe `legal.py`, reproduced in the results doc):
    pageable build and blocking copy are not capturable; a pre-pinned async
    transfer is."""
    c = Census()
    pin = torch.tensor([1, 2, 3], dtype=torch.int32).pin_memory()
    dst = torch.zeros(3, dtype=torch.int32, device="cuda")
    with c:
        torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")   # pageable
        torch.tensor([1, 2, 3], dtype=torch.int32).to("cuda")       # pageable .to
        dst.copy_(pin, non_blocking=False)                          # pinned but blocking
        pin.to("cuda", non_blocking=True)                           # legal
        int(torch.zeros(1, device="cuda")[0])                       # D2H sync
    return {"hazards_seen": len(c.h2d), "hazards_expected": 3,
            "legal_seen": len(c.ok_h2d), "legal_expected": 1,
            "sync_seen": len(c.syncs) >= 1,
            "blind": len(c.h2d) < 3 or len(c.ok_h2d) < 1 or not c.syncs}


# ---------------------------------------------------------------- the attempt
def _capture(step):
    """Warm on a side stream, capture, replay once. (ok, note). Same shape as
    probe_cudagraph_feasibility._capture -- H4 (JIT/autotune) and H5 (.grad
    buffers exist and are zeroed IN PLACE) are handled before capture there and
    here, so neither is what any FAIL below is measuring."""
    try:
        step()
        torch.cuda.synchronize()
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
        return True, "captured and replayed"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:300]}"


def build_step(arm, eids_form, spec, regime, device="cuda"):
    from nf4_qlora import gemm_4bit_grouped_train, lora_delta_grouped

    stack = H.QuantStack(spec, device)
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    eids_list = [int(e) for e, _ in groups]
    eids_dev = torch.tensor(eids_list, dtype=torch.int32, device=device)
    eids = eids_dev if eids_form == "devtensor" else eids_list
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()
    lora_A = (torch.randn(spec.E, RANK, spec.K, device=device,
                          dtype=torch.bfloat16) * 0.01).requires_grad_(True)
    lora_B = torch.zeros(spec.E, spec.N, RANK, device=device,
                         dtype=torch.bfloat16).requires_grad_(True)

    def zero_inplace():
        for t in (a_cat, lora_A, lora_B):
            if t.grad is not None:
                t.grad.zero_()

    def g_base():
        zero_inplace()
        out = gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes, eids,
                                      dgrad_kernel=True)
        out.float().pow(2).mean().backward()

    def g_full():
        zero_inplace()
        out = gemm_4bit_grouped_train(a_cat, B_pack, A_scale, sizes, eids,
                                      dgrad_kernel=True)
        d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids_list, 1.0)
        (out + d.to(out.dtype)).float().pow(2).mean().backward()

    def d_base():
        zero_inplace()
        outs, row = [], 0
        for gi, e in enumerate(eids_list):
            n = int(sizes[gi])
            if n == 0:
                continue
            outs.append(Fn.linear(a_cat[row:row + n], stack.dequant_bf16(e)))
            row += n
        torch.cat(outs).float().pow(2).mean().backward()

    step = {"G_base": g_base, "G_full": g_full, "D_base": d_base}[arm]
    return step, {"groups": len(groups), "rows": a_cat.shape[0],
                  "sizes_max": max(sizes)}


def run_one(model, regime, proj, arm, eids_form, patch_csv):
    names = [] if patch_csv in ("", "none") else patch_csv.split(",")
    for n in names:
        if n not in PATCHES:
            raise SystemExit(f"unknown patch {n!r}; known: {PATCHES}")
    installed = apply_patches(names) if names else []

    specs = [s for s in H.census_specs(H.REPO / "census" / "shape_census.json",
                                       [model]) if s.proj == proj]
    if not specs:
        raise SystemExit(f"no census spec for {model}/{proj}")
    spec = specs[0]

    selftest = census_selftest()
    step, shape = build_step(arm, eids_form, spec, regime)

    # Census TWICE, on un-captured calls, before the capture attempt (which may
    # die partway and leave a census half-populated).
    #
    # COLD vs WARM matters and conflating them would misattribute the failure.
    # Some sites fire once and cache -- `_lut()` builds the NF4 codebook on
    # device the first time it sees a device, and a memoised hoist populates its
    # key on the first call. Those are pre-capture-warmable and are NOT capture
    # hazards, because the 5 side-stream warm-up iterations run before capture.
    # The WARM census is the one that predicts capture; the COLD one is kept so
    # the difference is visible rather than assumed.
    cold, warm = Census(), Census()
    with cold:
        step()
    torch.cuda.synchronize()
    with warm:
        step()
    torch.cuda.synchronize()

    ok, note = _capture(step)
    print("RESULT " + json.dumps({
        "arm": arm, "eids_form": eids_form, "patches": names,
        "installed": installed, "captured": ok, "note": note,
        "shape": shape,
        "census_warm": {"h2d_sites": sorted(set(warm.h2d)),
                        "h2d_calls": len(warm.h2d),
                        "legal_h2d_sites": sorted(set(warm.ok_h2d)),
                        "legal_h2d_calls": len(warm.ok_h2d),
                        "syncs": len(warm.syncs), "sync_msgs": warm.syncs[:3]},
        "census_cold": {"h2d_sites": sorted(set(cold.h2d)),
                        "h2d_calls": len(cold.h2d),
                        "legal_h2d_calls": len(cold.ok_h2d),
                        "syncs": len(cold.syncs)},
        "census_selftest": selftest}))


# ------------------------------------------------------------------- the ladder
# arm, eids form, patches, what this attempt is for.
LADDER = [
    ("D_base", "list", "none", "POSITIVE CONTROL — the baseline must capture"),
    ("G_base", "list", "none", "reproduce the published FAIL (list form)"),
    ("G_base", "devtensor", "none", "reproduce the published FAIL (tensor form)"),
    ("G_base", "devtensor", "ctx", "candidate 1 alone (HA)"),
    ("G_base", "list", "gemm", "candidate 2 alone (HB)"),
    ("G_base", "list", "tiles", "HC alone"),
    ("G_base", "list", "dgrad", "HD alone"),
    ("G_base", "devtensor", "ctx,gemm", "BOTH NAMED CANDIDATES REMOVED"),
    ("G_base", "devtensor", "ctx,gemm,tiles", "+ HC"),
    ("G_base", "devtensor", "ctx,gemm,tiles,dgrad", "+ HD (all four)"),
    ("G_base", "list", "ctx,gemm,tiles,dgrad", "all four, list form"),
    ("G_full", "devtensor", "ctx,gemm,tiles,dgrad", "all four, + LoRA delta"),
]

# C2 verification: the SHIPPED call path, no probe patches at all. Every row
# here must capture, and `D_base` must still capture, or the fix is not a fix.
SHIPPED = [
    ("D_base", "list", "none", "POSITIVE CONTROL — the baseline still captures"),
    ("G_base", "list", "none", "shipped fused path, eids as a list"),
    ("G_base", "devtensor", "none", "shipped fused path, eids as a device tensor"),
    ("G_full", "list", "none", "shipped + LoRA delta, eids as a list"),
    ("G_full", "devtensor", "none", "shipped + LoRA delta, eids as a device tensor"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=6, default=None,
                    help="internal: <model> <regime> <proj> <arm> <eids> <patches>")
    ap.add_argument("--model", default="OLMoE")
    ap.add_argument("--regime", default="decode_m8")
    ap.add_argument("--proj", default="gate_up")
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/work/cap-out"))
    ap.add_argument("--tag", default="bis1")
    ap.add_argument("--mode", choices=["bisect", "shipped"], default="bisect",
                    help="bisect = C1's hazard ladder; shipped = C2's "
                         "verification of the unpatched call path")
    args = ap.parse_args()
    if args.child:
        run_one(*args.child)
        return

    ladder = LADDER if args.mode == "bisect" else SHIPPED
    out = {"probe": ("C1 — bisect the fused path's CUDA graph capture failure"
                     if args.mode == "bisect" else
                     "C2 — the SHIPPED call path, no probe patches"),
           "mode": args.mode,
           "tier": "EXPLORATORY / report-only, no prereg grades it",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__,
           "triton": __import__("triton").__version__,
           "model": args.model, "regime": args.regime, "proj": args.proj,
           "attempts": []}
    try:
        import bitsandbytes as bnb
        out["bitsandbytes"] = bnb.__version__
    except Exception:
        pass
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"capture_bisect_{args.tag}.json"

    print(f"{out['gpu']}  torch {out['torch']}  triton {out['triton']}  "
          f"{args.model} {args.proj} {args.regime}\n")
    for arm, form, patches, why in ladder:
        p = subprocess.run(
            [sys.executable, __file__, "--child", args.model, args.regime,
             args.proj, arm, form, patches],
            capture_output=True, text=True, timeout=1800,
            env={**os.environ, "DQF_REPO": str(_ROOT)})
        res = None
        for line in p.stdout.splitlines():
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
        if res is None:
            tail = (p.stderr or p.stdout).strip().splitlines()
            res = {"arm": arm, "eids_form": form, "patches": patches.split(","),
                   "captured": False,
                   "note": "child died: " + (tail[-1][:250] if tail else
                                             f"rc={p.returncode}")}
        res["why"] = why
        out["attempts"].append(res)
        art.write_text(json.dumps(out, indent=1, default=str))
        cen = res.get("census_warm", {})
        print("%-7s %-10s %-22s %s  %s" % (
            res["arm"], res["eids_form"], patches,
            "CAPTURED" if res["captured"] else "FAIL    ", why))
        print("        warm hazards: %-44s syncs: %s   (legal pinned H2D: %s)" % (
            ",".join(cen.get("h2d_sites") or []) or "-", cen.get("syncs", "?"),
            cen.get("legal_h2d_calls", "?")))
        if not res["captured"]:
            print("        %s" % res["note"][:150])
        st = res.get("census_selftest")
        if st and st.get("blind"):
            print("        !! CENSUS BLIND (%s/%s hazards, %s/%s legal, sync=%s) "
                  "— treat the census columns as unreported, not as zero"
                  % (st["hazards_seen"], st["hazards_expected"],
                     st["legal_seen"], st["legal_expected"], st["sync_seen"]))
        print(flush=True)

    ctl = out["attempts"][0]
    if not ctl["captured"]:
        print("!! POSITIVE CONTROL FAILED — D_base did not capture. Every FAIL "
              "above is uninterpretable; fix the instrument first.")
    if args.mode == "shipped":
        bad = [a for a in out["attempts"] if not a["captured"]]
        print("SHIPPED VERDICT: %d/%d captured%s" % (
            len(out["attempts"]) - len(bad), len(out["attempts"]),
            "" if not bad else "  — STILL FAILING: " + ", ".join(
                f"{a['arm']}({a['eids_form']})" for a in bad)))
    print("CAPTURE_BISECT_DONE  ->", art)


if __name__ == "__main__":
    main()
