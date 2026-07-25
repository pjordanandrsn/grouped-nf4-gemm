#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""T1 of PREREG-training-context.md — recompute vs offload, on a busy link.

Decode's per-token VRAM term is the KV cache. Training's is the activation stack
held for backward, and there are two ways to buy it back: spend GPU compute
(gradient checkpointing) or spend PCIe bytes (`save_on_cpu`). The scout measured
that when weights stream the link saturates and the GPU idles — so recompute
spends the idle resource and offload spends the contended one. That is T1c, and
it is a MECHANISM story, the class this document set keeps falsifying.

3 activation policies x 2 weight residences, so the crossing is visible rather
than inferred from two separate runs.
"""
from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/root/work")
from experts4bit_qlora import load_moe_4bit_streaming  # noqa: E402

OUT = os.environ.get("TC_OUT", "/root/work/train_context.json")
MODEL = os.environ.get("TC_MODEL", "allenai/OLMoE-1B-7B-0924")
SEQS = [int(x) for x in os.environ.get("TC_SEQ", "8192,32768").split(",")]
REPS = int(os.environ.get("TC_REPS", "3"))


def link_rate_gbs(mb=256):
    """Measured pinned H2D rate — T1d needs a link, not a datasheet number."""
    src = torch.empty(mb * 2 ** 20 // 2, dtype=torch.bfloat16).pin_memory()
    dst = torch.empty_like(src, device="cuda")
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return (src.numel() * 2 * 10) / (time.perf_counter() - t0) / 1e9


def step(model, ids, policy):
    """One fwd+bwd under the named activation policy."""
    ctx = (torch.autograd.graph.save_on_cpu(pin_memory=True)
           if policy == "offload" else contextlib.nullcontext())
    with ctx:
        out = model(ids, labels=ids)
        out.loss.backward()
    loss = float(out.loss.detach())
    del out
    model.zero_grad(set_to_none=True)
    return loss


def main():
    rate = link_rate_gbs()
    print(f"{MODEL}  seqs={SEQS}  measured pinned H2D {rate:.2f} GB/s", flush=True)
    rows = []
    for streamed in (False, True):
        model, _ = load_moe_4bit_streaming(
            MODEL, device="cuda:0", dtype=torch.bfloat16, r=8, alpha=16,
            offload=streamed, pin=streamed, quant_type="nf4")
        model.train()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_train > 0, "no trainable parameters — LoRA did not attach"
        print(f"\nweights {'STREAMED' if streamed else 'resident'}  "
              f"trainable={n_train / 1e6:.1f}M", flush=True)
        for seq in SEQS:
            ids = torch.randint(100, 20000, (1, seq), device="cuda")
            for policy in ("none", "recompute", "offload"):
                if policy == "recompute":
                    model.gradient_checkpointing_enable()
                else:
                    model.gradient_checkpointing_disable()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ts, loss, err = [], None, None
                try:
                    for i in range(REPS + 1):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        loss = step(model, ids, policy)
                        torch.cuda.synchronize()
                        if i:  # first is warmup
                            ts.append(time.perf_counter() - t0)
                except torch.cuda.OutOfMemoryError:
                    err = "OOM"
                    torch.cuda.empty_cache()
                peak = torch.cuda.max_memory_allocated()
                rows.append(dict(seq=seq, policy=policy, weights_streamed=streamed,
                                 s_per_step=statistics.median(ts) if ts else None,
                                 peak_bytes=peak if not err else None,
                                 loss=loss, error=err))
                if err:
                    print(f"  seq={seq:>6} {policy:<10} {err}", flush=True)
                else:
                    print(f"  seq={seq:>6} {policy:<10} "
                          f"{statistics.median(ts):7.3f} s/step  "
                          f"peak={peak / 2 ** 30:6.2f} GiB  loss={loss:.4f}",
                          flush=True)
                json.dump(rows, open(OUT, "w"), indent=2)
        del model
        torch.cuda.empty_cache()

    def g(seq, policy, streamed):
        r = next((x for x in rows if x["seq"] == seq and x["policy"] == policy
                  and x["weights_streamed"] == streamed), None)
        return r if r and r["s_per_step"] else None

    s0, s1 = SEQS[0], SEQS[-1]
    v = {}
    print("\n=== T1 scoring ===")

    a = g(s0, "recompute", False), g(s0, "none", False)
    if all(a):
        t1a = a[0]["s_per_step"] / a[1]["s_per_step"]
        ok = 1.15 <= t1a <= 1.50
        bad = not (1.05 <= t1a <= 1.80)
        v["T1a"] = dict(measured=t1a, interval=[1.15, 1.50],
                        verdict="CONFIRMED" if ok else ("FALSIFIED" if bad else "outside interval"))
        print(f"T1a recompute/none, weights resident @{s0} = {t1a:.3f}  [1.15,1.50]  "
              f"{v['T1a']['verdict']}")
        if bad:
            print("     -> harness is presumed wrong before the textbook is; "
                  "nothing below is scored.")

    b = g(s1, "offload", False), g(s1, "recompute", False)
    if all(b):
        t1b = b[0]["peak_bytes"] / b[1]["peak_bytes"]
        v["T1b"] = dict(measured=t1b, verdict="CONFIRMED" if t1b <= 1.25 else
                        ("FALSIFIED" if t1b > 1.6 else "outside interval"))
        print(f"T1b offload peak / recompute peak @{s1} = {t1b:.3f}  <=1.25  "
              f"{v['T1b']['verdict']}")

    res = g(s1, "offload", False), g(s1, "recompute", False)
    stm = g(s1, "offload", True), g(s1, "recompute", True)
    if all(res) and all(stm):
        r_res = res[0]["s_per_step"] / res[1]["s_per_step"]
        r_stm = stm[0]["s_per_step"] / stm[1]["s_per_step"]
        gap = r_stm - r_res
        v["T1c"] = dict(ratio_resident=r_res, ratio_streamed=r_stm, gap=gap,
                        verdict="CONFIRMED" if gap >= 0.15 else "FALSIFIED")
        print(f"T1c offload/recompute: resident {r_res:.3f} -> streamed {r_stm:.3f}  "
              f"gap {gap:+.3f}  >=0.15  {v['T1c']['verdict']}")
        print(f"     (MECHANISM prediction — recorded as such either way)")

        none_s = g(s1, "none", False)
        off_s = g(s1, "offload", False)
        if none_s and off_s and none_s["peak_bytes"]:
            act = none_s["peak_bytes"] - off_s["peak_bytes"]
            predicted = 2 * act / (rate * 1e9)          # down on fwd, up on bwd
            actual = stm[0]["s_per_step"] - stm[1]["s_per_step"]
            ratio = actual / predicted if predicted > 0 else float("nan")
            v["T1d"] = dict(activation_bytes=act, predicted_s=predicted,
                            actual_s=actual, ratio=ratio,
                            verdict="CONFIRMED" if 0.65 <= ratio <= 1.35 else
                            ("FALSIFIED" if not (0.40 <= ratio <= 1.60) else "outside interval"))
            print(f"T1d activations {act / 2 ** 30:.2f} GiB -> predicted "
                  f"{predicted:.3f} s, actual {actual:+.3f} s, ratio {ratio:.3f}  "
                  f"[0.65,1.35]  {v['T1d']['verdict']}")

    json.dump(dict(rows=rows, link_gbs=rate, verdicts=v), open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
