# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-1 OLMoE capture + audit (CHARTER §4). Runs in the QNAP gnf4-v6
container on the A2000. OLMoE-1B-7B-0924 is a BASE checkpoint (not fine-tuned)
— a legal Phase-1 capture (CHARTER §5 bright line 1 forbids only fine-tuned-
checkpoint H measurement).

Loads 4-bit (bnb NF4) — fits the 12 GB A2000 and is representative of how MoE
gets served (the e4b thesis); the router still makes well-defined choices.
Prefill excluded; only bs1 decode steps are captured (contract §3.2). One
capture serves all Δ; the ladder + reducer run here and print the per-family
verdict — the committed reducer, never this script, adjudicates.

Usage (inside gnf4-v6):
    python run_olmoe_capture.py --out /work/router_probe_olmoe --tokens 512 --prompts 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

RP = Path(__file__).resolve().parent.parent   # router_probe root (host-agnostic)
sys.path.insert(0, str(RP))
from capture.hooks import DecodeCapture         # noqa: E402
from capture.streams import ContractLoader, write_capture  # noqa: E402
from probes.ladder import train_eval_rung       # noqa: E402

PROMPTS = [
    "The history of the Roman aqueduct system begins with",
    "In quantum mechanics, the uncertainty principle states that",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "The recipe calls for two cups of flour, one egg, and",
    "Yesterday the central bank raised interest rates because",
    "Photosynthesis converts carbon dioxide and water into",
    "She opened the letter and read the first line aloud:",
    "The tectonic plates beneath the Pacific Ocean are",
    "According to the treaty signed in 1648, the boundaries",
    "The algorithm sorts the array in place by repeatedly",
    "Migratory birds navigate using a combination of",
    "The symphony's third movement modulates from C minor to",
]
LADDER = [
    {"name": "linear", "kind": "linear"},
    {"name": "mlp_d", "kind": "mlp", "width_mult": 1},
    {"name": "mlp_4d", "kind": "mlp", "width_mult": 4},
    {"name": "attn2", "kind": "attn", "heads": 4, "layers": 2, "width": 256},
    # procedure.yaml AMENDMENT A1 (2026-07-17): attention-family capacity climb —
    # Qwen3-30B read probe-limited with attn2 still gaining +0.20 over mlp_4d.
    {"name": "attn4", "kind": "attn", "heads": 4, "layers": 4, "width": 256},
    {"name": "attn4_w512", "kind": "attn", "heads": 8, "layers": 4, "width": 512},
    {"name": "attn6_w512", "kind": "attn", "heads": 8, "layers": 6, "width": 512},
]
TRAIN = {"epochs": 30, "batch": 512, "lr": 3.0e-3, "cosine": True, "weight_decay": 0.0, "seed": 20260716,
         "device": "cuda:0" if torch.cuda.is_available() else "cpu"}


FAMILIES = {
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
}


def capture(out_dir, n_tokens, n_prompts, family="olmoe", offload=False):
    from transformers import AutoTokenizer
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    name = FAMILIES[family]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # Experts4bit (bitsandbytes#1849/#1965): stock load_in_4bit's walker only swaps
    # nn.Linear modules, so OLMoE's FUSED expert stacks (OlmoeExperts.gate_up_proj/
    # down_proj, a 3-D nn.Parameter) stay bf16 -> ~9.55 GiB, which OOMs the 12 GB
    # A2000 even fully cleared. The streaming loader quantizes those fused experts to
    # NF4 on the way to the GPU (~4.7 GiB; the bf16 model is never materialized). The
    # router gate + block modules are untouched, so the DecodeCapture hooks attach
    # unchanged, and ExpertsLoRA is zero-init so the forward equals the frozen NF4
    # base — exactly the 4-bit model the probe is meant to characterize.
    # offload=True: frozen NF4 experts live in pinned CPU RAM and stream to the GPU
    # one layer at a time (prefetch overlaps the H2D copy) — Qwen3-30B decodes in
    # ~4.4 GB VRAM, so the capture fits a 12 GB card that can't hold the experts
    # resident. Router gate/hidden/embed stay on-GPU, so the contract streams and
    # hooks are unchanged; only wall-clock differs (~0.2 tok/s on the A2000).
    model, _ = load_moe_4bit_streaming(name, device="cuda:0", dtype=torch.bfloat16,
                                       r=8, alpha=16, offload=offload, prefetch=offload,
                                       quant_type="nf4")
    model.eval()
    cfg = model.config
    E, k = cfg.num_experts, cfg.num_experts_per_tok
    L = cfg.num_hidden_layers
    cap = DecodeCapture(model, family=family, layers=range(L))
    cap.set_k(k)
    cap.arm()
    tok_id = 0
    rec_tok, rec_layer = [], []
    for pi in range(min(n_prompts, len(PROMPTS))):
        ids = tok(PROMPTS[pi], return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        cap.begin_decode()
        for _ in range(n_tokens):
            before = len(cap.buf["topk"])
            with torch.no_grad():
                o = model(nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1:].argmax(-1)
            added = len(cap.buf["topk"]) - before
            rec_tok += [tok_id] * added
            rec_layer += list(range(added))       # layer-major within the token
            tok_id += 1
        cap._decode_mode = False
    cap.disarm()
    n = cap.save(out_dir, E=E, k=k,
                 extra_meta={"model": name, "family": family,
                             "load": "nf4-4bit" + ("+expert-offload" if offload else ""),
                             "n_layers": L, "prompts": min(n_prompts, len(PROMPTS)),
                             "tokens_per_prompt": n_tokens,
                             "input_note": "embedded diverse prompts, greedy decode"})
    # rewrite join sidecars aligned to the saved n
    write_capture(out_dir,
                  np.stack(cap.buf["hidden"][:n]), np.stack(cap.buf["logits"][:n]),
                  np.stack(cap.buf["embed"][:n]), np.stack(cap.buf["topk"][:n]),
                  json.loads((Path(out_dir) / "meta.json").read_text()),
                  record_token=rec_tok[:n], record_layer=rec_layer[:n])
    return E, k, L, n


def audit(out_dir, E, k, family):
    rows = []
    for delta in (1, 2, 4):
        ld = ContractLoader(out_dir, delta=delta)
        tX, ty, hX, hy = ld.split(heldout=max(2000, ld.y.shape[0] // 6), seed=TRAIN["seed"])
        rungs = []
        for rung in LADDER:
            r = train_eval_rung(rung, tX, ty, hX, hy, E, k, TRAIN)
            rungs.append({"name": rung["name"], **r})
            print(f"[{family} delta{delta}] {rung['name']:8s} train={r['train_h']:.4f} heldout={r['heldout_h']:.4f}", flush=True)
        rows.append({"family": family, "band": "all_layers", "delta": delta,
                     "masked_fraction": ld.masked_fraction, "rungs": rungs})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="olmoe", choices=list(FAMILIES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device-label", default="cloud A5000")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--offload", action="store_true",
                    help="e4b expert offload: experts stream from pinned CPU RAM "
                         "(fits Qwen3-30B capture in ~4.4GB VRAM; slow, quiet-window use)")
    # capture/audit split (A1 ops): cloud pods die ~50 min in (2026-07-17 incident),
    # so the pod runs --skip-audit (capture + stream tar only, in-window) and the
    # 7-rung ladder runs locally from the pulled streams via --audit-only.
    ap.add_argument("--skip-audit", action="store_true",
                    help="capture + write streams, no ladder/receipt (pod mode)")
    ap.add_argument("--audit-only", action="store_true",
                    help="run the ladder + reducer on an EXISTING stream dir (--out)")
    args = ap.parse_args()
    t0 = time.time()
    out = args.out or f"/root/router_probe_{args.family}"
    stream_dir = out + "/streams"
    if args.audit_only:
        meta = json.loads((Path(stream_dir) / "meta.json").read_text())
        E, k = int(meta["E"]), int(meta["k"])
        n = int(meta.get("n") or meta.get("records") or
                np.load(Path(stream_dir) / "topk_set.npy", mmap_mode="r").shape[0])
        L = int(meta.get("n_layers") or 0)
        print(f"audit-only: {n} records from {stream_dir} (E={E} k={k})", flush=True)
    else:
        E, k, L, n = capture(stream_dir, args.tokens, args.prompts, args.family, args.offload)
        print(f"captured {n} records (family={args.family} E={E} k={k} L={L})", flush=True)
        if args.skip_audit:
            print("skip-audit: streams written, exiting (pod mode)", flush=True)
            return
    rows = audit(stream_dir, E, k, args.family)
    date = time.strftime("%Y%m%d")
    rdir = RP / "receipts" / date
    rdir.mkdir(parents=True, exist_ok=True)
    receipt = {"tier": "EXPLORATORY", "charter": "router_probe/CHARTER.md",
               "phase": 1, "family": args.family, "device": args.device_label,
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "elapsed_s": round(time.time() - t0, 1), "records": n, "runs": rows}
    rp = rdir / f"EXPLORATORY_phase1_{args.family}.json"
    rp.write_text(json.dumps(receipt, indent=1))
    print(f"\nreceipt: {rp}\n--- committed reducer verdict (per delta) ---", flush=True)
    for row in rows:
        one = rdir / f".olmoe_d{row['delta']}.json"
        one.write_text(json.dumps(row))
        subprocess.run([sys.executable, str(RP / "reduce" / "reduce_ceiling.py"),
                        "ceiling", str(one)], check=True)
        one.unlink()


if __name__ == "__main__":
    main()
