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
    {"name": "attn2", "kind": "attn", "heads": 4, "layers": 2},
]
TRAIN = {"epochs": 30, "batch": 512, "lr": 3.0e-3, "cosine": True, "weight_decay": 0.0, "seed": 20260716}


FAMILIES = {
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
}


def capture(out_dir, n_tokens, n_prompts, family="olmoe"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    name = FAMILIES[family]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb,
                                                 device_map={"": 0}, trust_remote_code=True)
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
                 extra_meta={"model": name, "family": family, "load": "nf4-4bit",
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
    args = ap.parse_args()
    t0 = time.time()
    out = args.out or f"/root/router_probe_{args.family}"
    stream_dir = out + "/streams"
    E, k, L, n = capture(stream_dir, args.tokens, args.prompts, args.family)
    print(f"captured {n} records (family={args.family} E={E} k={k} L={L})", flush=True)
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
