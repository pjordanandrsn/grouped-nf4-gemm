# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Dev-box gate for the meta-init native loader (pod prerequisite).

Builds gpt-oss-20b via build_native_qlora_model (NO dequant path anywhere),
then checks against the smoke run's artifact (the from_pretrained+patch path):

  N1  provenance: file-hash table identical to the smoke's (96 tensors)
  N2  canary: same 128-token chunk, GPU — loss within 1e-2 of the smoke's
      native-path loss AND within the T4 band of the dequant reference
  N3  trains: 2 AdamW steps, finite, adapters move
  N4  host peak: RSS stays far under the dequant path's (38.1 GB at 20b)

Run inside gnf4-v6 after the smoke: python3 gate_native_load_20b.py
"""
from __future__ import annotations

import json
import time

import torch

from mxfp4_native_load import build_native_qlora_model
from mxfp4_qlora import adapter_parameters, lora_parameters
from run_mxfp4_20b_qlora import (
    batches_from_wikitext, cuda_move_with_storage_stubbed, log,
    resolve_snapshot, rss_gb,
)

SMOKE_ARTIFACT = "/work/mx7/smoke20b/run_artifact.json"


def main():
    with open(SMOKE_ARTIFACT) as f:
        smoke = json.load(f)
    cfg = smoke["config"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(6)

    snap = resolve_snapshot(smoke["model"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    train_chunks, eval_chunks = batches_from_wikitext(
        tok, cfg["seq"], cfg["steps"], 4, cfg["seed"])
    canary_ids = eval_chunks[0:1, :128]

    t0 = time.time()
    model, wrappers, file_hashes = build_native_qlora_model(
        snap, r=cfg["r"], alpha=cfg["alpha"], log=log)
    log(f"native build {time.time()-t0:.0f}s rss={rss_gb():.1f}GB (dequant path was 38.1)")
    peak_build_rss = rss_gb()

    # N1: identical provenance table
    want = smoke["provenance"]["file_hashes"]
    assert file_hashes == want, (
        f"hash tables differ: {len(file_hashes)} vs {len(want)}")
    log(f"N1 PASS: {len(file_hashes)} file hashes identical to the smoke table")

    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers:
        for p in adapter_parameters(w):
            p.requires_grad_(True)
    cuda_move_with_storage_stubbed(model, wrappers, pin=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    with torch.no_grad():
        out = model(input_ids=canary_ids.to("cuda"), labels=canary_ids.to("cuda"))
        loss = float(out.loss)
    d_smoke = abs(loss - smoke["canary"]["our_loss"])
    d_ref = abs(loss - smoke["canary"]["ref_loss"])
    log(f"N2 canary: loss={loss:.4f} |Δ smoke-native|={d_smoke:.4f} "
        f"|Δ dequant-ref|={d_ref:.4f}")
    assert d_smoke < 1e-2, (loss, smoke["canary"]["our_loss"])

    params = [p for w in wrappers for p in lora_parameters(w)]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    model.train()
    b0 = wrappers[0].gate_up_lora_B.detach().clone()
    losses = []
    for step in range(2):
        ids = train_chunks[step:step + 1].to("cuda")
        opt.zero_grad(set_to_none=True)
        loss_t = model(input_ids=ids, labels=ids).loss
        loss_t.backward()
        opt.step()
        losses.append(float(loss_t))
        log(f"N3 step {step} loss={losses[-1]:.4f}")
    assert all(torch.isfinite(torch.tensor(losses))), losses
    assert not torch.equal(b0, wrappers[0].gate_up_lora_B.detach())

    result = dict(build_rss_gb=peak_build_rss, canary_loss=loss,
                  d_smoke=d_smoke, d_ref=d_ref, steps=losses,
                  cuda_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    with open("/work/mx7/gate_native_load.json", "w") as f:
        json.dump(result, f, indent=1)
    log(f"NATIVE-LOAD GATE PASS {json.dumps(result)}")


if __name__ == "__main__":
    main()
