# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The provenance receipt, made executable — the launch-kit headline artifact.

Claim: *"Fine-tuned; the expert bytes are bit-identical to OpenAI's release —
here are the hashes."* This script lets anyone re-derive that table from two
inputs and check it against our banked receipt, with no trust in us:

  1. OpenAI's shipped gpt-oss safetensors (downloaded from the Hub), and
  2. our run artifact (`run_artifact.json`) carrying the per-tensor SHA-256
     table recorded at training time.

For every expert tensor it recomputes sha256 of the raw bytes IN THE FILE'S
DATA SECTION (streamed from the header's byte range — no torch load, no
dequant: the actual on-disk bytes of the release) and asserts it equals the
hash our run recorded. Equality across all N tensors IS the receipt: the
bytes we trained over — and, since `pre_equals_post` holds, the bytes AFTER
training — are the release bytes.

  python verify_provenance.py --artifact run_artifact.json \
      --model openai/gpt-oss-120b            # resolves the local HF snapshot
  python verify_provenance.py --artifact ... --snapshot /path/to/snapshot

Exit 0 = every hash matches; nonzero = mismatch (a provenance failure is
never a tolerance). Uses only the stdlib + the byte-range reader from
mxfp4_loader; no GPU, no model materialization.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from mxfp4_loader import file_tensor_sha256


def resolve_snapshot(model_id: str) -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    base = os.path.join(HF_HUB_CACHE,
                        f"models--{model_id.replace('/', '--')}", "snapshots")
    cands = sorted(glob.glob(os.path.join(base, "*")))
    if not cands:
        raise FileNotFoundError(
            f"no local snapshot for {model_id} under {base} — download it first "
            "(huggingface-cli download / snapshot_download)")
    return cands[-1]


def shard_of(name: str, weight_map: dict, snapshot: str) -> str:
    if name not in weight_map:
        raise KeyError(f"{name} not in the checkpoint index")
    return os.path.join(snapshot, weight_map[name])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True,
                    help="run_artifact.json with provenance.file_hashes")
    ap.add_argument("--model", default=None, help="HF id, e.g. openai/gpt-oss-120b")
    ap.add_argument("--snapshot", default=None, help="explicit snapshot dir")
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the first N tensors (smoke)")
    args = ap.parse_args()

    art = json.load(open(args.artifact))
    prov = art["provenance"]
    table = prov["file_hashes"]
    model_id = args.model or art.get("model")
    snapshot = args.snapshot or resolve_snapshot(model_id)
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    weight_map = index["weight_map"]

    names = sorted(table)
    if args.limit:
        names = names[: args.limit]
    print(f"model      : {model_id}")
    print(f"snapshot   : {snapshot}")
    print(f"artifact   : {args.artifact}")
    print(f"recorded   : pre_equals_post={prov.get('pre_equals_post')} "
          f"n_file_tensors={prov.get('n_file_tensors')}")
    print(f"checking   : {len(names)} expert tensors "
          f"(sha256 of file data-section bytes)\n")

    bad = []
    for i, name in enumerate(names, 1):
        want = table[name]
        got = file_tensor_sha256(shard_of(name, weight_map, snapshot), name)
        ok = got == want
        if not ok:
            bad.append(name)
        if i <= 3 or not ok or i == len(names):
            print(f"  [{i:3d}/{len(names)}] {'OK  ' if ok else 'FAIL'} "
                  f"{name}  {got[:16]}")

    print()
    if bad:
        print(f"PROVENANCE FAIL: {len(bad)}/{len(names)} tensors differ from "
              f"the recorded release hashes:")
        for n in bad[:8]:
            print(f"  - {n}")
        return 1
    print(f"PROVENANCE OK: all {len(names)} expert tensors bit-identical to "
          f"{model_id}'s released bytes.")
    if prov.get("pre_equals_post"):
        print("               (run recorded pre==post: unchanged by training.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
