#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license.
"""Frozen-expert-bytes provenance table — the reference ("before") column.

Hashes the EXACT serialized bytes of every MXFP4 expert tensor in a gpt-oss
checkpoint, straight from the safetensors shards: the 8-byte header-length +
JSON header give each tensor's data_offsets; we seek/read/sha256 that byte
range. No dtype interpretation, no framework load — what is hashed is what
OpenAI shipped.

This is the reference column of the AGENT-PLAN-mxfp4 B1 provenance artifact
("Fine-tuned; expert bytes bit-identical to the release — here are the
hashes"). The post-training half (Phase 2.5, training over native MXFP4)
re-hashes the same keys after N steps; equality against THIS table is the
claim in artifact form. Until that run exists this table is the shipped-bytes
reference, tier: reference/pre-training (R3).

stdlib only. Usage:
    python3 hash_expert_bytes.py <snapshot_dir> --json out.json --md out.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

EXPERT_KEY_RE = re.compile(
    r"\.mlp\.experts\.(gate_up_proj_blocks|gate_up_proj_scales|"
    r"gate_up_proj_bias|down_proj_blocks|down_proj_scales|down_proj_bias)$"
)
CHUNK = 8 * 1024 * 1024


def sha256_range(fh, start, end):
    fh.seek(start)
    h = hashlib.sha256()
    left = end - start
    while left > 0:
        b = fh.read(min(CHUNK, left))
        if not b:
            raise IOError("short read")
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def shard_header(path):
    """-> (header dict, data_base_offset)."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="HF snapshot dir (contains *.safetensors + index)")
    ap.add_argument("--json", dest="json_out", required=True)
    ap.add_argument("--md", dest="md_out", required=True)
    args = ap.parse_args()
    snap = Path(args.snapshot)

    idx_path = snap / "model.safetensors.index.json"
    weight_map = json.loads(idx_path.read_text())["weight_map"]
    expert_keys = sorted(k for k in weight_map if EXPERT_KEY_RE.search(k))
    if not expert_keys:
        raise SystemExit("no MXFP4 expert keys found — wrong snapshot?")

    by_shard = {}
    for k in expert_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    tensors, shards = [], {}
    for shard_name in sorted(by_shard):
        p = snap / shard_name
        hdr, base = shard_header(p)
        with open(p, "rb") as fh:
            for k in by_shard[shard_name]:
                meta = hdr[k]
                s, e = meta["data_offsets"]
                tensors.append({
                    "key": k,
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "nbytes": e - s,
                    "shard": shard_name,
                    "sha256": sha256_range(fh, base + s, base + e),
                })
        shards[shard_name] = {"sha256": sha256_file(p), "nbytes": p.stat().st_size}

    tensors.sort(key=lambda t: t["key"])
    agg = hashlib.sha256()
    for t in tensors:
        agg.update(f"{t['key']} {t['sha256']}\n".encode())
    aggregate = agg.hexdigest()

    cfg = json.loads((snap / "config.json").read_text())
    receipt = {
        "artifact": "frozen-expert-bytes provenance table (reference column)",
        "tier": "reference/pre-training",
        "model_id": "openai/gpt-oss-20b",
        "snapshot_revision": snap.name,
        "quantization": "MXFP4 (on-disk blocks/scales + bf16 biases), as released",
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_local_experts": cfg.get("num_local_experts"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": ("sha256 over each tensor's exact serialized byte range "
                   "(safetensors header data_offsets); no dtype interpretation"),
        "expert_tensor_count": len(tensors),
        "expert_bytes_total": sum(t["nbytes"] for t in tensors),
        "aggregate_sha256_of_sorted_key_hash_lines": aggregate,
        "index_json_sha256": sha256_file(idx_path),
        "config_json_sha256": sha256_file(snap / "config.json"),
        "shards": shards,
        "tensors": tensors,
    }
    Path(args.json_out).write_text(json.dumps(receipt, indent=1))

    md = []
    md.append("# gpt-oss-20b frozen-expert-bytes provenance table\n")
    md.append("**Tier: reference/pre-training (the \"before\" column).** "
              "SHA-256 of the exact serialized bytes of every MXFP4 expert "
              "tensor as released in `openai/gpt-oss-20b` (snapshot `%s`). "
              "The Phase-2.5 native-MXFP4 training demo re-hashes the same "
              "keys after N steps; bit-identity against this table is the "
              "provenance claim in artifact form.\n" % snap.name)
    md.append("- expert tensors: **%d** (%d layers x 32 experts x 6 keys), "
              "**%.2f GiB** of frozen expert bytes" % (
                  len(tensors), cfg.get("num_hidden_layers", 0),
                  receipt["expert_bytes_total"] / 2**30))
    md.append("- aggregate digest (sha256 over sorted `key hash` lines): "
              "`%s`" % aggregate)
    md.append("- generated %s; method: byte-range hash from safetensors "
              "headers, no framework load\n" % receipt["generated_utc"])
    md.append("| shard | bytes | sha256 |")
    md.append("|---|---|---|")
    for name in sorted(shards):
        md.append("| `%s` | %d | `%s` |" % (name, shards[name]["nbytes"],
                                            shards[name]["sha256"]))
    md.append("")
    md.append("| tensor | dtype | shape | bytes | sha256 |")
    md.append("|---|---|---|---|---|")
    for t in tensors:
        md.append("| `%s` | %s | %s | %d | `%s` |" % (
            t["key"], t["dtype"],
            "x".join(str(d) for d in t["shape"]), t["nbytes"], t["sha256"]))
    md.append("")
    Path(args.md_out).write_text("\n".join(md))
    print("tensors=%d total_bytes=%d aggregate=%s" % (
        len(tensors), receipt["expert_bytes_total"], aggregate))
    print("json -> %s\nmd   -> %s" % (args.json_out, args.md_out))


if __name__ == "__main__":
    main()
