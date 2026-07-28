#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Adjudicate `PREREG-nvme-bake-determinism.md`: did two independent boxes
turn the same shipped checkpoint into the same arena bytes?

Compares two bake manifests row-by-row, segment-by-segment, over their
overlapping (layer, expert) keys:

  D3 (control, first): the SOURCE hashes each bake recorded for the tensors
      it consumed must agree. If the two downloads differ, D1 is void — the
      finding would be about the download, not the bake.
  D1 (primary): fraction of overlapping segment sha256 that match.

Reports the fraction, the per-suffix breakdown (so a scales-only or
blocks-only divergence is visible rather than averaged away), and a sample
of mismatches. Exit 0 iff D1 >= --bar and D3 is clean.

  python3 compare_bake_determinism.py --a A.manifest.json --b B.manifest.json \
      --label-a L40S --label-b QNAP --bar 0.999 --out determinism.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys


def index_manifest(path):
    m = json.load(open(path))
    segs, srcs = {}, {}
    for row in m["rows"]:
        for s in row["segments"]:
            key = (row["layer"], row["expert"], s["suffix"])
            segs[key] = s["sha256"]
            for src in (s.get("sources") or ([{
                    "source_file": s["source_file"],
                    "source_range": s["source_range"],
                    "sha256": s["sha256"]}] if "source_file" in s else [])):
                srcs[(row["layer"], row["expert"], s["suffix"],
                      src["source_file"], tuple(src["source_range"]))] = src["sha256"]
    return m, segs, srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--bar", type=float, default=0.999)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ma, sa, qa = index_manifest(args.a)
    mb, sb, qb = index_manifest(args.b)
    common = sorted(set(sa) & set(sb))
    if not common:
        print("NO OVERLAP between the two manifests — nothing to compare")
        return 2

    # ---- D3: source control -------------------------------------------
    src_common = set(qa) & set(qb)
    src_bad = [k for k in src_common if qa[k] != qb[k]]
    d3_ok = not src_bad
    print(f"D3 source control: {len(src_common)} shared source records, "
          f"{len(src_bad)} differ -> {'CLEAN' if d3_ok else 'DIVERGENT'}")
    if src_bad:
        for k in src_bad[:3]:
            print(f"   layer={k[0]} expert={k[1]} {k[2]} file={k[3]}")

    # ---- D1: arena segment hashes --------------------------------------
    match = [k for k in common if sa[k] == sb[k]]
    frac = len(match) / len(common)
    per_suffix = collections.Counter()
    per_suffix_tot = collections.Counter()
    for k in common:
        per_suffix_tot[k[2]] += 1
        if sa[k] == sb[k]:
            per_suffix[k[2]] += 1

    print(f"\nD1 arena determinism: {len(match)}/{len(common)} segments match "
          f"= {frac:.6f}  (bar {args.bar})")
    print(f"   layers compared: {len(set(k[0] for k in common))}, "
          f"experts: {len(set(k[1] for k in common))}")
    for suf in sorted(per_suffix_tot):
        t, m_ = per_suffix_tot[suf], per_suffix[suf]
        print(f"   {suf:28s} {m_:6d}/{t:6d} = {m_/t:.6f}")
    bad = [k for k in common if sa[k] != sb[k]]
    for k in bad[:5]:
        print(f"   MISMATCH layer={k[0]} expert={k[1]} {k[2]}: "
              f"{args.label_a} {sa[k][:12]} != {args.label_b} {sb[k][:12]}")

    verdict = bool(frac >= args.bar and d3_ok)
    print(f"\nVERDICT: D1 {'PASS' if frac >= args.bar else 'FAIL'}"
          f"{'' if d3_ok else ' (VOID — D3 source control divergent)'}")
    rep = {
        "prereg": "PREREG-nvme-bake-determinism.md",
        "a": {"label": args.label_a, "path": args.a,
              "bake_mode": ma.get("bake_mode"),
              "quantizer": ma.get("quantizer")},
        "b": {"label": args.label_b, "path": args.b,
              "bake_mode": mb.get("bake_mode"),
              "quantizer": mb.get("quantizer")},
        "segments_compared": len(common), "segments_match": len(match),
        "D1_fraction": frac, "D1_bar": args.bar,
        "D3_source_records": len(src_common), "D3_divergent": len(src_bad),
        "D3_clean": d3_ok,
        "per_suffix": {s: {"match": per_suffix[s], "total": per_suffix_tot[s]}
                       for s in per_suffix_tot},
        "mismatch_sample": [{"layer": k[0], "expert": k[1], "suffix": k[2],
                             "a": sa[k], "b": sb[k]} for k in bad[:20]],
        "pass": verdict,
    }
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
