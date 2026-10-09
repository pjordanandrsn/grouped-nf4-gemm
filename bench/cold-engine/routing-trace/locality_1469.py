"""experts4bit-qlora#1469, item 2: what nearby-token expert reuse is worth in bytes, on the committed routing traces.

Stepped MoE (arXiv:2610.07348) holds one routing decision for a segment of S tokens, so a segment's experts are loaded
once. Existing checkpoints route every token, and their routing is not changed here. What this measures is how much
reuse their *exact* routing already has within nearby tokens, and what that is worth in bytes moved per decoded token
and in link time, beside the replacement policies this campaign already scored.

Inputs, all committed and read as they are (no box, no GPU):

* the 12 decode traces ``{olmoe,granite,qwen}_{prose,code,math,dialogue}.jsonl`` (``capture_routing.py``; 512
  autoregressive steps each; ``qwen`` is Qwen1.5-MoE-A2.7B), loaded with ``score_policies.load``;
* ``oracle-headroom.json`` (``oracle_headroom.py``): row transfers for the shipped device cache, pure LRU and Belady's
  optimum at 1, 1.5 and 2 steps' worth of rows. Read, not recomputed.

Per trace, for windows of W in (1, 16, 32, 64, 128) tokens, non-overlapping and within the sequence:

* **union**: distinct experts a layer routes to within the window (mean / p50 / p95 / max over windows and layers);
* **loads per token**: union / W per layer, summed over layers -- the rows a token costs if a cache held exactly
  the current window's experts and nothing across windows. W = 1 is no reuse at all (k per layer);
* **churn**: the fraction of a layer's top-k that changes from one token to the next.

Bytes: one row is one expert of one layer, gate, up and down in NF4 with an fp32 absmax per 64 weights, the host
format experts4bit-qlora stages (``3HI/2 + 3HI/64 x 4`` bytes). Link floors divide bytes per token by measured
host-to-device and NVMe throughputs (``--pcie-gbs``, ``--nvme-gbs``; defaults from committed receipts, see below).
A floor is a lower bound on transfer time per token, not a prediction of a step.

    python bench/cold-engine/routing-trace/locality_1469.py [--out locality-1469.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle_headroom import belady, keystream  # noqa: E402
from score_policies import load  # noqa: E402

PROMPTS = ("prose", "code", "math", "dialogue")
WINDOWS = (1, 16, 32, 64, 128)
#: hidden size and routed-expert intermediate size, from each model's config.json on the Hub
#: (allenai/OLMoE-1B-7B-0924, ibm-granite/granite-3.0-3b-a800m-instruct, Qwen/Qwen1.5-MoE-A2.7B)
SHAPES = {"olmoe": (2048, 1024), "granite": (1536, 512), "qwen": (2048, 1408)}
#: measured host-to-device copy throughput (loggetta receipts' ``link_h2d_gbps``: RTX A2000 median of 10, RTX 5090
#: median of 3) and NVMe sequential read on the owned NAS (bench/nvme/receipts, 3.30-3.47 GB/s)
LINKS = {"pcie_a2000": 6.24, "pcie_5090": 20.75, "nvme_nas": 3.3}


def row_bytes(hidden: int, inter: int) -> int:
    n = 3 * hidden * inter                         # gate, up, down
    return n // 2 + (n // 64) * 4                  # NF4 codes + fp32 absmax per 64 weights


def windows_of(recs, layers: int, w: int):
    """Per layer, the distinct experts of each full window of w consecutive decode steps."""
    out = {L: [] for L in range(layers)}
    for start in range(0, len(recs) - w + 1, w):
        chunk = recs[start:start + w]
        for L in range(layers):
            out[L].append(len({e for r in chunk for e in r["routed"].get(str(L), ())}))
    return out


def churn(recs, layers: int, k: int) -> float:
    vals = []
    for a, b in zip(recs, recs[1:]):
        for L in range(layers):
            prev, cur = set(a["routed"].get(str(L), ())), set(b["routed"].get(str(L), ()))
            if prev and cur:
                vals.append(len(cur - prev) / k)
    return statistics.fmean(vals) if vals else float("nan")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def census(path):
    meta, recs = load(path)
    L, k = meta["layers"], meta["top_k"]
    out = {"meta": meta, "steps": len(recs), "churn": churn(recs, L, k), "windows": {}}
    for w in WINDOWS:
        per_layer = windows_of(recs, L, w)
        flat = [u for v in per_layer.values() for u in v]
        # the rows a cache holding exactly the current window's experts needs: every layer's largest window
        cap = sum(max(v) for v in per_layer.values())
        out["windows"][w] = {
            "union_mean": statistics.fmean(flat), "union_p50": pct(flat, 0.5), "union_p95": pct(flat, 0.95),
            "union_max": max(flat),
            # rows a token costs with a cache holding exactly the current window's experts
            "loads_per_token": sum(statistics.fmean(v) for v in per_layer.values()) / w,
            "capacity_rows": cap,
            # the optimum any replacement policy reaches at that same capacity (Belady, oracle_headroom.belady)
            "belady_loads_per_token": belady(keystream(recs), cap) / len(recs),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--out", default=None)
    for name, gbs in LINKS.items():
        ap.add_argument(f"--{name.replace('_', '-')}-gbs", type=float, default=gbs, dest=name)
    a = ap.parse_args()
    links = {n: getattr(a, n) for n in LINKS}
    oracle = json.load(open(os.path.join(a.dir, "oracle-headroom.json")))["rows"]

    result = {"links_gbs": links, "row_bytes": {m: row_bytes(*s) for m, s in SHAPES.items()}, "traces": [],
              "policies": []}
    print("| model | prompt | churn | " + " | ".join(f"union W={w}" for w in WINDOWS) + " | "
          + " | ".join(f"loads/token W={w}" for w in WINDOWS) + " |")
    print("|---|---|---|" + "---|" * (2 * len(WINDOWS)))
    for m in SHAPES:
        for p in PROMPTS:
            f = os.path.join(a.dir, f"{m}_{p}.jsonl")
            if not os.path.exists(f):
                continue
            c = census(f)
            c.update(model=m, prompt=p)
            result["traces"].append(c)
            ws = c["windows"]
            print(f"| {m} | {p} | {c['churn']:.3f} | "
                  + " | ".join(f"{ws[w]['union_mean']:.1f}" for w in WINDOWS) + " | "
                  + " | ".join(f"{ws[w]['loads_per_token']:.1f}" for w in WINDOWS) + " |")

    print("\nBytes per decoded token and link floors (ms per token), mean over the four prompts")
    print("| model | row MB | source | rows/token | MB/token | " + " | ".join(f"{n} ms" for n in links) + " |")
    print("|---|---|---|---|---|" + "---|" * len(links))
    for m, shape in SHAPES.items():
        rb = row_bytes(*shape)
        tr = [t for t in result["traces"] if t["model"] == m]
        if not tr:
            continue
        lines = []
        for w in WINDOWS:
            cap = round(statistics.fmean(t["windows"][w]["capacity_rows"] for t in tr))
            lines.append((f"window W={w}, ~{cap} rows", statistics.fmean(t["windows"][w]["loads_per_token"] for t in tr)))
            lines.append((f"belady at that capacity, ~{cap} rows",
                          statistics.fmean(t["windows"][w]["belady_loads_per_token"] for t in tr)))
        for held in (1.0, 1.5, 2.0):
            rows = [r for r in oracle if r["model"] == m and r["steps_held"] == held]
            if not rows:
                continue
            cap = rows[0]["cap"]
            for pol in ("cache", "lru", "belady"):
                steps = tr[0]["steps"]
                lines.append((f"{pol}, {cap} rows ({held:g} steps held)",
                              statistics.fmean(r[pol] for r in rows) / steps))
        for src, per_tok in lines:
            mb = per_tok * rb / 1e6
            result["policies"].append({"model": m, "source": src, "rows_per_token": per_tok, "mb_per_token": mb,
                                       "floor_ms": {n: mb / g for n, g in links.items()}})
            print(f"| {m} | {rb / 1e6:.2f} | {src} | {per_tok:.1f} | {mb:.1f} | "
                  + " | ".join(f"{mb / g:.2f}" for g in links.values()) + " |")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
        print(f"\nreceipt -> {a.out}")


if __name__ == "__main__":
    main()
