# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Contiguous vs coalesced packed layout, same kernel. See PREREG-coalesced-repack.md.

#50 measured the decode GEMV within 1.01-1.06x of a stripped kernel doing the
same required work, and the whole remaining gap as the access pattern: strided
150.0 vs coalesced 487.7 GB/s on an A6000 (3.25x), 402.9 vs 1404.3 on a 4090
(3.49x). This tests whether transposing the packed store recovers it.

The two arms differ ONLY in `B.stride()`. The kernel is stride-generic, the
logical bytes are identical (asserted), and `B.shape` is unchanged — so this is
a layout comparison and not a code comparison.

Paired timing throughout: the contiguous arm is re-timed immediately before each
coalesced arm and the ratio taken per pair, because an unpaired sweep on a
drifting box once reported the default config as 1.283x faster than itself (#46).
A self-pair is included and must read ~1.000x or the run is void.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

import torch
import triton

sys.path.insert(0, "kernel")
from nf4_grouped import gemm_4bit_grouped  # noqa: E402

# (name, K, N, E) — the census decode shapes, as used by #43's landed measurement
SHAPES = [("olmoe_gu", 2048, 2048, 64), ("olmoe_dn", 2048, 1024, 64),
          ("qwen_gu", 1536, 2048, 128), ("qwen_dn", 2048, 768, 128),
          ("gemma_gu", 1408, 2816, 128), ("gemma_dn", 2816, 704, 128),
          ("gptoss_gu", 5760, 2880, 128), ("gptoss_dn", 2880, 2880, 128)]


def build(E, N, K, T, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g)
    am = (torch.rand(E, N, K // 64, generator=g) * .5 + .05).float().to(dev)
    a = (torch.randn(T, K, generator=g) * .1).bfloat16().to(dev)
    eids = (torch.arange(T) % E).to(torch.int32).to(dev)

    b_contig = raw.to(dev)
    b_coal = torch.empty(E, K // 2, N, dtype=torch.uint8, device=dev).transpose(1, 2)
    b_coal.copy_(b_contig)
    # the arms must differ ONLY in strides
    assert b_contig.shape == b_coal.shape
    assert torch.equal(b_contig, b_coal), "arms hold different logical bytes"
    assert b_contig.stride() != b_coal.stride(), "layouts did not actually differ"
    return b_contig, b_coal, am, a, eids


def timed(B, am, a, eids, sizes, iters):
    def go():
        return gemm_4bit_grouped(a, B, am, sizes, eids)
    for _ in range(8):
        go()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); out = go(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print(f"# {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} "
          f"triton {triton.__version__}")
    print(f"# T={args.T}  paired, median of {args.pairs} pair-ratios\n")

    rows, sp = [], []
    for name, K, N, E in SHAPES:
        bc, bt, am, a, eids = build(E, N, K, args.T, dev)
        sizes = [1] * args.T
        ratios, ms_c, ms_t, agree = [], None, None, None
        for _ in range(args.pairs):
            ms_c, o_c = timed(bc, am, a, eids, sizes, args.iters)
            ms_t, o_t = timed(bt, am, a, eids, sizes, args.iters)
            ratios.append(ms_c / ms_t)              # >1 == coalesced is faster
            agree = ((o_t.float() - o_c.float()).abs().max().item()
                     / max(o_c.float().abs().max().item(), 1e-9))
        r = statistics.median(ratios)
        rows.append({"shape": name, "K": K, "N": N, "E": E, "contig_ms": ms_c,
                     "coal_ms": ms_t, "speedup": r, "agreement": agree})
        sp.append(r)
        gate = "OK " if agree <= 8e-03 else "FAIL"
        print(f"  {name:11s} contig {ms_c:7.4f}  coal {ms_t:7.4f}  "
              f"{r:6.3f}x   agree {agree:.2e} [{gate}]")

    # self-pair: contiguous against itself, same protocol -- must read ~1.000x
    bc, _, am, a, eids = build(64, 2048, 2048, args.T, dev)
    sizes = [1] * args.T
    selfs = []
    for _ in range(args.pairs):
        m1, _ = timed(bc, am, a, eids, sizes, args.iters)
        m2, _ = timed(bc, am, a, eids, sizes, args.iters)
        selfs.append(m1 / m2)
    self_pair = statistics.median(selfs)

    gm = math.exp(sum(math.log(x) for x in sp) / len(sp))
    worst = min(sp)
    bad = [r["shape"] for r in rows if r["agreement"] > 8e-03]
    print(f"\n# self-pair (contig vs contig): {self_pair:.3f}x  "
          f"{'OK' if 0.95 < self_pair < 1.05 else 'VOID -- too noisy'}")
    print(f"# geomean {gm:.3f}x   worst shape {worst:.3f}x   "
          f"agreement failures: {bad or 'none'}")
    print(f"# prereg bar: >=1.6x geomean AND no shape <1.2x AND agreement <=8e-03")
    print(f"# -> {'MEETS the landing bar on this card' if (gm >= 1.6 and worst >= 1.2 and not bad) else 'DOES NOT meet the bar on this card'}")

    if args.out:
        json.dump({"device": p.name, "sm": f"{p.major}{p.minor}", "rows": rows,
                   "geomean": gm, "worst": worst, "self_pair": self_pair},
                  open(args.out, "w"), indent=2)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
