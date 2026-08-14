#!/usr/bin/env python3
"""C2 fidelity gate: the capturability change must move NO VALUE.

The change is a call-path change -- index tensors reach the device through one
pinned transfer instead of several pageable ones, and `expert_ids` stops being
iterated in Python. No kernel source, no tiling constant, no dispatch threshold,
no dtype. So every output must be BITWISE identical, not merely close: a
tolerance here would hide exactly the class of mistake this change could make.

Run once per kernel directory, then diff the two receipts:

    python ab_capturability_bitexact.py --kdir /path/to/old/kernel --out old.pt
    python ab_capturability_bitexact.py --kdir /path/to/new/kernel --out new.pt
    python ab_capturability_bitexact.py --compare old.pt new.pt

Two directories rather than two imports on one path: the old `nf4_qlora` does
`from nf4_grouped import ...` at call time, so a single process would silently
mix the two versions and the comparison would be against itself.
"""
from __future__ import annotations

import argparse
import sys

import torch

E, N, K, G, M, RANK = 16, 256, 256, 6, 8, 8
SEED = 20260814


def build_inputs(device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    w = (torch.randn(E, N, K, generator=g, dtype=torch.float32) * 0.02)
    a = (torch.randn(G * M, K, generator=g, dtype=torch.float32) * 0.5)
    lA = (torch.randn(E, RANK, K, generator=g, dtype=torch.float32) * 0.01)
    lB = (torch.randn(E, N, RANK, generator=g, dtype=torch.float32) * 0.01)
    go = (torch.randn(G * M, N, generator=g, dtype=torch.float32) * 0.1)
    return (w.to(device), a.to(device, torch.bfloat16), lA.to(device, torch.bfloat16),
            lB.to(device, torch.bfloat16), go.to(device, torch.bfloat16))


def run(kdir, device="cuda"):
    sys.path.insert(0, kdir)
    from bitsandbytes import functional as BF
    from nf4_grouped import (build_group_tiles, dgrad_4bit_grouped,
                             gemm_4bit_grouped, repack_from_bnb)
    from nf4_qlora import fused_grouped_lora, lora_delta_grouped

    w, a, lA, lB, go = build_inputs(device)
    packed, states = [], []
    for e in range(E):
        q, st = BF.quantize_4bit(w[e].to(torch.bfloat16), blocksize=64,
                                 quant_type="nf4")
        packed.append(q)
        states.append(st)
    B_pack, A_scale = repack_from_bnb(packed, states, N, K)

    # Group sizes deliberately jagged, including a zero group and a group that
    # spans more than one M-tile, so build_group_tiles has real work to do.
    sizes = [M, 1, 0, M * 3, 2, M]
    assert len(sizes) == G
    a_cat = a[:sum(sizes)].contiguous()
    eids_list = [0, 3, 7, 5, 11, 2]
    eids_dev = torch.tensor(eids_list, dtype=torch.int32, device=device)

    r = {}
    for tag, eids in (("list", eids_list), ("dev", eids_dev)):
        r[f"gemm_{tag}"] = gemm_4bit_grouped(a_cat, B_pack, A_scale, sizes, eids)
        r[f"dgrad_{tag}"] = dgrad_4bit_grouped(go[:sum(sizes)].contiguous(),
                                               B_pack, A_scale, sizes, eids)
        r[f"lora_{tag}"] = lora_delta_grouped(a_cat, lA, lB, sizes, eids, 2.0)
        x = a_cat.detach().clone().requires_grad_(True)
        out = fused_grouped_lora(x, B_pack, A_scale, sizes, eids,
                                 lora_A=lA.detach().clone().requires_grad_(True),
                                 lora_B=lB.detach().clone().requires_grad_(True),
                                 scaling=2.0, dgrad_kernel=True)
        out.float().pow(2).mean().backward()
        r[f"fused_out_{tag}"] = out.detach()
        r[f"fused_gradx_{tag}"] = x.grad.detach()
        # dgrad_kernel=False exercises the per-expert fallback loop, which is
        # where the `.tolist()` materialisation moved to.
        x2 = a_cat.detach().clone().requires_grad_(True)
        out2 = fused_grouped_lora(x2, B_pack, A_scale, sizes, eids,
                                  lora_A=lA.detach().clone().requires_grad_(True),
                                  lora_B=lB.detach().clone().requires_grad_(True),
                                  scaling=2.0, dgrad_kernel=False)
        out2.float().pow(2).mean().backward()
        r[f"loop_gradx_{tag}"] = x2.grad.detach()

    for bm in (16, 32, 64, 128):
        t0, t1, t2 = build_group_tiles(sizes, bm, device)
        r[f"tiles{bm}_row0"], r[f"tiles{bm}_rows"], r[f"tiles{bm}_grp"] = t0, t1, t2

    # The skew fallback in lora_delta_grouped (past _PAD_WASTE_LIMIT) takes the
    # per-expert loop; it must move no value either.
    skew = [1, 1, 1, 1, 1, 200]
    r["lora_skew"] = lora_delta_grouped(a[:sum(skew)].contiguous(), lA, lB, skew,
                                        eids_list, 2.0)
    r["lora_skew_dev"] = lora_delta_grouped(a[:sum(skew)].contiguous(), lA, lB,
                                            skew, eids_dev, 2.0)
    return {k: v.detach().cpu() for k, v in r.items() if v is not None}


def compare(pa, pb):
    a, b = torch.load(pa, weights_only=False), torch.load(pb, weights_only=False)
    keys = sorted(set(a) | set(b))
    bad = []
    for k in keys:
        if k not in a or k not in b:
            bad.append((k, "MISSING"))
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append((k, f"shape/dtype {tuple(x.shape)}{x.dtype} vs "
                           f"{tuple(y.shape)}{y.dtype}"))
        elif not torch.equal(x, y):
            d = (x.float() - y.float()).abs().max().item()
            bad.append((k, f"VALUES DIFFER, max abs {d:.3e}"))
    print(f"{len(keys)} tensors compared")
    for k, why in bad:
        print(f"  MISMATCH {k}: {why}")
    print("BITEXACT_OK" if not bad else f"BITEXACT_FAIL ({len(bad)} mismatches)")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kdir")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        raise SystemExit(compare(*args.compare))
    r = run(args.kdir)
    torch.save(r, args.out)
    print(f"wrote {len(r)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
