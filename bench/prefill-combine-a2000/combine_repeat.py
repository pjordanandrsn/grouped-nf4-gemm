"""Kernel-level repeat test of the MXFP4 pipelined prefill combine (grouped-nf4-gemm
mxfp4_pipelined.py _forward_prefill, `out.index_add_(0, rows, dn * wt...)`).

Replays that loop's exact combine on synthetic inputs at Kimi-K3 geometry (topk 16 of 896
experts, k = 16 slots per chunk, hidden 7168): same routing, same dn, same weights, N calls.
Reports how many DISTINCT outputs N identical calls produce, per arm:
  plain   -- the shipped combine, torch's default (non-deterministic) algorithms
  det     -- the shipped combine under torch.use_deterministic_algorithms(True)
  ordered -- the proposed combine: every index_add_ call has unique rows, so no two threads
             race on one address; each token's terms land in ascending-expert order
and whether `ordered` equals a CPU sequential fp32 sum bit for bit.
Usage: python combine_repeat.py [T ...]   (default 6 90)
"""
import hashlib
import sys

import torch

torch.utils.deterministic.fill_uninitialized_memory = False
DEV = "cuda"
TOPK, E, K, N, REPS = 16, 896, 16, 7168, 50


def routing(T, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)])
    dn = torch.randn(T * TOPK, N, generator=g)            # per-pair down output, fp32
    wt = torch.rand(T * TOPK, generator=g)
    return ids, dn, wt


def plan(ids):
    T = ids.shape[0]
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    pair_tok = torch.div(order, TOPK, rounding_mode="floor")
    uniq, counts = torch.unique_consecutive(flat.index_select(0, order), return_counts=True)
    chunks, p = [], 0
    cl = counts.tolist()
    for c0 in range(0, len(cl), K):
        n = sum(cl[c0:c0 + K])
        chunks.append((p, n))
        p += n
    return T, order, pair_tok, chunks


def add_ordered(out, rows, src):
    """out[rows[i]] += src[i] with no two threads on one row per call (see module doc)."""
    n = rows.numel()
    srt, perm = torch.sort(rows, stable=True)
    pos = torch.arange(n, device=rows.device)
    start = torch.ones(n, dtype=torch.bool, device=rows.device)
    start[1:] = srt[1:] != srt[:-1]
    rank = pos - torch.cummax(pos * start, 0).values
    by_rank = torch.argsort(rank, stable=True)
    _, per = torch.unique_consecutive(rank.index_select(0, by_rank), return_counts=True)
    idx = perm.index_select(0, by_rank)
    lo = 0
    for c in per.tolist():
        sel = idx[lo:lo + c]
        out.index_add_(0, rows.index_select(0, sel), src.index_select(0, sel))
        lo += c
    return out


def combine(T, order, pair_tok, chunks, dn, wt, how, dev):
    out = torch.zeros(T, N, dtype=torch.float32, device=dev)
    for p, n in chunks:
        rows = pair_tok[p:p + n]
        # the engine: dn rows are group-sorted, i.e. pair p's dn is dn[order[p]] here
        src = dn.index_select(0, order[p:p + n]) * wt.index_select(0, order[p:p + n])[:, None]
        if how == "ordered":
            add_ordered(out, rows, src)
        else:
            out.index_add_(0, rows, src)
    return out


def sequential_cpu(T, order, pair_tok, chunks, dn, wt):
    out = torch.zeros(T, N, dtype=torch.float32)
    for p, n in chunks:
        for q in range(p, p + n):
            j = int(order[q])
            out[int(pair_tok[q])] += dn[j] * wt[j]          # one fp32 multiply, one fp32 add
    return out


def h(t):
    t = t.detach().cpu().contiguous()
    t = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  reps {REPS}")
for T in [int(a) for a in sys.argv[1:]] or [6, 90]:
    ids, dn, wt = routing(T, seed=T)
    T, order, pair_tok, chunks = plan(ids)
    mult = []
    for p, n in chunks:
        _, c = torch.unique(pair_tok[p:p + n], return_counts=True)
        mult.append(int(c.max()))
    print(f"\nT={T}: {len(chunks)} chunks, {T*TOPK} pairs, max rows-per-token within a chunk "
          f"{max(mult)} (chunks with a repeat: {sum(m > 1 for m in mult)}/{len(chunks)})")
    g = [x.to(DEV) for x in (order, pair_tok, dn, wt)]
    ref = sequential_cpu(T, order, pair_tok, chunks, dn, wt)
    for how, det in (("plain", False), ("det", True), ("ordered", False)):
        torch.use_deterministic_algorithms(det)
        hs, hb, outs = set(), set(), []
        for _ in range(REPS):
            o = combine(T, g[0], g[1], chunks, g[2], g[3], how, DEV)
            torch.cuda.synchronize()
            hs.add(h(o))
            hb.add(h(o.to(torch.bfloat16)))
            if len(outs) < 2 and h(o) not in {h(x) for x in outs}:
                outs.append(o.cpu())
        torch.use_deterministic_algorithms(False)
        o = o.cpu()
        extra = ""
        if len(outs) > 1:
            d = (outs[0] - outs[1]).abs()
            extra = f"  two distinct fp32 outputs differ in {int((d > 0).sum())} of {d.numel()} elements, max {float(d.max()):.3e}"
        print(f"  {how:8s} distinct fp32 outputs {len(hs):2d}/{REPS}   distinct after bf16 cast "
              f"{len(hb):2d}/{REPS}   == CPU sequential: {torch.equal(o, ref)}{extra}")
