"""Kernel-level receipt for the grouped-nf4-gemm prefill-combine PR, run with the PR's own code.

Same replay as combine_repeat.py: the chunked combine of mxfp4_pipelined._forward_prefill at
Kimi-K3 geometry (topk 16 of 896 experts, k = 16 slots per chunk, width 7168), identical
inputs every call. Arm `pr` is the PR's `_index_add_ordered_`, imported from the PR tree.
Per T: distinct outputs over 50 calls for plain / det / pr, equality with a CPU sequential
fp32 sum, and the combine's own time (CUDA events, median of 20 after 3 warm-ups), plain vs pr.
Usage: python combine_pr.py PR_KERNEL_DIR [T ...]
"""
import hashlib
import statistics
import sys

import torch

sys.path.insert(0, sys.argv[1])
import mxfp4_pipelined  # noqa: E402
from mxfp4_pipelined import _index_add_ordered_  # noqa: E402

torch.utils.deterministic.fill_uninitialized_memory = False
TOPK, E, K, N, REPS = 16, 896, 16, 7168, 50


def routing(T, seed):                                   # as combine_repeat.py
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)])
    return ids, torch.randn(T * TOPK, N, generator=g), torch.rand(T * TOPK, generator=g)


def plan(ids):                                          # as combine_repeat.py
    T = ids.shape[0]
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    pair_tok = torch.div(order, TOPK, rounding_mode="floor")
    _, counts = torch.unique_consecutive(flat.index_select(0, order), return_counts=True)
    cl, chunks, p = counts.tolist(), [], 0
    for c0 in range(0, len(cl), K):
        n = sum(cl[c0:c0 + K])
        chunks.append((p, n))
        p += n
    return T, order, pair_tok, chunks


def combine(T, order, pair_tok, chunks, dn, wt, how):
    out = torch.zeros(T, N, dtype=torch.float32, device=dn.device)
    for p, n in chunks:
        rows = pair_tok[p:p + n]
        src = dn.index_select(0, order[p:p + n]) * wt.index_select(0, order[p:p + n])[:, None]
        if how == "pr":
            _index_add_ordered_(out, rows, src)
        else:
            out.index_add_(0, rows, src)                # the shipped combine
    return out


def sequential_cpu(T, order, pair_tok, chunks, dn, wt):
    out = torch.zeros(T, N, dtype=torch.float32)
    for p, n in chunks:
        for q in range(p, p + n):
            j = int(order[q])
            out[int(pair_tok[q])] += dn[j] * wt[j]      # one fp32 multiply, one fp32 add
    return out


def h(t):
    t = t.detach().cpu().contiguous()
    t = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  reps {REPS}  "
      f"helper from {mxfp4_pipelined.__file__}")
for T in [int(a) for a in sys.argv[2:]] or [6, 90, 512]:
    ids, dn, wt = routing(T, seed=T)
    T, order, pair_tok, chunks = plan(ids)
    od, pt, dv, wv = (x.cuda() for x in (order, pair_tok, dn, wt))
    ref = sequential_cpu(T, order, pair_tok, chunks, dn, wt)
    print(f"\nT={T}: {len(chunks)} chunks, {T * TOPK} pairs")
    for how, det in (("plain", False), ("det", True), ("pr", False)):
        torch.use_deterministic_algorithms(det)
        hs, hb = set(), set()
        for _ in range(REPS):
            o = combine(T, od, pt, chunks, dv, wv, how)
            torch.cuda.synchronize()
            hs.add(h(o))
            hb.add(h(o.to(torch.bfloat16)))
        torch.use_deterministic_algorithms(False)
        print(f"  {how:6s} distinct fp32 {len(hs):2d}/{REPS}  distinct bf16 {len(hb):2d}/{REPS}  "
              f"== CPU sequential: {torch.equal(o.cpu(), ref)}")
    for how in ("plain", "pr"):
        ts = []
        for i in range(23):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            a.record()
            combine(T, od, pt, chunks, dv, wv, how)
            b.record()
            torch.cuda.synchronize()
            if i >= 3:
                ts.append(a.elapsed_time(b) * 1000)
        med = statistics.median(ts)
        print(f"  time {how:6s} median {med:9.1f} us per combine ({med / len(chunks):7.1f} us per chunk)")
