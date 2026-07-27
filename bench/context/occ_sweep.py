"""Occupancy + ILP sweep on the post-#43 GEMV. See PREREG-occupancy-reopened.md.

#41 swept BLOCK_N x warps x split_k on the OLD kernel and found nothing, because
it was sector-bound. #45 shows the post-#43 kernel is LATENCY-bound, so warps
(latency hiding) and num_stages (in-flight loads) are re-tested -- num_stages is
the lever #41 never swept.
"""
import argparse, itertools, json, math, statistics, sys
import torch, triton
sys.path.insert(0, "/w")
import gemv_issue_bound as G

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=7)
ap.add_argument("--out", default="")
a_ = ap.parse_args()

dev = "cuda"
E, N, K, T = 8, 3072, 4096, 8
packed, absmax, a, eids, lut = G.make_data(E, N, K, T, dev)
ref = G.reference(packed, absmax, a, eids, lut, N, K)
print(f"# {torch.cuda.get_device_name(0)}  triton {triton.__version__}")
print(f"# post-#43 GEMV (v1_h4), E={E} N={N} K={K} T={T}")


def bench(bn, warps, stages, iters):
    out = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
    grid = (T, triton.cdiv(N, bn))

    def go():
        G._v1_h4[grid](a, packed, absmax, out, lut, eids, K, N,
                       packed.stride(0), packed.stride(1),
                       absmax.stride(0), absmax.stride(1),
                       BLOCK_N=bn, BLOCK_K=64, num_warps=warps, num_stages=stages)
    for _ in range(8):
        go()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); go(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    return statistics.median(ts), err


DEFAULT = (64, 2, 3)          # the shipped decode plan for this shape
base, base_err = bench(*DEFAULT, a_.iters)
print(f"# default BLOCK_N=64 warps=2 stages=3 -> {base:.4f} ms  (err {base_err:.2e})")
print("# all speedups below are PAIRED: default re-timed before each candidate,\n#   median of 3 pair-ratios. A self-pair should read ~1.000x.\n")

# PAIRED measurement: the default is re-timed immediately before every
# candidate and the ratio taken per pair, so shared-GPU drift cancels. A first
# pass with a single up-front baseline reported the DEFAULT CONFIG as 1.283x
# faster than itself -- the A2000 drifts that much between runs.
rows = []
for bn, warps, stages in itertools.product((32, 64, 128), (1, 2, 4, 8), (1, 2, 3, 4, 5, 6)):
    try:
        ratios = []
        for _ in range(3):
            b_ms, _ = bench(*DEFAULT, 3)
            c_ms, err = bench(bn, warps, stages, 3)
            ratios.append(b_ms / c_ms)
        ms = c_ms
    except Exception:
        continue
    if err > 8e-03:
        continue
    rows.append((statistics.median(ratios), bn, warps, stages, ms, err))

rows.sort(reverse=True)
print("# top 12 configs")
for sp, bn, w, st, ms, err in rows[:12]:
    print(f"  BLOCK_N={bn:4d} warps={w} stages={st}  {ms:7.4f} ms  {sp:6.3f}x  err {err:.1e}")

# isolated levers, holding the other two at the shipped default
def best_of(pred):
    c = [r for r in rows if pred(r)]
    return max(c)[0] if c else float("nan")

warps_only = best_of(lambda r: r[1] == 64 and r[3] == 3)
stages_only = best_of(lambda r: r[1] == 64 and r[2] == 2)
print(f"\n# num_warps alone  (BLOCK_N=64, stages=3): {warps_only:.3f}x   [pred 1.10-1.35x]")
print(f"# num_stages alone (BLOCK_N=64, warps=2):  {stages_only:.3f}x   [pred 1.10-1.40x]")
print(f"# best combined:                           {rows[0][0]:.3f}x   [pred >=1.25x to land]")

if a_.out:
    json.dump({"base_ms": base, "default": DEFAULT,
               "rows": [{"speedup": r[0], "block_n": r[1], "warps": r[2],
                         "stages": r[3], "ms": r[4], "err": r[5]} for r in rows],
               "warps_only": warps_only, "stages_only": stages_only,
               "best": rows[0][0]}, open(a_.out, "w"), indent=2)
    print(f"# wrote {a_.out}")
