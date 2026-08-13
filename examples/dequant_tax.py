#!/usr/bin/env python3
"""The dequantize tax, measured on YOUR GPU in about a minute.

Any stack storing MoE experts at 4 bits and running a bf16 GEMM must decode the
weights to bf16 and read them back. This times that round trip against computing on
the packed bytes directly, at a census shape, across three points on the M axis.
Synthetic weights, no download.

    pip install grouped-nf4-gemm && python examples/dequant_tax.py   # --sweep: more
"""
import argparse, statistics, sys, time  # noqa: E401 — one-file example

import torch

SHAPE = {"name": "Qwen3-30B-A3B gate_up", "N": 1536, "K": 2048, "E": 128, "k": 8}
CENSUS = [SHAPE, {"name": "OLMoE-1B-7B gate_up", "N": 2048, "K": 2048, "E": 64, "k": 8},
          {"name": "gpt-oss-120b down", "N": 2880, "K": 2880, "E": 128, "k": 4}]
CPU_NOTE = """No CUDA device — the fused kernel is CUDA + Triton only, so the timing this
script exists to show cannot run here. Needs an NVIDIA GPU, torch with CUDA, triton>=3.4
(Linux). The decode oracle the kernel is checked against DOES run on CPU:
    packed, absmax = nf4_pack_ref.quantize_pack_nf4(torch.randn(256, 512))
    w = nf4_grouped.dequant_ref(packed, absmax, 256, 512)"""
FOOTER = """
  ratio      dequant-then-GEMM / fused; >1 = computing on the packed bytes won.
  self-pair  the fused arm against itself — the instrument's own spread. ANY RATIO
             INSIDE IT IS NOT A MEASUREMENT (self-pair 0.95 + ratio 1.09 = noise).

Boundaries of what just ran
  * Synthetic weights, uniform routing — real router skew moves the prefill row.
  * Small-expert shapes LOSE (0.24-0.35x in this repo's census); the kernel ships a
    dispatch floor sending them back to the dequant path. --sweep shows the spread.
  * Already-bf16-resident weights: no round trip to skip, so a bf16-native grouped
    GEMM is the faster tool and none of this applies.
  * The margin decays as the cell becomes compute-bound, and on low-SM cards prefill
    inverts outright. That decay is the honest edge of the claim.

Edit SHAPE at the top of this file for your own N/K/E/top_k and re-run."""

def build_stack(N, K, E, dev, seed=0):
    """bnb's layout IS this kernel's, so with bnb the dequant arm gets its OPTIMIZED
    CUDA decode. Without it, the reference decode — an oracle, not a fast path — so
    the ratio becomes an upper bound and the output labels it."""
    g = torch.Generator().manual_seed(seed)
    draw = lambda: (torch.randn(N, K, generator=g) * 0.02).to(dev)  # noqa: E731
    try:
        from bitsandbytes import functional as F
        qs = [F.quantize_4bit(draw().to(torch.bfloat16), blocksize=64, quant_type="nf4")
              for _ in range(E)]
        return (torch.stack([q.view(N, K // 2) for q, _ in qs]),
                torch.stack([st.absmax.view(N, K // 64).float() for _, st in qs]),
                [st for _, st in qs])
    except ImportError:
        from nf4_pack_ref import quantize_pack_nf4
        pk, am = zip(*(quantize_pack_nf4(draw()) for _ in range(E)))
        return torch.stack(pk), torch.stack(am), None

def make_groups(regime, N, K, E, k, dev, seed=7):
    """(a_cat [T,K] group-sorted, per-group sizes, expert_ids) — what the op takes."""
    g = torch.Generator().manual_seed(seed)
    if regime == "prefill_2048":
        ids, m = list(range(E)), max(1, round(2048 * k / E))
    else:
        ids, m = list(range(k)), (1 if regime == "decode_bs1" else 8)
    a = (torch.randn(len(ids) * m, K, generator=g) * 0.5).to(dev, torch.bfloat16)
    return a, [m] * len(ids), torch.tensor(ids, dtype=torch.int32, device=dev)

def timed(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)

def joules(fn, h, secs=1.0):
    """J per call from NVML's energy counter."""
    import pynvml
    torch.cuda.synchronize()
    j0, t0, n = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.monotonic(), 0
    while time.monotonic() - t0 < secs:
        fn(); n += 1
    torch.cuda.synchronize()
    return (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - j0) / 1e3 / n

def run_shape(sh, reps, dev, h):
    from nf4_grouped import dequant_ref, gemm_4bit_grouped
    N, K, E, k = sh["N"], sh["K"], sh["E"], sh["k"]
    packed, absmax, states = build_stack(N, K, E, dev)
    bnbf = __import__("bitsandbytes", fromlist=["functional"]).functional if states else None
    print(f"\n{sh['name']}   N={N} K={K} E={E} top_k={k}   synthetic weights")
    print("  dequant arm: " + ("bnb dequantize_4bit (optimized)" if states else
          "dequant_ref (ORACLE, not optimized -> ratio is an UPPER BOUND)"))
    print(f"  {'regime':<14}{'tokens':>7}{'fused ms':>10}{'dequant ms':>12}{'ratio':>7}"
          f"{'self-pair':>11}{'J/tok fused':>13}{'J/tok deq':>12}")
    for regime in ("decode_bs1", "decode_m8", "prefill_2048"):
        a, sizes, ids = make_groups(regime, N, K, E, k, dev)
        fused = lambda: gemm_4bit_grouped(a, packed, absmax, sizes, ids)  # noqa: E731
        def deq():  # what a bf16 grouped GEMM must do with a 4-bit checkpoint
            i = 0
            for gi, e in enumerate(ids.tolist()):
                w = (bnbf.dequantize_4bit(packed[e].reshape(-1, 1), states[e]).view(N, K)
                     if states else dequant_ref(packed[e], absmax[e], N, K).to(torch.bfloat16))
                a[i:i + sizes[gi]] @ w.t()
                i += sizes[gi]
        # base, comparator, base again — the 2nd base IS the self-pair, so the
        # instrument's spread prints beside the ratio it qualifies.
        rs, sps, base = [], [], 0.0
        for _ in range(reps):
            b1, d, b2 = timed(fused), timed(deq), timed(fused)
            rs.append(d / b1); sps.append(b2 / b1); base = b1
        ratio, sp = statistics.median(rs), statistics.median(sps)
        jf, jd = (joules(fused, h) / a.shape[0], joules(deq, h) / a.shape[0]) if h else (0, 0)
        print(f"  {regime:<14}{a.shape[0]:>7}{base:>10.3f}{base * ratio:>12.3f}{ratio:>7.2f}"
              f"{sp:>11.3f}{(f'{jf:.2e}' if h else 'n/a'):>13}"
              f"{(f'{jd:.2e}' if h else 'n/a'):>12}")
    a, sizes, ids = make_groups("decode_m8", N, K, E, k, dev)
    out, num, den, i = gemm_4bit_grouped(a, packed, absmax, sizes, ids), 0.0, 0.0, 0
    for gi, e in enumerate(ids.tolist()):
        r = a[i:i + sizes[gi]].double() @ dequant_ref(packed[e], absmax[e], N, K).double().t()
        num += (out[i:i + sizes[gi]].double() - r).pow(2).sum().item()
        den += r.pow(2).sum().item(); i += sizes[gi]
    print(f"  fidelity: fused b_rel vs fp64 over the same decoded values = {(num / den) ** .5:.2e}")

def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the dequantize tax on your GPU.")
    ap.add_argument("--sweep", action="store_true", help="run every census shape")
    args = ap.parse_args()
    if not torch.cuda.is_available():  # CI exercises this path; exit 0, not a traceback
        print(CPU_NOTE)
        return 0
    print(f"{torch.cuda.get_device_name(0)}   torch {torch.__version__}   "
          f"cc {'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    except Exception as e:
        h = None
        print(f"  energy: NVML unavailable ({type(e).__name__}) — J/token reads n/a")
    for sh in (CENSUS if args.sweep else [SHAPE]):
        run_shape(sh, 3, "cuda", h)
    print(FOOTER)
    return 0
if __name__ == "__main__":
    sys.exit(main())
