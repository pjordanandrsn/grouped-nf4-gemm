#!/usr/bin/env python3
"""Standalone probe: does bf16 `F.linear`'s accuracy AND speed on a given GPU
track `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`, and
does the split-K component account for it?

torch only, no other dependencies. Relative Frobenius error is measured against
an fp64 GEMM on the SAME values, so it isolates the GEMM's own reduction and
rounding rather than any difference in the inputs.

Context. Reduced-precision reduction is a documented, intentional trade: bf16
accumulation of split-K partials buys speed and costs accuracy, and PyTorch
exposes a flag (default: allowed). The question this probe asks is narrower and
is the only thing that would be worth reporting anywhere: are there shapes
where the default setting costs accuracy WITHOUT buying speed?

Settings compared, where the installed torch supports them:
  default            whatever the build ships (usually reduced-precision ON)
  redprec=on,  spk=off   reduced precision allowed, split-K heuristics barred
  redprec=off            reduced precision barred entirely
"""
import argparse
import statistics
import torch
import torch.nn.functional as F

# (label, M, K, N): out = F.linear(x[M,K], W[N,K]) -> [M,N]
# "A" rows are the shapes that read anomalously high error on an RTX 4090 in
# the dequant-on-forward leg; "C" rows are controls that did not, including one
# that shares N and K with an A row and differs only in M.
SHAPES = [
    ("A qwen3-gate_up ", 128, 2048, 1536),
    ("A qwen3-gate_up ", 738, 2048, 1536),
    ("A gemma4-gate_up", 128, 2816, 1408),
    ("A gemma4-gate_up", 738, 2816, 1408),
    ("A gptoss-down   ", 64, 2880, 2880),
    ("A gptoss-down   ", 369, 2880, 2880),
    ("A gptoss-gate_up", 369, 2880, 5760),
    ("C gptoss-gate_up", 64, 2880, 5760),
    ("C olmoe-gate_up ", 256, 2048, 2048),
    ("C qwen3-down    ", 128, 768, 2048),
]


def get_flags():
    m = torch.backends.cuda.matmul
    out = {}
    for a in ("allow_bf16_reduced_precision_reduction",
              "allow_bf16_reduced_precision_reduction_split_k"):
        try:
            out[a] = getattr(m, a)
        except Exception:
            out[a] = "unavailable"
    return out


def set_mode(mode):
    """Returns True if the mode is settable on this torch, False to skip it."""
    m = torch.backends.cuda.matmul
    try:
        if mode == "redprec_on_splitk_off":
            m.allow_bf16_reduced_precision_reduction = (True, False)
            # a torch that ignores the tuple form would silently do the wrong
            # thing, so verify the component actually took
            return getattr(m, "allow_bf16_reduced_precision_reduction_split_k",
                           None) is False
        if mode == "redprec_off":
            m.allow_bf16_reduced_precision_reduction = False
            return m.allow_bf16_reduced_precision_reduction is False
        m.allow_bf16_reduced_precision_reduction = True
        return True
    except Exception:
        return False


def rel_err(out, x, w):
    ref = x.to(torch.float64) @ w.to(torch.float64).t()
    return ((out.to(torch.float64) - ref).norm() / ref.norm()).item()


def timed(fn, iters):
    for _ in range(30):          # long enough to have clocks up on a consumer card
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"gpu   {torch.cuda.get_device_name(0)}  "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"flags at start: {get_flags()}")

    modes = ["default", "redprec_on_splitk_off", "redprec_off"]
    usable = []
    for mode in modes:
        ok = set_mode(mode)
        print(f"  mode {mode:24s} settable={ok}  flags={get_flags()}")
        if ok:
            usable.append(mode)
    set_mode("default")
    print()

    g = torch.Generator(device="cpu").manual_seed(0)
    hdr = f"{'shape':17} {'M':>5} {'K':>5} {'N':>5}"
    for mode in usable:
        hdr += f" | {mode[:12]:>12} err {'ms':>8}"
    print(hdr)

    rows = []
    for label, M, K, N in SHAPES:
        x = (torch.randn(M, K, generator=g, dtype=torch.float32) * 0.5).to(
            "cuda", torch.bfloat16)
        w = (torch.randn(N, K, generator=g, dtype=torch.float32) * 0.02).to(
            "cuda", torch.bfloat16)
        res = {}
        for mode in usable:
            set_mode(mode)
            torch.cuda.empty_cache()
            res[mode] = (rel_err(F.linear(x, w), x, w),
                         timed(lambda: F.linear(x, w), args.iters))
        line = f"{label:17} {M:5d} {K:5d} {N:5d}"
        for mode in usable:
            e, t = res[mode]
            line += f" | {e:16.4e} {t:8.3f}"
        print(line)
        rows.append((label, M, K, N, res))
        del x, w

    set_mode("default")
    if "redprec_off" in usable:
        print()
        print("default vs redprec_off  (err>1 = default less accurate; "
              "ms<1 = default faster, which is the trade being paid for)")
        for label, M, K, N, res in rows:
            e0, t0 = res["default"]
            e1, t1 = res["redprec_off"]
            verdict = ("costs accuracy, buys NO speed"
                       if e0 / e1 > 1.05 and t0 / t1 > 0.98 else "")
            print(f"  {label:17} {M:5d} {K:5d} {N:5d}  err {e0/e1:6.3f}  "
                  f"ms {t0/t1:6.3f}   {verdict}")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
