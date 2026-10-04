"""Host time of ONE e4b fused MoE layer (ExpertsLoRA + enable_fast_train(dgrad=True)) at Qwen3-30B-A3B's per-layer shape.

No profiler. Each phase starts with the GPU idle (synchronize first), so the forward's one routing sync waits only for the
routing kernels and the per-phase wall time is ~host work. Median of --reps. --cprofile: top functions by own time instead.
--check: also dump the forward output and every gradient to --check for bitwise comparison across variants.
"""
import argparse, cProfile, io, pstats, time
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=380); ap.add_argument("--E", type=int, default=128)
ap.add_argument("--hid", type=int, default=2048); ap.add_argument("--inter", type=int, default=768)
ap.add_argument("--topk", type=int, default=8); ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=16)
ap.add_argument("--reps", type=int, default=200); ap.add_argument("--cprofile", action="store_true"); ap.add_argument("--check", default=None); ap.add_argument("--deterministic", action="store_true")
a = ap.parse_args()
if a.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
dev = "cuda"
from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_fast_train
torch.manual_seed(0)
gu = torch.randn(a.E, 2 * a.inter, a.hid, dtype=torch.bfloat16, device=dev) * 0.02
dn = torch.randn(a.E, a.hid, a.inter, dtype=torch.bfloat16, device=dev) * 0.02
base = Experts4bit.from_float(gu, dn, quant_type="nf4", compute_dtype=torch.bfloat16)
del gu, dn; torch.cuda.empty_cache()
mod = ExpertsLoRA(base, r=a.r, alpha=a.alpha, dtype=torch.bfloat16).to(dev).train()
with torch.no_grad():
    for p in (mod.gate_up_lora_B, mod.down_lora_B):
        p.normal_(0, 0.02)
assert enable_fast_train(mod, dgrad=True) == 1
skew = torch.linspace(1.5, -1.5, a.E, device=dev)          # a router with hot and cold experts, like a real one
def inputs(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    hs = torch.randn(a.tokens, a.hid, dtype=torch.bfloat16, device=dev, generator=g).requires_grad_(True)
    logits = torch.randn(a.tokens, a.E, device=dev, generator=g) + skew
    w, idx = logits.softmax(-1).topk(a.topk, dim=1)
    return hs, idx, (w / w.sum(-1, keepdim=True)).to(torch.bfloat16).requires_grad_(True)   # router weights carry grad in training
def run(seed):
    hs, idx, wts = inputs(seed)
    out = mod(hs, idx, wts); out.float().sum().backward()
    return hs, wts, out
for i in range(5):
    run(i)
torch.cuda.synchronize()
if a.check:
    for p in mod.parameters(): p.grad = None
    hs, wts, out = run(12345); torch.cuda.synchronize()
    torch.save({"out": out.detach().cpu(), "hs_grad": hs.grad.cpu(), "w_grad": wts.grad.cpu(),
                **{n: p.grad.cpu() for n, p in mod.named_parameters() if p.grad is not None}}, a.check)
    print("check written", a.check)
if a.cprofile:
    pr = cProfile.Profile()
    for i in range(a.reps):
        hs, idx, wts = inputs(1000 + i); torch.cuda.synchronize()
        pr.enable(); out = mod(hs, idx, wts); out.float().sum().backward(); pr.disable()
        torch.cuda.synchronize()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30); print(s.getvalue()[:9000])
    raise SystemExit
fw, bw, fd, bd = [], [], [], []
for i in range(a.reps):
    hs, idx, wts = inputs(2000 + i)
    for p in mod.parameters(): p.grad = None
    torch.cuda.synchronize()
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    t0 = time.perf_counter(); e0.record(); out = mod(hs, idx, wts); loss = out.float().sum(); e1.record(); t1 = time.perf_counter()
    torch.cuda.synchronize(); t1b = time.perf_counter()
    loss.backward(); e2.record(); t2 = time.perf_counter(); torch.cuda.synchronize()
    fw.append((t1 - t0) * 1e3); bw.append((t2 - t1b) * 1e3); fd.append(e0.elapsed_time(e1)); bd.append(e1.elapsed_time(e2))
q = lambda v, f: sorted(v)[int(len(v) * f)]
print("fwd host ms p25/p50/p75 %.3f %.3f %.3f | bwd host ms %.3f %.3f %.3f | fwd dev span %.3f bwd %.3f" % (
    q(fw, .25), q(fw, .5), q(fw, .75), q(bw, .25), q(bw, .5), q(bw, .75), q(fd, .5), q(bd, .5)))
try:
    import nf4_grouped as _ng
    print("host_reuse", getattr(_ng, "HOST_REUSE_STATS", None))
except Exception:
    pass

