"""MXFP4 decode GEMV: does split-K vs row count R behave like the int4 kernel's?

gnf4#357 threaded R into `int4_b32._plan` and left `mxfp4_grouped.gemv_mxfp4_b32`
on the N-only plan, saying so: same grid `(cdiv(N,128), R, sk)`, same partials
reduce, but an e2m1 inner loop the int4 sweep never measured. This is that
measurement, same rules as bench/int4/sk_sweep.py -- graph replay, full cost
including the reduce, every sk to the span cap plus the planned one, ONE SHAPE
PER PROCESS (the graph-pool fault), R = 1..128.

Shapes: gpt-oss-20b (E=32, H=2880, I=2880, top_k=4) is the MXFP4 family served
natively; qwen3_moe's shapes are added so the e2m1 loop can be compared with the
int4 sweep like-for-like.
Usage: python mx_sweep.py <family> <proj>   -> rows/mx_<family>_<proj>.json
"""
import json, os, pathlib, sys
import torch, triton
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "kernel"))
from int4_b32 import _plan, reduce_partials            # noqa: E402  (plan shared with mxfp4)
from mxfp4_grouped import _gemv_mxfp4_b32              # noqa: E402
from mxfp4_pack_ref import quantize_pack_mxfp4         # noqa: E402

DEV = "cuda"
FAMILIES = {"gpt_oss_20b": (2880, 2880, 32), "qwen3_moe": (2048, 768, 128)}
RS = [int(x) for x in os.environ.get("RS", "1,2,4,8,16,32,64,128").split(",")]
SKS = [int(x) for x in os.environ.get("SKS", "1,2,3,4,6,8,16").split(",")]


def timed_replay(fn, iters=60, warmup=12):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    name, proj = sys.argv[1], sys.argv[2]
    H, I, E = FAMILIES[name]
    N, K = (2 * I, H) if proj == "gate_up" else (H, I)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    gpu = torch.cuda.get_device_name(0)
    torch.manual_seed(0)
    # pack on the CPU, one expert at a time: quantize_pack_mxfp4 materialises an int64
    # [N, K//32, 32] code tensor (8 B/weight), which for E=128 x 2048 x 768 is 12 GB on
    # the GPU -- the first launch OOM'd every shape on a card that also hosts other
    # containers. Only the u8 outputs go to the device.
    bl, sc = [], []
    for e in range(E):
        w = torch.randn(N, K, dtype=torch.float32) * 0.5
        b, c = quantize_pack_mxfp4(w)                      # [N,K//32,16] u8, [N,K//32] u8
        bl.append(b.reshape(N, K // 2)); sc.append(c)
    blocks = torch.stack(bl).contiguous().to(DEV)            # the kernel's [E, N, K//2]
    scales = torch.stack(sc).contiguous().to(DEV)
    del bl, sc
    bn, wp, sk_plan, ku = _plan(N, K)
    kb = K // 32; sk_cap = max(1, kb // ku); tiles = triton.cdiv(N, bn)
    sks = sorted({s for s in SKS + [sk_plan] if s <= sk_cap})
    print(f"## mxfp4 {name} {proj}  N={N} K={K} E={E}  plan sk={sk_plan} (cap {sk_cap}), tiles={tiles}, {gpu} {sms} SMs", flush=True)
    rows = []
    for R in RS:
        xq = torch.randint(-127, 127, (R, K), dtype=torch.int8, device=DEV)
        xs = torch.rand(R, K // 32, dtype=torch.float32, device=DEV) * 0.01   # per-(row, 32-block) scales
        eids = torch.randint(0, E, (R,), dtype=torch.int32, device=DEV)
        times = {}
        for sk in sks:
            part = torch.empty(sk * R, N, dtype=torch.float32, device=DEV)
            dst = torch.empty(R, N, dtype=torch.bfloat16, device=DEV)
            def call(sk=sk, part=part, dst=dst):
                _gemv_mxfp4_b32[(tiles, R, sk)](xq, xs, blocks, scales, eids, part,
                                                N, K=K, R=R, BLOCK_N=bn, SK=sk, KU=ku, num_warps=wp)
                if sk > 1: reduce_partials(part, sk, R, N, out=dst)
            try:
                times[sk] = timed_replay(call)
            except Exception as e:                                  # noqa: BLE001
                print(f"    sk={sk} R={R} FAILED {type(e).__name__}: {e}", flush=True); raise
        best = min(times, key=times.get); pen = times[sk_plan] / times[best]
        print(f"   R={R:4d} | " + " ".join(f"sk{s}={times[s]*1000:.1f}" for s in sks) + f" us | BEST sk={best}"
              + ("  = plan" if best == sk_plan else f"  (plan sk={sk_plan} costs x{pen:.3f})"), flush=True)
        rows.append({"family": name, "proj": proj, "format": "mxfp4", "N": N, "K": K, "E": E, "R": R,
                     "sk_plan": sk_plan, "sk_best": best, "sk_cap": sk_cap, "tiles": tiles, "sms": sms,
                     "gpu": gpu, "ms": {str(k): v for k, v in times.items()}, "plan_penalty": pen})
    out = pathlib.Path(__file__).resolve().parent / "rows"; out.mkdir(exist_ok=True)
    json.dump({"gpu": gpu, "sms": sms, "torch": torch.__version__, "rows": rows},
              (out / f"mx_{name}_{proj}.json").open("w"), indent=1)


if __name__ == "__main__":
    main()
