"""K16 bench (PREREG-k16-smallm-int4-gemm.md): K14's instrument over the registered config space. Usage: k16_bench.py <out.json>.
K14's instrument: CUDA-graph replay medians, the launch floor, and the bf16 dequant path as the comparator."""
import json, sys, torch, triton
from int4_pack_ref import dequant_int4_ref, pack_int4_b32
from int4_b32 import quant_x_rows, gemm_int4_b32_grouped_captured
from int4_smallm import gemm_int4_b32_smallm, smallm_workspace
from k14_bench import timed_replay, launch_floor

SHAPES = [("q_proj", 4096, 2048), ("k_proj", 512, 2048), ("v_proj", 512, 2048), ("o_proj", 2048, 4096)]
M = 16
dev = "cuda"
floor = launch_floor(dev)
print(f"launch floor {floor*1e3:.2f} us on {torch.cuda.get_device_name()}")
rows = []
for name, N, K in SHAPES:
    torch.manual_seed(0)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) / (K ** 0.5)
    packed, scales = pack_int4_b32(w.float().cpu())
    packed = packed.to(dev).contiguous(); scales = scales.to(dev).contiguous()
    w_deq = dequant_int4_ref(packed.cpu(), scales.cpu(), N, K).to(torch.bfloat16).to(dev)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    ref = (x.float() @ w_deq.float().t())
    out = {}
    def a_bf16(): return x @ w_deq.t()
    out["bf16"] = timed_replay(a_bf16, iters=200)
    # the shipped M-tile kernel at K14's best swept config (bn16/w2) on a one-expert tile table
    p3 = packed.reshape(1, N, K // 2); s3 = scales.reshape(1, N, K // 32)
    t_row0 = torch.tensor([0], dtype=torch.int32, device=dev); t_rows = torch.tensor([M], dtype=torch.int32, device=dev); t_group = torch.tensor([0], dtype=torch.int32, device=dev)
    def a_gemm():
        xq, xs = quant_x_rows(x)
        return gemm_int4_b32_grouped_captured(xq, xs, p3, s3, t_row0, t_rows, t_group, block_m=16, block_n=16, warps=2)
    try: out["k14_gemm_bn16_w2"] = timed_replay(a_gemm, iters=100)
    except Exception as e: out["k14_gemm_bn16_w2"] = f"error: {type(e).__name__}"
    best = None
    for bn in (32, 64, 128):
        for kc in (128, 256):
            for sk in (1, 2, 4, 8):
                if (K // kc) % sk: continue
                for warps in (4, 8):
                    for stages in (2, 3):
                        ws = smallm_workspace(N, block_n=bn, sk=sk, device=dev)
                        def a_k16(bn=bn, kc=kc, sk=sk, warps=warps, stages=stages, ws=ws):
                            return gemm_int4_b32_smallm(x, packed, scales, block_n=bn, kc=kc, sk=sk, warps=warps, stages=stages, workspace=ws)
                        tag = f"k16_bn{bn}_kc{kc}_sk{sk}_w{warps}_s{stages}"
                        try:
                            y = a_k16(); err = float((y.float() - ref).abs().max() / ref.abs().max())
                            ms = timed_replay(a_k16, iters=100)
                            out[tag] = ms
                            if err > 2 ** -7: out[tag] = f"NUMERICS {err:.3e}"
                            elif best is None or ms < best[1]: best = (tag, ms, err)
                        except Exception as e:
                            out[tag] = f"error: {type(e).__name__}: {str(e)[:60]}"
    row = {"shape": name, "N": N, "K": K, "M": M, "bf16_us": out["bf16"] * 1e3, "k14_us": out["k14_gemm_bn16_w2"] * 1e3 if isinstance(out["k14_gemm_bn16_w2"], float) else out["k14_gemm_bn16_w2"],
           "k16_best": best[0] if best else None, "k16_best_us": best[1] * 1e3 if best else None, "k16_err_rel": best[2] if best else None,
           "k16_all": {k: (v * 1e3 if isinstance(v, float) else v) for k, v in out.items() if k.startswith("k16")}}
    rows.append(row)
    k16s = f"{row['k16_best_us']:.2f} us ({row['k16_best']})" if best else "none"
    k14s = f"{row['k14_us']:.2f} us" if isinstance(row["k14_us"], float) else str(row["k14_us"])
    print(f"{name:7s} N{N} K{K}: bf16 {row['bf16_us']:.2f} us | k14 gemm {k14s} | K16 best {k16s} | ratio bf16/K16 {row['bf16_us']/row['k16_best_us'] if best else float('nan'):.3f}", flush=True)
json.dump({"device": torch.cuda.get_device_name(), "launch_floor_us": floor * 1e3, "rows": rows}, open(sys.argv[1] if len(sys.argv) > 1 else "k16_a2000.json", "w"), indent=1)
