# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""G7 bandwidth clause: FP8 paged decode attention, achieved GB/s.

Reports what the kernel actually moves per launch — packed K+V payload
plus scale tails for every resident token — divided by measured kernel
time, alongside the fraction of the box's ``B_vram`` (from the Phase-0
calibration blob when given, else reported as raw GB/s only).

The comparison arm is bf16 SDPA with ``enable_gqa=True`` on identical
dequantized tensors — the honest baseline, stated loudly because the
in-tree record contains a 19x-wrong claim produced by comparing a fused
KV kernel against SDPA fed a 16x replicated cache. SDPA reads 2x the
bytes (bf16 vs fp8+scales), so wall-time parity means the fp8 kernel is
running at roughly half SDPA's efficiency; the gate is the roofline
fraction, not beating SDPA's clock.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def _build_layer(B, hkv, d, T, k_groups, seed=0):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
    from fp8_kv import kv_block_bytes, pack_kv_block, quantize_kv_fp8
    g = torch.Generator().manual_seed(seed)
    BT = 16
    n_blk = (T + BT - 1) // BT
    k_row = kv_block_bytes(BT, hkv, d) + BT * hkv * 4 * (k_groups - 1)
    v_row = kv_block_bytes(BT, hkv, d)
    k_pool = torch.zeros(B * n_blk * k_row, dtype=torch.uint8)
    v_pool = torch.zeros(B * n_blk * v_row, dtype=torch.uint8)
    table = torch.zeros(B, n_blk, dtype=torch.int32)
    k_ref = torch.randn(B, T, hkv, d, generator=g) * 1.5
    v_ref = torch.randn(B, T, hkv, d, generator=g)
    row = 0
    for b in range(B):
        for i in range(n_blk):
            table[b, i] = row
            qk, sk = quantize_kv_fp8(k_ref[b, i * BT:(i + 1) * BT],
                                     group=d // k_groups)
            pack_kv_block(qk, sk, k_pool[row * k_row:(row + 1) * k_row])
            qv, sv = quantize_kv_fp8(v_ref[b, i * BT:(i + 1) * BT])
            pack_kv_block(qv, sv, v_pool[row * v_row:(row + 1) * v_row])
            row += 1
    return (k_pool, v_pool, table, k_ref, v_ref,
            B * n_blk * (k_row + v_row))


def _time_cuda(fn, iters):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main(shapes, T, iters, k_groups, calib, out_dir):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
    from fp8_paged_attn import fp8_paged_decode_attention, paged_attn_ref

    b_vram = None
    if calib:
        blob = json.loads(Path(calib).read_text())
        b_vram = blob.get("b_vram_gbps") or blob.get("gpus", [{}])[0].get(
            "b_vram_gbps")

    results = []
    for hq, hkv, d in shapes:
        G = hq // hkv
        for B in (1, 8, 16, 25, 32):
            kp, vp, tab, k_ref, v_ref, kv_bytes = _build_layer(
                B, hkv, d, T, k_groups, seed=B)
            q = (torch.randn(B, hq, d) * 0.5).to(torch.bfloat16).cuda()
            kp, vp, tab = kp.cuda(), vp.cuda(), tab.cuda()
            lens = torch.full((B,), T, dtype=torch.int32).cuda()

            out = fp8_paged_decode_attention(
                q, kp, vp, tab, lens, n_kv_heads=hkv, head_dim=d,
                k_groups=k_groups)
            want = paged_attn_ref(q.cpu(), kp.cpu(), vp.cpu(), tab.cpu(),
                                  lens.cpu(), n_kv_heads=hkv, head_dim=d,
                                  k_groups=k_groups)
            torch.testing.assert_close(out.cpu().float(), want.float(),
                                       rtol=2e-2, atol=2e-2)

            t_kernel = _time_cuda(
                lambda: fp8_paged_decode_attention(
                    q, kp, vp, tab, lens, n_kv_heads=hkv, head_dim=d,
                    k_groups=k_groups), iters)

            # honest baseline: bf16 SDPA, enable_gqa, contiguous cache
            kb = k_ref.permute(0, 2, 1, 3).contiguous().to(
                torch.bfloat16).cuda()
            vb = v_ref.permute(0, 2, 1, 3).contiguous().to(
                torch.bfloat16).cuda()
            q4 = q[:, :, None].reshape(B, hq, 1, d)
            t_sdpa = _time_cuda(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q4, kb, vb, enable_gqa=True), iters)
            sdpa_bytes = 2 * B * T * hkv * d * 2

            gbps = kv_bytes / t_kernel / 1e9
            row = {"hq": hq, "hkv": hkv, "d": d, "B": B, "T": T,
                   "kv_bytes": kv_bytes,
                   "t_kernel_us": t_kernel * 1e6, "gbps": gbps,
                   "t_sdpa_us": t_sdpa * 1e6,
                   "sdpa_gbps": sdpa_bytes / t_sdpa / 1e9,
                   "wall_ratio_vs_sdpa": t_kernel / t_sdpa}
            if b_vram:
                row["frac_of_b_vram"] = gbps / b_vram
            results.append(row)
            frac = f" frac={row.get('frac_of_b_vram', float('nan')):.3f}" \
                if b_vram else ""
            print(f"BENCH hq={hq} hkv={hkv} d={d} B={B:2d} T={T} "
                  f"kernel={t_kernel*1e6:7.1f}us {gbps:7.1f} GB/s{frac} | "
                  f"sdpa={t_sdpa*1e6:7.1f}us "
                  f"({sdpa_bytes/t_sdpa/1e9:7.1f} GB/s) "
                  f"wall x{t_kernel/t_sdpa:.2f}", flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "fp8_attn_bench.json").write_text(
        json.dumps({"T": T, "k_groups": k_groups, "b_vram": b_vram,
                    "rows": results}, indent=2))
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--k-groups", type=int, default=4)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    shapes = [(64, 4, 128), (32, 4, 128)]
    main(shapes, a.t, a.iters, a.k_groups, a.calib, a.out)
