# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Real-bytes gate for the GGUF dX bar: the same floor and fault scores as calibrate.py, on sha-pinned released GGUF
tensors (gnf4 scripts/fetch_gguf_fixtures.py), with the dmin/d the files actually ship. CPU only.

Per case it reports the realized bound min(2 * e_ctrl, 1e-2) and each fault's margin over it (fault / bound).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import gguf.quants as gq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernel"))
import kquant_ref as kq  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import dx_bf16, m1_scale_spill, m5_dmin_zero, m6_signed_as_unsigned, rel  # noqa: E402

ORACLE = {kq.GGML_Q2_K: gq.Q2_K, kq.GGML_Q3_K: gq.Q3_K, kq.GGML_Q4_K: gq.Q4_K, kq.GGML_Q5_K: gq.Q5_K,
          kq.GGML_Q6_K: gq.Q6_K, kq.GGML_Q8_0: gq.Q8_0}
IN_SCOPE = {kq.GGML_Q8_0, kq.GGML_Q4_K, kq.GGML_Q5_K, kq.GGML_Q6_K}
B_ABS = 1e-2
FP16_COLS = {kq.GGML_Q8_0: [(0, 2)], kq.GGML_Q2_K: [(80, 82), (82, 84)], kq.GGML_Q3_K: [(108, 110)],
             kq.GGML_Q4_K: [(0, 2), (2, 4)], kq.GGML_Q5_K: [(0, 2), (2, 4)], kq.GGML_Q6_K: [(208, 210)]}


def scales_bf16(gtype, b):
    """CALIBRATION, not a fault: every fp16 scale field rounded to bf16 precision, then stored back as fp16."""
    b2 = b.copy()
    for lo, hi in FP16_COLS[gtype]:
        f = torch.from_numpy(b2[:, lo:hi].copy().view(np.float16).astype(np.float32)).to(torch.bfloat16).float()
        b2[:, lo:hi] = f.numpy().astype(np.float16).view(np.uint8).reshape(-1, 2)
    return b2


def dmin_over_d(gtype, b):
    """Median |dmin| / |d| over blocks (Q4_K/Q5_K: d at 0:2, dmin at 2:4), the quantity the dmin fault scales with."""
    if gtype not in (kq.GGML_Q4_K, kq.GGML_Q5_K):
        return None
    d = np.abs(b[:, 0:2].copy().view(np.float16).astype(np.float64)).ravel()
    dm = np.abs(b[:, 2:4].copy().view(np.float16).astype(np.float64)).ravel()
    ok = d > 0
    return float(np.median(dm[ok] / d[ok])) if ok.any() else None


def main(fx_dir, out_path):
    fx = Path(fx_dir)
    manifest = json.loads((fx / "manifest.json").read_text())
    rows = []
    for e in manifest["entries"]:
        gtype = e["ggml_type"]
        if gtype not in ORACLE:
            continue
        raw = (fx / e["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"], f"{e['file']}: sha256 mismatch"
        N, K = e["shape"]
        elems, bbytes, _ = kq.GGML_DEQUANT[gtype]
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bbytes).copy()
        w64 = torch.from_numpy(ORACLE[gtype].dequantize_blocks(b).astype(np.float64)).reshape(N, K)
        w32 = kq.dequantize_ggml(gtype, raw, (N, K))
        assert torch.equal(w32.double(), w64), f"{e['tensor']}: decoder disagrees with gguf-py"
        for M in (17, 128):
            for seed in range(3):
                g = torch.Generator().manual_seed(gtype * 1000 + M + seed)
                dy = torch.randn(M, N, generator=g).to(torch.bfloat16)
                ref = dy.double() @ w64
                ctrl = rel(dx_bf16(dy, w32), ref)
                bound = min(2 * ctrl, B_ABS)
                faults = {"zero_dX": 1.0, "m4_row_shift": rel(dx_bf16(dy, torch.roll(w32, 1, 0)), ref)}
                if gtype in (kq.GGML_Q4_K, kq.GGML_Q5_K):
                    faults["m1_scale_spill"] = rel(dx_bf16(dy, m1_scale_spill(gtype, b, N, K)), ref)
                    faults["m5_dmin_zero"] = rel(dx_bf16(dy, m5_dmin_zero(gtype, b, N, K)), ref)
                if gtype in (kq.GGML_Q8_0, kq.GGML_Q6_K):
                    faults["m6_signed_as_unsigned"] = rel(dx_bf16(dy, m6_signed_as_unsigned(gtype, b, N, K)), ref)
                calib = rel(dx_bf16(dy, kq.dequantize_ggml(gtype, scales_bf16(gtype, b).tobytes(), (N, K))), ref)
                s = min(N, K)                                   # m3 needs N == K: the leading square block
                ref_s = dy[:, :s].double() @ w64[:s, :s]
                faults["m3_transpose_square"] = rel(dx_bf16(dy[:, :s], w32[:s, :s].T.contiguous()), ref_s)
                ctrl_s = rel(dx_bf16(dy[:, :s], w32[:s, :s]), ref_s)
                rows.append({"source": e["source"], "tensor": e["tensor"], "type": e["type_name"],
                             "in_scope": gtype in IN_SCOPE, "N": N, "K": K, "M": M, "seed": seed,
                             "partial": e.get("partial"), "slice": e.get("slice", ""), "dmin_over_d_median": dmin_over_d(gtype, b),
                             "control": ctrl, "bound": bound, "calib_scales_bf16": calib, "control_square": ctrl_s, "faults": faults,
                             "margin": {f: v / (min(2 * ctrl_s, B_ABS) if f == "m3_transpose_square" else bound)
                                        for f, v in faults.items()}})
    json.dump(rows, open(out_path, "w"), indent=1)
    for key in sorted({(r["type"], r["source"], r["tensor"] + r.get("slice", "")) for r in rows}):
        rs = [r for r in rows if (r["type"], r["source"], r["tensor"] + r.get("slice", "")) == key]
        r0 = rs[0]
        print(f"{key[0]:5} {key[1]:17} {key[2]:26} N={r0['N']:5} K={r0['K']:5} dmin/d={r0['dmin_over_d_median']}"
              f" ctrl max {max(r['control'] for r in rs):.2e} bound max {max(r['bound'] for r in rs):.2e}"
              f" scales->bf16 {min(r['calib_scales_bf16'] for r in rs):.1e}..{max(r['calib_scales_bf16'] for r in rs):.1e} |",
              " ".join(f"{f} {min(r['faults'][f] for r in rs):.2e} (x{min(r['margin'][f] for r in rs):.1f})"
                       for f in sorted(r0["faults"]) if f != "zero_dX"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
