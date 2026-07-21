"""Built-artifact smoke: run against the INSTALLED wheel (never the repo tree).

The import battery below mirrors the README's Install-comment surface — the
invariant this file enforces is: README's promised surface == the wheel's
importable surface. If you change one, change both.

CPU-safe micros only: pack->dequant_ref roundtrip, verify_provenance -h,
moonshot_gather.discover_layer on a synthetic in-memory weight_map.
"""
import subprocess
import sys

# README "## Install" comment, verbatim surface (plus the run/gate harnesses
# the wheel also ships):
BATTERY = [
    "nf4_grouped", "nf4_pack_ref", "host_gather",
    "mxfp4_pack_ref", "mxfp4_grouped", "mxfp4_loader", "mxfp4_pipelined",
    "mxfp4_qlora", "mxfp4_native_load", "moonshot_gather", "verify_provenance",
    "run_mxfp4_20b_qlora", "gate_native_load_20b",
]

def main() -> int:
    import importlib
    for name in BATTERY:
        importlib.import_module(name)
    print(f"import battery: {len(BATTERY)}/{len(BATTERY)} OK")

    import torch
    from nf4_grouped import dequant_ref  # the README's promised symbol pair
    from nf4_grouped import gemm_4bit_grouped  # noqa: F401  (import only; CUDA to run)
    from nf4_pack_ref import quantize_pack_nf4
    w = torch.randn(4, 128)
    packed, absmax = quantize_pack_nf4(w)
    back = dequant_ref(packed, absmax, 4, 128)
    relerr = ((back - w).norm() / w.norm()).item()
    assert relerr < 0.25, f"pack->dequant_ref roundtrip relerr {relerr}"
    print(f"dequant_ref roundtrip OK (relerr {relerr:.4f})")

    r = subprocess.run([sys.executable, "-m", "verify_provenance", "-h"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "artifact" in r.stdout, r.stderr[-400:]
    print("verify_provenance -h OK")

    from moonshot_gather import discover_layer
    wm = {}
    for e in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            wm[f"model.layers.0.mlp.experts.{e}.{proj}.weight"] = "shard-0.safetensors"
            wm[f"model.layers.0.mlp.experts.{e}.{proj}.weight_scale_inv"] = "shard-0.safetensors"
    d = discover_layer(wm, 0)
    assert d["n_experts"] == 2, d
    print(f"discover_layer synthetic OK ({d['n_experts']} experts, scale={d.get('scale_suffix')})")
    print("WHEEL SMOKE: ALL GREEN")
    return 0

if __name__ == "__main__":
    sys.exit(main())
