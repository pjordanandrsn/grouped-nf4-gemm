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
    "nvme_arena", "nvme_bake_nf4", "nvme_reader", "arena_experts", "arena_moe_patch",
    "nf4_qlora",
    "cpu_grouped", "gnf4_native",
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

    # NVMe arena: the distribution claim is "bake locally, verify against a
    # published manifest", so the wheel must actually be able to DO that --
    # an import alone would not catch a broken CLI or a missing dependency.
    r = subprocess.run([sys.executable, "-m", "nvme_arena", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "verify" in r.stdout, r.stderr[-400:]
    print("nvme_arena CLI (bake/bake-experts/verify) OK")

    from nvme_arena import load_index, row_offset  # noqa: F401
    from nvme_reader import alloc_landing, check_aligned
    mv, keep = alloc_landing(1 << 16)          # mmap-backed, page-aligned
    check_aligned(mv, 4096)                    # raises if O_DIRECT would EINVAL
    try:
        check_aligned(mv[1:], 4096)            # deliberately off by one byte
        raise AssertionError("check_aligned accepted a misaligned buffer")
    except ValueError:
        pass
    del mv, keep
    print("nvme_reader landing alloc + O_DIRECT alignment guard OK")

    # the differentiable training wrapper must be importable from the wheel:
    # without it, training silently falls back off the fused kernel
    from nf4_qlora import (FusedGroupedNf4, gemm_4bit_grouped_train,  # noqa: F401
                           fused_grouped_lora, lora_delta_grouped)
    import torch as _t
    _d = lora_delta_grouped(_t.zeros(2, 8), _t.zeros(1, 4, 8), _t.zeros(1, 6, 4), [2], [0])
    assert _d.shape == (2, 6) and _t.count_nonzero(_d) == 0
    print("nf4_qlora training wrapper OK (zero-B delta is exactly zero)")

    # arena_experts: the arena -> kernel link. Import-only would pass even if
    # the K3 spelling map were empty, so assert the mapping the caller relies on.
    from arena_experts import K3_KINDS, K3_PROJ, expert_bytes_per_token
    assert K3_PROJ == {"gate": "w1", "up": "w3", "down": "w2"}, K3_PROJ
    assert len(K3_KINDS) == 6 and all("." in k for k in K3_KINDS), K3_KINDS
    _idx = {"segments": [{"suffix": "w1.weight_packed", "length": 100}],
            "rows": [[1, 0, 0], [1, 1, 4096], [2, 0, 8192]]}
    assert expert_bytes_per_token(_idx, 4) == 4 * 2 * 100
    print("arena_experts surface (K3 map + bytes/token) OK")

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
