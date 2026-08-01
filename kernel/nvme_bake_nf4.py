# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""NF4 quantize-at-bake: bf16 per-expert checkpoints (Qwen3-235B class) ->
the expert-major NVMe arena, quantizing once at bake time.

Provenance here is TWO-HOP, honestly labeled: the arena bytes are not the
shipped bytes (quantization is a transform), so each segment records
  * sha256 of the ARENA bytes (self-integrity — what `verify` re-checks), and
  * the exact SOURCE tensors consumed: file, absolute byte range, sha256 of
    the shipped bf16 bytes (what `verify --against-source` re-checks), plus
  * the quantizer record (bnb/torch versions, blocksize, quant_type, GPU).
The claim this supports: *"these rows were produced from exactly these
shipped bytes by exactly this quantizer"* — weaker than relocation's
bit-identity, and the manifest says which claim it is making
(`bake_mode: nf4-quantize`).

Numerics: `quantize_expert` mirrors the flagship harness
(bench/phase3/offload_generate_235b.py) verbatim — cat[gate;up] quantized
as one [2I, H] matrix, blocksize 64, nested absmax dequantized to fp32 —
so arena rows are byte-compatible with the pinned stacks the flagship
engine builds at load. Row layout: gu_blocks u8 [2I,H/2] · gu_absmax f32
[2I,H/64] · dn_blocks u8 [H,I/2] · dn_absmax f32 [H,I/64], segments
8-aligned, row padded to the device block (same discipline as relocation).

Requires torch + bitsandbytes + CUDA (quantize runs on GPU). The
relocation bake (`nvme_arena.bake`) stays torch-free; the reader/verifier/
tier consume either arena identically. `quantize_fn` is injectable so the
geometry/manifest path is CPU-testable with a mock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mxfp4_loader import _read_st_header  # noqa: E402
from nvme_arena import MAGIC, VERSION, _align, _align8  # noqa: E402

PROJ = ("gate_proj", "up_proj", "down_proj")
# Per-expert MXFP4 checkpoints (Kimi K3, DeepSeek-V4) spell the same three
# projections w1/w3/w2 -- gate/up/down, confirmed by SHAPES not convention --
# and carry a companion scale tensor per projection instead of a bf16 weight.
PROJ_W123 = ("w1", "w3", "w2")


def _expert_names(layer, e, prefix="model.layers", proj=PROJ, suffix="weight",
                  moe="mlp"):
    # `moe` is the attribute the experts hang off. transformers-canonical
    # checkpoints say `mlp`; a checkpoint published in DeepSeek's own reference
    # spelling says `ffn` and drops the `model.` prefix entirely.
    base = f"{prefix}.{layer}.{moe}.experts.{e}."
    return {p: f"{base}{p}.{suffix}" for p in proj}


def default_quantize_expert(dev="cuda"):
    """The flagship quantize, verbatim: bf16 [N,K] -> (packed u8 [N,K/2],
    absmax f32 [N,K/64]), returned as CPU tensors."""
    import torch
    from bitsandbytes import functional as F

    def q(w_bf16):
        w = w_bf16.to(dev)
        qt, st = F.quantize_4bit(w, blocksize=64, quant_type="nf4")
        N, K = w.shape
        packed = qt.reshape(N, K // 2)
        am = st.absmax
        if getattr(st, "nested", False):
            am = F.dequantize_blockwise(st.absmax, st.state2) + st.offset
        absmax = am.to(torch.float32).reshape(N, K // 64)
        out = (packed.cpu(), absmax.cpu())
        del w, qt, st, am
        return out
    return q


class _Shards:
    def __init__(self, snapshot):
        import torch
        self.torch = torch
        self.snapshot = snapshot
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wm = json.load(open(idx))["weight_map"]
        else:
            # Unsharded snapshot: no index is emitted for a single-file
            # checkpoint, so synthesize the map from the file's own key list.
            single = os.path.join(snapshot, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(
                    f"{snapshot}: neither model.safetensors.index.json (sharded) "
                    "nor model.safetensors (single-file)")
            hdr, _ = _read_st_header(single)
            self.wm = {k: "model.safetensors" for k in hdr
                       if k != "__metadata__"}
        self._hdrs = {}

    def locate(self, name):
        shard = self.wm[name]
        path = os.path.join(self.snapshot, shard)
        if shard not in self._hdrs:
            self._hdrs[shard] = _read_st_header(path)
        hdr, data_start = self._hdrs[shard]
        lo, hi = hdr[name]["data_offsets"]
        return path, data_start + lo, data_start + hi, hdr[name]["shape"], \
            hdr[name]["dtype"]

    def _raw(self, name):
        path, lo, hi, shape, dtype = self.locate(name)
        with open(path, "rb") as f:
            f.seek(lo)
            raw = f.read(hi - lo)
        return raw, shape, dtype, (os.path.basename(path), lo, hi,
                                   hashlib.sha256(raw).hexdigest())

    def read_mxfp4(self, base, weight_suffix=".weight", scale_suffix=".scale"):
        """`base` is the projection stem; reads the packed-nibble weight and its
        e8m0 scale and returns the dequantized bf16 matrix.

        **The suffixes differ by checkpoint and are not guessable from the stem.**
        DeepSeek-V4 spells them `.weight` / `.scale` (the defaults); Kimi K3 spells
        them `.weight_packed` / `.weight_scale` — compare `V4_RESIDENCY_KINDS` against
        `K3_RESIDENCY_KINDS` in `mxfp4_residency`. This used to hardcode V4's pair while
        the docstring named K3, so a K3 bake through `source="mxfp4"` looked supported
        and could not resolve a single tensor.

        The DTYPE labels differ too — V4 says `I8`/`F8_E8M0`, K3 says `U8` for both —
        but the bytes are identical, so they are read as raw uint8 rather than through
        torch's dtype table, which also means this works on a torch too old to name
        `float8_e8m0fnu` at all.
        """
        from mxfp4_pack_ref import dequant_mxfp4
        torch = self.torch
        b_raw, b_shape, _, b_src = self._raw(base + weight_suffix)
        s_raw, s_shape, _, s_src = self._raw(base + scale_suffix)
        rows, kh = b_shape
        groups = s_shape[1]
        blocks = torch.frombuffer(bytearray(b_raw), dtype=torch.uint8).reshape(
            rows, groups, kh // groups)
        scales = torch.frombuffer(bytearray(s_raw), dtype=torch.uint8).reshape(s_shape)
        w = dequant_mxfp4(blocks, scales, dtype=torch.float32).to(torch.bfloat16)
        return w, [b_src, s_src]

    def read_bf16(self, name):
        path, lo, hi, shape, dtype = self.locate(name)
        assert dtype == "BF16", (name, dtype)
        with open(path, "rb") as f:
            f.seek(lo)
            raw = f.read(hi - lo)
        t = self.torch.frombuffer(bytearray(raw), dtype=self.torch.bfloat16)
        return t.reshape(shape), (os.path.basename(path), lo, hi,
                                  hashlib.sha256(raw).hexdigest())


def bake_nf4(snapshot, out, *, layers=None, prefix="model.layers",
             align=4096, limit_experts=0, quantize_fn=None, log=print,
             proj=PROJ, source="bf16", moe="mlp",
             mxfp4_suffixes=(".weight", ".scale")):
    """Quantize-bake. Discovers (L, E) from the checkpoint index; emits the
    same arena/index/manifest triple as the relocation bake, with the
    two-hop provenance schema."""
    if source not in ("bf16", "mxfp4"):
        raise ValueError(f"source must be 'bf16' or 'mxfp4'; got {source!r}")
    sh = _Shards(snapshot)
    gate_key = f"{proj[0]}.weight"
    marker = f".{moe}.experts."
    depth = len(prefix.split("."))          # `model.layers` -> 2, `layers` -> 1
    lays, es = set(), set()
    for name in sh.wm:
        if marker in name and name.endswith(gate_key) and name.startswith(prefix + "."):
            parts = name.split(".")
            lays.add(int(parts[depth]))
            es.add(int(parts[parts.index("experts") + 1]))
    if layers is None:
        layers = sorted(lays)
    E = max(es) + 1
    n_e = min(E, limit_experts) if limit_experts else E

    # geometry from layer0/expert0 shapes
    _n0 = _expert_names(layers[0], 0, prefix, proj, moe=moe)
    g_shape = sh.locate(_n0[proj[0]])[3]
    d_shape = sh.locate(_n0[proj[2]])[3]
    if source == "mxfp4":
        # packed [rows, K//2] on disk; the logical matrix is twice as wide in K
        g_shape = [g_shape[0], g_shape[1] * 2]
        d_shape = [d_shape[0], d_shape[1] * 2]
    I, H = g_shape
    assert d_shape == [H, I], (g_shape, d_shape)
    segs = [
        ("nf4.gate_up_blocks", (2 * I, H // 2), "U8", 2 * I * (H // 2)),
        ("nf4.gate_up_absmax", (2 * I, H // 64), "F32", 2 * I * (H // 64) * 4),
        ("nf4.down_blocks", (H, I // 2), "U8", H * (I // 2)),
        ("nf4.down_absmax", (H, I // 64), "F32", H * (I // 64) * 4),
    ]
    seg_geo, off = [], 0
    for suf, shape, dt, ln in segs:
        seg_geo.append({"suffix": suf, "seg_off": off, "length": ln,
                        "shape_per_expert": list(shape), "dtype": dt})
        off = _align8(off + ln)
    row_bytes = off
    row_stride = _align(row_bytes, align)
    log(f"nf4 bake: L={len(layers)} E={E} (baking {n_e}/layer) I={I} H={H} "
        f"row={row_bytes} stride={row_stride} "
        f"total={len(layers) * n_e * row_stride / 1e9:.1f} GB")

    if quantize_fn is None:
        quantize_fn = default_quantize_expert()
    import torch  # after quantize_fn resolution so mocks stay torch-light

    rows, man_rows = [], []
    arena_off = 0
    t0 = time.time()
    with open(out, "wb") as dst:
        for li, lay in enumerate(layers):
            for e in range(n_e):
                names = _expert_names(lay, e, prefix, proj, moe=moe)
                if source == "mxfp4":
                    stem = names[proj[0]].rsplit(".", 1)[0]
                    gate, src_g = sh.read_mxfp4(stem, *mxfp4_suffixes)
                    up, src_u = sh.read_mxfp4(names[proj[1]].rsplit(".", 1)[0], *mxfp4_suffixes)
                    down, src_d = sh.read_mxfp4(names[proj[2]].rsplit(".", 1)[0], *mxfp4_suffixes)
                else:
                    gate, src_g = sh.read_bf16(names[proj[0]])
                    up, src_u = sh.read_bf16(names[proj[1]])
                    down, src_d = sh.read_bf16(names[proj[2]])
                    src_g, src_u, src_d = [src_g], [src_u], [src_d]
                gu_b, gu_a = quantize_fn(torch.cat([gate, up], 0))
                dn_b, dn_a = quantize_fn(down)
                del gate, up, down
                seg_bytes = [
                    (gu_b.contiguous().view(torch.uint8).numpy().tobytes(),
                     src_g + src_u),
                    (gu_a.contiguous().view(torch.uint8).numpy().tobytes(),
                     src_g + src_u),
                    (dn_b.contiguous().view(torch.uint8).numpy().tobytes(),
                     src_d),
                    (dn_a.contiguous().view(torch.uint8).numpy().tobytes(),
                     src_d),
                ]
                del gu_b, gu_a, dn_b, dn_a
                man_segs = []
                for g, (raw, srcs) in zip(seg_geo, seg_bytes):
                    assert len(raw) == g["length"], (g["suffix"], len(raw))
                    dst.seek(arena_off + g["seg_off"])
                    dst.write(raw)
                    man_segs.append({
                        "suffix": g["suffix"],
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "sources": [{"source_file": f, "source_range": [lo, hi],
                                     "sha256": h} for f, lo, hi, h in srcs],
                    })
                rows.append([lay, e, arena_off])
                man_rows.append({"layer": lay, "expert": e,
                                 "offset": arena_off, "segments": man_segs})
                arena_off += row_stride
            if (li + 1) % 4 == 0 or li == len(layers) - 1:
                log(f"  baked layer {lay} ({li + 1}/{len(layers)}, "
                    f"{arena_off / 1e9:.1f} GB, {time.time() - t0:.0f}s)")
        # truncate-to-size, not a per-row pad write: the latter clobbers the
        # last data byte when row_bytes == row_stride (4096-aligned row —
        # Qwen3-235B's NF4 row is exactly aligned; caught by the on-pod smoke)
        dst.flush()
        dst.truncate(arena_off)
        os.fsync(dst.fileno())

    try:
        import bitsandbytes
        bnb_v = bitsandbytes.__version__
    except Exception:
        bnb_v = None
    quantizer = {"kind": "bnb.quantize_4bit", "quant_type": "nf4",
                 "blocksize": 64, "bnb": bnb_v,
                 "torch": __import__("torch").__version__,
                 "layout": "cat[gate;up] dim0, nested-absmax dequantized f32",
                 "source": source, "proj": list(proj), "moe_attr": moe}
    index = {"magic": MAGIC, "version": VERSION, "snapshot": snapshot,
             "prefix": prefix, "align": align, "row_bytes": row_bytes,
             "row_stride": row_stride, "n_layers": len(layers),
             "n_experts_per_layer": n_e, "segments": seg_geo, "rows": rows,
             "arena_bytes": arena_off, "bake_mode": "nf4-quantize",
             "quantizer": quantizer,
             "model_dims": {"I": I, "H": H, "E": E}}
    manifest = {"magic": MAGIC, "version": VERSION, "algo": "sha256",
                "snapshot": snapshot, "bake_mode": "nf4-quantize",
                "quantizer": quantizer,
                "baked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rows": man_rows}
    with open(out + ".index.json", "w") as f:
        json.dump(index, f)
    with open(out + ".manifest.json", "w") as f:
        json.dump(manifest, f)
    log(f"nf4 bake complete: {len(rows)} rows ({arena_off / 1e9:.1f} GB) "
        f"in {time.time() - t0:.0f}s")
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None, help="e.g. 0-93")
    ap.add_argument("--align", type=int, default=4096)
    ap.add_argument("--limit-experts", type=int, default=0)
    args = ap.parse_args()
    layers = None
    if args.layers:
        if "-" in args.layers:
            a, b = args.layers.split("-")
            layers = list(range(int(a), int(b) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]
    bake_nf4(args.snapshot, args.out, layers=layers, align=args.align,
             limit_experts=args.limit_experts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
