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
    mid = f"{moe}." if moe else ""       # falsy moe: experts sit on the layer
    base = f"{prefix}.{layer}.{mid}experts.{e}."
    return {p: f"{base}{p}.{suffix}" for p in proj}


#: absmax storage options. "f32" is what shipped first and is always safe;
#: "bf16" halves the absmax segment (-5.6% of a Qwen3-30B row) and is BITWISE
#: LOSSLESS for a bf16 checkpoint; "auto" picks bf16 exactly when the source
#: dtype proves it lossless.
ABSMAX_DTYPES = ("f32", "bf16", "auto")

#: source dtypes for which bf16 absmax is lossless BY CONSTRUCTION. absmax is
#: ``|w|.amax()`` over a block, so it IS one of the source magnitudes; the max
#: of a set of bf16 values is a bf16 value. Measured on Qwen3-30B: 80/80 expert
#: tensors bitwise identical after a bf16 round-trip, and an fp32 control
#: correctly NOT identical -- which is why this is a whitelist, not a default.
#:
#: **fp16 is deliberately NOT here.** It has 10 mantissa bits to bf16's 7, so an
#: fp16 magnitude is not generally bf16-representable and the proof above does
#: not carry. ``auto`` therefore leaves an fp16 checkpoint on f32 rather than
#: choosing a mode ``cast_absmax`` would then refuse tensor by tensor. Storing
#: fp16 absmax AS fp16 would be lossless and equally small, and the consumer
#: side already allows the f16->f32 widening; it is simply not implemented here,
#: because no checkpoint in use needs it.
_ABSMAX_LOSSLESS_SOURCES = ("bf16",)


def resolve_absmax_dtype(absmax_dtype: str, source: str) -> str:
    """('auto'|'bf16'|'f32', source) -> 'bf16' | 'f32'.

    ``auto`` decides from the SOURCE dtype rather than by sampling values: the
    losslessness is a proof about bf16 arithmetic, and a sample could pass on
    the experts it looked at and lose precision on one it did not.
    """
    if absmax_dtype not in ABSMAX_DTYPES:
        raise ValueError(f"absmax_dtype must be one of {ABSMAX_DTYPES}, "
                         f"got {absmax_dtype!r}")
    if absmax_dtype != "auto":
        return absmax_dtype
    return "bf16" if source in _ABSMAX_LOSSLESS_SOURCES else "f32"


def cast_absmax(am, want: str):
    """Cast absmax to the arena's storage dtype, refusing a LOSSY cast.

    The check is per tensor and deliberately not a warning. ``bf16`` is chosen
    because it is exact, so an inexact tensor means the assumption behind the
    choice is false for this checkpoint — and silently rounding the scale of
    every weight in the block is not a thing to discover later from an accuracy
    regression. Cheap: absmax is K/64 of the weight elements.

    Applied in ``bake_nf4`` rather than only inside the default quantizer, so an
    INJECTED ``quantize_fn`` cannot hand back a dtype the row geometry did not
    budget for — that would write a short row and corrupt every later offset.
    """
    import torch

    if want == "f32":
        return am.to(torch.float32)
    if want != "bf16":
        raise ValueError(f"unknown absmax storage dtype {want!r}")
    am32 = am.to(torch.float32)
    small = am32.to(torch.bfloat16)
    if not torch.equal(small.to(torch.float32), am32):
        raise ValueError(
            "refusing to store absmax as bf16: the round-trip is not exact for "
            "this tensor, so it would change what the model computes. This is "
            "expected for an fp32 checkpoint — bake with --absmax-dtype f32 "
            "(or 'auto', which picks f32 for such sources).")
    return small


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


_MXFP4_BYTE_DTYPES = ("U8", "I8", "F8_E8M0")
"""Byte-typed labels seen across MXFP4 checkpoints for BOTH weight and scale.
V4 says I8/F8_E8M0, K3 says U8/U8 -- same bytes, different label, so the reader
accepts the set rather than one spelling."""

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
        b_raw, b_shape, b_dt, b_src = self._raw(base + weight_suffix)
        s_raw, s_shape, s_dt, s_src = self._raw(base + scale_suffix)
        # Deliberately NOT an equality check: V4 labels these I8/F8_E8M0 and K3 labels
        # both U8 for byte-identical content, so the reader takes raw bytes. What must
        # be rejected is the OTHER format -- block-scaled FP8 spells its tensors
        # `.weight`/`.scale` too, and crossing it into here otherwise dies much later on
        # an opaque reshape (its F32 scale carries 4x the bytes the shape implies).
        for nm, dt, ok in ((weight_suffix, b_dt, _MXFP4_BYTE_DTYPES),
                           (scale_suffix, s_dt, _MXFP4_BYTE_DTYPES)):
            if dt not in ok:
                hint = ("; this is block-scaled FP8 -- use source='fp8'"
                        if dt in ("F8_E4M3", "F32") else "")
                raise ValueError(
                    f"{base + nm!r} is {dt}, expected one of {sorted(ok)} for "
                    f"source='mxfp4'{hint}")
        rows, kh = b_shape
        groups = s_shape[1]
        blocks = torch.frombuffer(bytearray(b_raw), dtype=torch.uint8).reshape(
            rows, groups, kh // groups)
        scales = torch.frombuffer(bytearray(s_raw), dtype=torch.uint8).reshape(s_shape)
        w = dequant_mxfp4(blocks, scales, dtype=torch.float32).to(torch.bfloat16)
        return w, [b_src, s_src]

    def read_fp8(self, base, weight_suffix=".weight", scale_suffix=".scale",
                 block=(128, 128)):
        """Block-scaled FP8: `F8_E4M3` weights + one **F32** scale per `block` tile.

        This is a DIFFERENT format from `read_mxfp4`, despite DeepSeek-V4 spelling both
        pairs `.weight`/`.scale`. Two things change and both are silent if crossed:

        * the on-disk weight shape IS the logical shape. MXFP4 packs two nibbles per byte
          so its K is half the logical K; FP8 is one byte per element and doubling K here
          would read a matrix twice as wide as the model has.
        * the scale is F32 covering a `[128, 128]` TILE, not an e8m0 byte covering 32
          contiguous elements. It is already the multiplier, so it is applied directly --
          no `2**(x-127)`.

        The instruct V4-Flash ships MXFP4 experts (137 GiB); V4-Flash-Base ships these
        (258 GiB). Same architecture, same tensor names, 1.9x the bytes.
        """
        torch = self.torch
        w_raw, w_shape, w_dt, w_src = self._raw(base + weight_suffix)
        s_raw, s_shape, s_dt, s_src = self._raw(base + scale_suffix)
        if w_dt != "F8_E4M3":
            raise ValueError(f"{base + weight_suffix!r} is {w_dt}, expected F8_E4M3 for "
                             f"source='fp8' (MXFP4 checkpoints use source='mxfp4')")
        if s_dt != "F32":
            raise ValueError(f"{base + scale_suffix!r} is {s_dt}, expected F32 block "
                             f"scales; an e8m0 byte scale means this is MXFP4, not FP8")
        # read as raw bytes and reinterpret, so a torch too old to NAME float8_e4m3fn in
        # `frombuffer` still works -- the same discipline read_mxfp4 uses.
        w = (torch.frombuffer(bytearray(w_raw), dtype=torch.uint8)
             .view(torch.float8_e4m3fn).reshape(w_shape).float())
        sc = torch.frombuffer(bytearray(s_raw), dtype=torch.float32).reshape(s_shape)
        bh, bw = block
        exp = sc.repeat_interleave(bh, 0).repeat_interleave(bw, 1)
        if exp.shape[0] < w_shape[0] or exp.shape[1] < w_shape[1]:
            raise ValueError(f"scale {tuple(s_shape)} x block {block} = {tuple(exp.shape)} "
                             f"does not cover weight {tuple(w_shape)}")
        return (w * exp[:w_shape[0], :w_shape[1]]).to(torch.bfloat16), [w_src, s_src]

    def read_bf16_slab(self, name, e, n_experts):
        """One expert out of a FUSED [E, X, Y] tensor, by byte range.

        A contiguous [E, X, Y] tensor is E contiguous [X, Y] slabs, so expert
        `e` is a sub-range of the parent's own range and needs no whole-layer
        materialization -- Gemma-4's per-layer gate_up is 1.01 GB.

        Provenance is unchanged in KIND: the record is already
        (file, byte range, sha256), and a slab is a byte range like any other,
        so `verify --against-source` re-checks exactly the bytes consumed with
        no schema change.
        """
        path, lo, hi, shape, dtype = self.locate(name)
        assert dtype == "BF16", (name, dtype)
        if len(shape) != 3:
            raise ValueError(f"{name!r} is {len(shape)}-D, expected [E, X, Y]")
        E, X, Y = shape
        if E != n_experts:
            raise ValueError(f"{name!r} has E={E}, expected {n_experts}")
        slab = X * Y * 2
        if (hi - lo) != E * slab:
            raise ValueError(f"{name!r}: byte range {hi - lo} != E*X*Y*2 {E * slab}")
        s_lo = lo + e * slab
        with open(path, "rb") as f:
            f.seek(s_lo)
            raw = f.read(slab)
        t = self.torch.frombuffer(bytearray(raw), dtype=self.torch.bfloat16)
        return t.reshape(X, Y), (os.path.basename(path), s_lo, s_lo + slab,
                                 hashlib.sha256(raw).hexdigest())

    def read_bf16(self, name):
        path, lo, hi, shape, dtype = self.locate(name)
        assert dtype == "BF16", (name, dtype)
        with open(path, "rb") as f:
            f.seek(lo)
            raw = f.read(hi - lo)
        t = self.torch.frombuffer(bytearray(raw), dtype=self.torch.bfloat16)
        return t.reshape(shape), (os.path.basename(path), lo, hi,
                                  hashlib.sha256(raw).hexdigest())


def _explain_no_experts(sh, prefix, marker, gate_key):
    """Say what was searched for and what the checkpoint actually has.

    Discovery matching nothing used to surface as ``max() arg is an empty
    sequence`` one line later, which names none of the three things that decide
    the match. That has now cost a diagnosis twice: Kimi K3 spelling its weights
    ``.weight_packed``, and Gemma-4 nesting its stack under
    ``model.language_model.layers`` with a FUSED expert tensor and no per-expert
    index. Both were minutes of reading this function to learn what it wanted.
    """
    near = sorted({n for n in sh.wm if "expert" in n.lower()})
    # Detect a FUSED layout from the checkpoint itself rather than from whatever
    # got through the strict filter. Gemma-4 differs from the default in THREE
    # ways at once -- no MoE attribute, no per-expert index, no `.weight` suffix
    # -- so a user running the defaults matches nothing, and a diagnosis that
    # only fires once they have already guessed two of the three is no help.
    unindexed = []
    for n in near:
        parts = n.split(".")
        if "experts" in parts:
            i = parts.index("experts")
            if i + 1 < len(parts) and not parts[i + 1].isdigit():
                unindexed.append(n)
    lines = [
        "found no per-expert tensors to bake.",
        f"  searched: names starting {prefix + '.'!r}, containing {marker!r}, ending {gate_key!r}",
    ]
    if unindexed:
        lines += [
            # NOT "matched": these come from the checkpoint's whole key list, and
            # in the default Gemma-4 path they fail all three checks above. Saying
            # "matched" directly under the `searched:` line claimed the opposite
            # of what happened, in the one message meant to end the confusion.
            f"  the checkpoint has {len(unindexed)} key(s) with NO per-expert index "
            f"after 'experts' (they need not have matched the search), e.g.",
            f"    {unindexed[0]}",
            "  that is a FUSED expert layout (one 3-D tensor per layer). This bake path",
            "  reads per-expert 2-D tensors and does not support it.",
        ]
    elif near:
        lines += [
            f"  the checkpoint has {len(near)} key(s) containing 'expert', e.g.",
            *[f"    {n}" for n in near[:3]],
            "  adjust prefix=/moe=/proj= to match, or the source= suffix if it is not bf16.",
        ]
    else:
        lines.append("  the checkpoint has NO keys containing 'expert' at all.")
    raise ValueError("\n".join(lines))


def bake_nf4(snapshot, out, *, layers=None, prefix="model.layers",
             align=4096, limit_experts=0, quantize_fn=None, log=print,
             proj=PROJ, source="bf16", moe="mlp",
             mxfp4_suffixes=(".weight", ".scale"),
             fused_proj=("gate_up_proj", "down_proj"),
             absmax_dtype="f32"):
    """Quantize-bake. Discovers (L, E) from the checkpoint index; emits the
    same arena/index/manifest triple as the relocation bake, with the
    two-hop provenance schema.

    ``absmax_dtype`` is ``"f32"`` (default, unchanged), ``"bf16"``, or
    ``"auto"``. bf16 halves the absmax segment — 11.1% of a Qwen3-30B row down
    to 5.6% — and is bitwise lossless for a bf16 checkpoint; see
    :func:`cast_absmax`. The default stays f32 because the arena's index is
    self-describing but *older consumers are not*: a reader that predates bf16
    absmax refuses the segment outright, so flipping the default would break
    them on a library upgrade alone.
    Use it to quantise a bf16 checkpoint's experts to NF4 at bake time into an arena the
    NF4 residency tiers serve; the manifest's ``bake_mode`` records that the bytes are the
    quantiser's output, not a bf16 release. Pure torch, no GPU. See
    ``docs/solutions/stream-moe-experts-from-host-or-nvme.md``.
    """
    am_store = resolve_absmax_dtype(absmax_dtype, source)
    if source not in ("bf16", "mxfp4", "fp8"):
        raise ValueError(f"source must be 'bf16', 'mxfp4' or 'fp8'; got {source!r}")
    sh = _Shards(snapshot)
    # Discovery and the geometry probe must look for the suffix this SOURCE actually
    # uses. 0.5.0 parameterized the READ (`mxfp4_suffixes`) but left these two hardcoded
    # to `.weight`, so a checkpoint spelling it otherwise -- Kimi K3's `.weight_packed`
    # -- matched zero keys and died on `max()` of an empty sequence, one line into the
    # bake. Parameterizing the read was necessary and not sufficient; only running it on
    # real K3 bytes showed that.
    wsuf = mxfp4_suffixes[0] if source in ("mxfp4", "fp8") else ".weight"
    gate_key = f"{proj[0]}{wsuf}"
    # A falsy `moe` means the experts hang straight off the layer, with no
    # block attribute between -- Gemma-4 spells it
    # `model.language_model.layers.0.experts.gate_up_proj`. Building the
    # marker unconditionally as `.{moe}.experts.` made that layout
    # unmatchable for EVERY value of `moe` ("" gives "..experts."), so the
    # fused-layout diagnosis below could never fire on the checkpoint that
    # motivated it (Bugbot, PR #55).
    marker = f".{moe}.experts." if moe else ".experts."
    depth = len(prefix.split("."))          # `model.layers` -> 2, `layers` -> 1
    lays, es = set(), set()
    for name in sh.wm:
        if marker in name and name.endswith(gate_key) and name.startswith(prefix + "."):
            parts = name.split(".")
            after = parts[parts.index("experts") + 1]
            if not after.isdigit():
                # No per-expert index to parse. Skip rather than crash on
                # int(); _explain_no_experts re-derives this from the checkpoint.
                continue
            lays.add(int(parts[depth]))
            es.add(int(after))
    # FUSED layout: one 3-D [E, X, Y] tensor per layer, no per-expert index.
    # Detected rather than flagged, so a Gemma-4 bake needs only the prefix the
    # #55 error already prints. E comes from the SHAPE, not from a name.
    fused_names = {}
    if not es and source == "bf16":
        for name in sh.wm:
            if (name.startswith(prefix + ".") and ".experts." in name
                    and name.endswith("." + fused_proj[0])):
                # The segment at `depth` must BE the layer id. A near-miss
                # prefix -- `model.language_model` for a checkpoint whose stack
                # is `model.language_model.layers` -- still matches every key
                # here but puts "layers" at that position, and int() on it
                # crashes with exactly the raw error this module exists to
                # replace. Skip instead, so discovery comes up empty and
                # _explain_no_experts prints the structured diagnostic naming
                # what was searched for (Bugbot, PR #57).
                seg = name.split(".")[depth]
                if not seg.isdigit():
                    continue
                fused_names[int(seg)] = name
    fused = bool(fused_names)
    if not es and not fused:
        _explain_no_experts(sh, prefix, marker, gate_key)

    if fused:
        if layers is None:
            layers = sorted(fused_names)
        g_full = sh.locate(fused_names[layers[0]])[3]          # [E, 2I, H]
        d_full = sh.locate(fused_names[layers[0]].replace(
            "." + fused_proj[0], "." + fused_proj[1]))[3]      # [E, H, I]
        if len(g_full) != 3 or len(d_full) != 3:
            raise ValueError(f"fused tensors must be 3-D; got {g_full} and {d_full}")
        E = g_full[0]
        g_shape = [g_full[1] // 2, g_full[2]]                  # -> [I, H]
        d_shape = [d_full[1], d_full[2]]                       # -> [H, I]
        if g_full[1] % 2:
            raise ValueError(f"fused gate_up dim1={g_full[1]} is odd; expected 2*intermediate")
        # Per-slab and whole-tensor blocking coincide ONLY when each expert's
        # numel is a multiple of the blocksize -- bitsandbytes blocks 64
        # CONTIGUOUS elements. Verified bitwise on Gemma-4 before this was
        # written (gnf4#56 step 0); a checkpoint that fails it would bake rows
        # that silently do not match what the loader builds, so refuse instead.
        for lbl, (a, b) in (("gate_up", (g_full[1], g_full[2])),
                            ("down", (d_full[1], d_full[2]))):
            if (a * b) % 64:
                raise ValueError(
                    f"fused {lbl} slab is {a}x{b} = {a * b} elements, not a multiple of "
                    f"the 64-element block. Per-expert and whole-stack quantization would "
                    f"disagree, so the baked rows would not match what the loader builds.")
        log(f"nf4 bake: FUSED layout, E={E} from {fused_names[layers[0]].split('.')[-1]} shape {g_full}")
    else:
        if layers is None:
            layers = sorted(lays)
        E = max(es) + 1
        # geometry from layer0/expert0 shapes
        _n0 = _expert_names(layers[0], 0, prefix, proj, suffix=wsuf.lstrip("."), moe=moe)
        g_shape = sh.locate(_n0[proj[0]])[3]
        d_shape = sh.locate(_n0[proj[2]])[3]
    n_e = min(E, limit_experts) if limit_experts else E
    if source == "mxfp4":
        # packed [rows, K//2] on disk; the logical matrix is twice as wide in K.
        # FP8 is one byte per element, so its on-disk shape is ALREADY logical --
        # doubling it here would describe a matrix twice as wide as the model has.
        g_shape = [g_shape[0], g_shape[1] * 2]
        d_shape = [d_shape[0], d_shape[1] * 2]
    I, H = g_shape
    assert d_shape == [H, I], (g_shape, d_shape)
    am_dt = "BF16" if am_store == "bf16" else "F32"
    am_sz = 2 if am_store == "bf16" else 4
    segs = [
        ("nf4.gate_up_blocks", (2 * I, H // 2), "U8", 2 * I * (H // 2)),
        ("nf4.gate_up_absmax", (2 * I, H // 64), am_dt, 2 * I * (H // 64) * am_sz),
        ("nf4.down_blocks", (H, I // 2), "U8", H * (I // 2)),
        ("nf4.down_absmax", (H, I // 64), am_dt, H * (I // 64) * am_sz),
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
                if fused:
                    gname = fused_names[lay]
                    dname = gname.replace("." + fused_proj[0], "." + fused_proj[1])
                    # Already [2I, H] on disk -- no cat, unlike the per-expert path
                    # which concatenates gate and up to build the same matrix.
                    gate_up, src_gu = sh.read_bf16_slab(gname, e, E)
                    down, src_d = sh.read_bf16_slab(dname, e, E)
                    # ONE source slab feeds the gate_up segments. The per-expert
                    # path lists two because it concatenates two separate
                    # tensors; carrying that shape over here would record the
                    # same byte range twice and overstate what was consumed.
                    src_g, src_u = [src_gu], []
                    src_d = [src_d]
                    gu_b, gu_a = quantize_fn(gate_up)
                    dn_b, dn_a = quantize_fn(down)
                    del gate_up, down
                    _fused_done = True
                else:
                    _fused_done = False
                    names = _expert_names(lay, e, prefix, proj, moe=moe)
                if not _fused_done and source in ("mxfp4", "fp8"):
                    rd = sh.read_mxfp4 if source == "mxfp4" else sh.read_fp8
                    stems = [names[p].rsplit(".", 1)[0] for p in proj]
                    gate, src_g = rd(stems[0], *mxfp4_suffixes)
                    up, src_u = rd(stems[1], *mxfp4_suffixes)
                    down, src_d = rd(stems[2], *mxfp4_suffixes)
                elif not _fused_done:
                    gate, src_g = sh.read_bf16(names[proj[0]])
                    up, src_u = sh.read_bf16(names[proj[1]])
                    down, src_d = sh.read_bf16(names[proj[2]])
                    src_g, src_u, src_d = [src_g], [src_u], [src_d]
                if not _fused_done:
                    gu_b, gu_a = quantize_fn(torch.cat([gate, up], 0))
                    dn_b, dn_a = quantize_fn(down)
                    del gate, up, down
                # AFTER the quantizer, so an injected one cannot hand back a
                # width the row geometry did not budget for -- that would write
                # a short row and shift every offset after it.
                gu_a = cast_absmax(gu_a, am_store)
                dn_a = cast_absmax(dn_a, am_store)
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
                 "layout": f"cat[gate;up] dim0, nested-absmax dequantized, "
                           f"absmax stored {am_store}",
                 "absmax_dtype": am_store,
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
    # bake_nf4() has always taken these; the CLI did not expose them, so a
    # checkpoint that nests its stack (Gemma-4: model.language_model.layers) or
    # names its MoE block differently (Mixtral: block_sparse_moe) could only be
    # baked by importing the function.
    ap.add_argument("--prefix", default="model.layers",
                    help="layer-stack prefix, e.g. model.language_model.layers")
    ap.add_argument("--absmax-dtype", default="f32", choices=ABSMAX_DTYPES,
                    help="absmax storage. bf16 halves that segment (-5.6%% of "
                         "the row) and is bitwise lossless for a bf16 "
                         "checkpoint; auto picks it only for such sources. "
                         "Default f32 — consumers older than bf16 absmax "
                         "refuse the segment.")
    ap.add_argument("--moe", default="mlp",
                    help="MoE block attribute between the layer and 'experts'")
    args = ap.parse_args()
    layers = None
    if args.layers:
        if "-" in args.layers:
            a, b = args.layers.split("-")
            layers = list(range(int(a), int(b) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]
    bake_nf4(args.snapshot, args.out, layers=layers, align=args.align,
             limit_experts=args.limit_experts, prefix=args.prefix, moe=args.moe,
             absmax_dtype=args.absmax_dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
