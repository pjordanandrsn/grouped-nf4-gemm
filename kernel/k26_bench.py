"""Lane K26 bench (``PREREG-k26-nf4-decode-ablation.md``): where does K25's time go on the NF4 families' B=16 shapes?

experts4bit-qlora lane P92 read K25's GEMM within 4 % of the served NF4 GEMM in-model (Granite 6.46 vs 6.60 ms/step,
OLMoE 10.85 vs 10.39). Both decode each nibble through a codebook. This bench ablates K25's decode on one card:

- ``pair``      the product kernel (``nf4_smallm.gemm_nf4_grouped_smallm``, its default plan and decode)
- ``copy``      a bench-local copy of K25's kernel, the same decode: the control (bit-equal outputs, time within 5 %)
- ``affine``    the copy with the codebook replaced by ``nibble - 8`` (same bytes, same absmax, no lookup)
- ``noscale``   ``affine`` without the absmax load and multiply
- ``bytes``     the packed bytes widened straight to the MMA operand (no nibble split, no scale)
- ``tree``      the copy with the codebook lookup replaced by a 4-level select tree on the nibble's bits over the 16
                fp32 codebook values (loaded once per program): the same weights, so a candidate exact decode
- ``served``    ``nf4_grouped.gemm_4bit_grouped_captured`` (TF32, BLOCK_K 64, the route OFF serves)
- ``scopy``     a bench-local copy of the served kernel's path (the control: bit-equal, time within 5 %)
- ``stree``     that copy with the exact select tree in place of its tl.gather codebook
- ``load``      the product kernel with ``lut="load"`` at KC 128 (it overflows shared memory at KC 256); descriptive

Every arm times only its two GEMMs per layer (gate_up with the in-kernel gather, down on sorted rows), with the tile
tables and inputs built outside the CUDA graph, over each layer's OWN synthetic NF4 stores (K24's lesson) and seeded
synthetic top-k routing at B=16. The floor is the step's active-expert bytes over a device-copy bandwidth.

    python k26_bench.py out.json [--families granite,olmoe] [--quick]
    python k26_bench.py --self-test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

FAMILIES = {                          # experts, top-k, layers, gate_up (N, K), down (N, K)
    "granite": dict(E=40, k=8, L=32, gu=(1024, 1536), dn=(1536, 512)),
    "olmoe": dict(E=64, k=8, L=16, gu=(2048, 2048), dn=(2048, 1024)),
}
B = 16
STEPS = 8
ITERS = 20
PLAN = dict(block_n=32, kc=256, warps=4, stages=2)        # K25's default (nf4_smallm.gemm_nf4_grouped_smallm)
LOAD_PLAN = dict(block_n=32, kc=128, warps=4, stages=2)   # "load" overflows 99 KB of shared memory at KC 256
COPY_BAND = 0.05
DECODE, NOT_DECODE = 0.60, 0.85
TREE_BAR, STREE_BAR = 0.80, 0.90          # an exact decode qualifies when bit-equal and at most this fraction of the time
NUM_TOL = 2 ** -6
ARMS = ("pair", "copy", "affine", "noscale", "bytes", "tree", "served", "scopy", "stree", "load")


def verdict(rec):
    """``(verdict, reason)`` from the per-family medians; the rule is the prereg's."""
    fams = rec.get("families") or {}
    if set(fams) != set(FAMILIES):
        return "VOID", f"families measured {sorted(fams)}, registered {sorted(FAMILIES)}"
    ratios = {}
    for name, f in fams.items():
        if not f.get("numerics_ok"):
            return "VOID", f"{name}: numerics failed ({f.get('numerics')})"
        if not f.get("copy_bit_equal"):
            return "VOID", f"{name}: the bench copy is not bit-equal to the product kernel"
        med = f.get("median_ms") or {}
        missing = [a for a in ("pair", "copy", "affine", "noscale", "bytes", "served") if not med.get(a)]
        if missing:
            return "VOID", f"{name}: arms missing {missing}"
        if abs(med["copy"] / med["pair"] - 1) > COPY_BAND:
            return "VOID", f"{name}: copy {med['copy']:.3f} ms vs product {med['pair']:.3f} ms, outside {COPY_BAND:.0%}"
        ratios[name] = med["affine"] / med["pair"]
    txt = ", ".join(f"{n} {r:.3f}" for n, r in ratios.items())
    if all(r <= DECODE for r in ratios.values()):
        v, why = "DECODE", f"affine/pair {txt}: <= {DECODE} in both -- the codebook decode is the bottleneck"
    elif all(r >= NOT_DECODE for r in ratios.values()):
        v, why = "NOT_DECODE", f"affine/pair {txt}: >= {NOT_DECODE} in both -- the codebook decode is not the bottleneck"
    else:
        v, why = "MIXED", f"affine/pair {txt}: between {DECODE} and {NOT_DECODE}, or split across families"
    return v, why + "; next: " + next_lane(fams)


def next_lane(fams):
    """The registered pointer: which exact decode, if any, the next lane builds. An exact decode must be bit-equal to
    the kernel it replaces (then it is a pure speed change and no quality read is needed) and fast enough in BOTH
    families."""
    out = []
    t = [f["median_ms"].get("tree", 0) / f["median_ms"]["pair"] for f in fams.values()]
    if all(f.get("tree_bit_equal") for f in fams.values()) and all(0 < r <= TREE_BAR for r in t):
        out.append(f"K25 takes the select-tree decode (bit-equal; tree/pair {max(t):.3f} <= {TREE_BAR})")
    s = [f["median_ms"].get("stree", 0) / f["median_ms"]["served"] for f in fams.values()]
    if all(f.get("stree_bit_equal") for f in fams.values()) and all(0 < r <= STREE_BAR for r in s):
        out.append(f"the served kernel takes the select tree (bit-equal; stree/served {max(s):.3f} <= {STREE_BAR})")
    return "; ".join(out) or "no exact decode qualifies"


def self_test():
    def fam(pair=10.0, copy=10.1, affine=4.0, ok=True, eq=True, drop=None, tree=5.0, teq=True, stree=11.0, seq=False):
        med = {"pair": pair, "copy": copy, "affine": affine, "noscale": 3.5, "bytes": 3.0, "tree": tree, "served": 10.0,
               "scopy": 10.0, "stree": stree, "load": 12.0}
        if drop:
            med.pop(drop)
        return {"numerics_ok": ok, "copy_bit_equal": eq, "tree_bit_equal": teq, "stree_bit_equal": seq, "median_ms": med}

    def rec(**kw):
        return {"families": {"granite": fam(**kw.get("g", {})), "olmoe": fam(**kw.get("o", {}))}}
    assert verdict(rec())[0] == "DECODE"
    assert verdict(rec(g={"affine": 6.0}, o={"affine": 6.0}))[0] == "DECODE"                    # the boundary
    assert verdict(rec(g={"affine": 9.0}, o={"affine": 8.5}))[0] == "NOT_DECODE"
    assert verdict(rec(g={"affine": 4.0}, o={"affine": 9.0}))[0] == "MIXED"                     # split
    assert verdict(rec(g={"affine": 7.0}, o={"affine": 7.0}))[0] == "MIXED"
    assert verdict(rec(g={"copy": 10.6}))[0] == "VOID"                                          # copy 6 % off
    assert verdict(rec(g={"copy": 9.6}))[0] == "DECODE"                                         # 4 %: inside
    assert verdict(rec(o={"eq": False}))[0] == "VOID"
    assert verdict(rec(o={"ok": False}))[0] == "VOID"
    assert verdict(rec(g={"drop": "noscale"}))[0] == "VOID"
    assert verdict({"families": {"granite": fam()}})[0] == "VOID"
    assert "K25 takes the select-tree decode" in verdict(rec())[1]                                 # 0.5 <= 0.80, equal
    assert "no exact decode qualifies" in verdict(rec(g={"tree": 8.5}))[1]                          # 0.85 > 0.80
    assert "no exact decode qualifies" in verdict(rec(o={"teq": False}))[1]                         # not bit-equal
    assert "the served kernel takes" in verdict(rec(g={"stree": 8.0, "seq": True}, o={"stree": 9.0, "seq": True}))[1]
    assert "the served kernel takes" not in verdict(rec(g={"stree": 8.0, "seq": True}, o={"stree": 9.5, "seq": True}))[1]
    print("k26_bench self-test OK (16 cases)")


# The bench-local copy of ``nf4_smallm._gemm_nf4_grouped_smallm`` (EVEN_K, gather, bf16 MMA, no scatter) whose decode is
# a constexpr: 0 the pair codebook (the control), 1 affine ``nibble - 8`` with the absmax, 2 affine without it, 3 the
# packed bytes widened straight to the operand, 4 the exact select tree. Module level, behind the shim (triton's JIT resolves module globals),
# so ``--self-test`` imports without triton.
from _triton_shim import tl, triton  # noqa: E402

_TL_INTERLEAVE = getattr(tl, "interleave", None)


@triton.jit
def _k26(x_ptr, ord_ptr, w_ptr, am_ptr, lut_ptr, lut16_ptr, row0_ptr, rows_ptr, grp_ptr, out_ptr,
         N, stride_we, stride_wn, stride_ae, stride_an,
         K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KC: tl.constexpr,
         GATHER: tl.constexpr, DEC: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = tl.load(rows_ptr + g)
    if rows == 0:
        return
    row0 = tl.load(row0_ptr + g).to(tl.int64)
    eid = tl.load(grp_ptr + g).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < rows
    if GATHER:
        src = tl.load(ord_ptr + row0 + offs_m, mask=m_mask, other=0).to(tl.int64)
    else:
        src = row0 + offs_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    NKC: tl.constexpr = K // KC
    NSC: tl.constexpr = KC // 64
    offs_kc = tl.arange(0, KC)
    offs_kh = tl.arange(0, KC // 2)
    offs_sc = tl.arange(0, NSC)
    wbase = w_ptr + eid * stride_we + offs_n.to(tl.int64)[:, None] * stride_wn
    abase = am_ptr + eid * stride_ae + offs_n.to(tl.int64)[:, None] * stride_an
    if DEC == 4:                                       # the 16 codebook values, once per program
        c0 = tl.load(lut16_ptr + 0)
        c1 = tl.load(lut16_ptr + 1)
        c2 = tl.load(lut16_ptr + 2)
        c3 = tl.load(lut16_ptr + 3)
        c4 = tl.load(lut16_ptr + 4)
        c5 = tl.load(lut16_ptr + 5)
        c6 = tl.load(lut16_ptr + 6)
        c7 = tl.load(lut16_ptr + 7)
        c8 = tl.load(lut16_ptr + 8)
        c9 = tl.load(lut16_ptr + 9)
        c10 = tl.load(lut16_ptr + 10)
        c11 = tl.load(lut16_ptr + 11)
        c12 = tl.load(lut16_ptr + 12)
        c13 = tl.load(lut16_ptr + 13)
        c14 = tl.load(lut16_ptr + 14)
        c15 = tl.load(lut16_ptr + 15)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c in range(0, NKC):
        k0 = c * KC
        a = tl.load(x_ptr + src[:, None] * K + k0 + offs_kc[None, :], mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        wb = tl.load(wbase + (k0 // 2) + offs_kh[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        if DEC == 0:
            am = tl.load(abase + (k0 // 64) + offs_sc[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
            pv = tl.load(lut_ptr + wb)
            ev = pv.to(tl.int32).to(tl.float32, bitcast=True)
            od = (pv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            w = _TL_INTERLEAVE(ev, od)
            w = tl.reshape(tl.reshape(w, (BLOCK_N, NSC, 64)) * am[:, :, None], (BLOCK_N, KC))
        elif DEC == 1:
            am = tl.load(abase + (k0 // 64) + offs_sc[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
            w = _TL_INTERLEAVE((((wb >> 4) & 0xF) - 8).to(tl.float32), ((wb & 0xF) - 8).to(tl.float32))
            w = tl.reshape(tl.reshape(w, (BLOCK_N, NSC, 64)) * am[:, :, None], (BLOCK_N, KC))
        elif DEC == 2:
            w = _TL_INTERLEAVE((((wb >> 4) & 0xF) - 8).to(tl.float32), ((wb & 0xF) - 8).to(tl.float32))
        elif DEC == 4:
            am = tl.load(abase + (k0 // 64) + offs_sc[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
            nib = _TL_INTERLEAVE((wb >> 4) & 0xF, wb & 0xF)
            b0 = (nib & 1) != 0
            b1 = (nib & 2) != 0
            b2 = (nib & 4) != 0
            b3 = (nib & 8) != 0
            s0 = tl.where(b0, c1, c0)
            s1 = tl.where(b0, c3, c2)
            s2 = tl.where(b0, c5, c4)
            s3 = tl.where(b0, c7, c6)
            s4 = tl.where(b0, c9, c8)
            s5 = tl.where(b0, c11, c10)
            s6 = tl.where(b0, c13, c12)
            s7 = tl.where(b0, c15, c14)
            u0 = tl.where(b1, s1, s0)
            u1 = tl.where(b1, s3, s2)
            u2 = tl.where(b1, s5, s4)
            u3 = tl.where(b1, s7, s6)
            v0 = tl.where(b2, u1, u0)
            v1 = tl.where(b2, u3, u2)
            w = tl.where(b3, v1, v0)
            w = tl.reshape(tl.reshape(w, (BLOCK_N, NSC, 64)) * am[:, :, None], (BLOCK_N, KC))
        else:
            wf = wb.to(tl.float32)
            w = _TL_INTERLEAVE(wf, wf)
        acc += tl.dot(a, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    ooff = (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + ooff, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# A bench-local copy of the SERVED kernel's path (``nf4_grouped._gemm_nf4_grouped``: VARIANT 1, GROUPS 1, BLOCK_K 64,
# TF32 on fp32 weights), its decode a constexpr: 0 the register codebook through tl.gather (the control, bit-equal to the
# served kernel), 1 the exact select tree. A tree that is bit-equal here would speed the default NF4 route with no
# arithmetic change.
_TL_GATHER = getattr(tl, "gather", None)


@triton.jit
def _k26s(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, t_row0_ptr, t_rows_ptr, t_group_ptr, K, N,
          stride_be, stride_bn, stride_ae, stride_an,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DEC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m).to(tl.int64)
    rows = tl.load(t_rows_ptr + pid_m)
    if rows == 0:
        return
    eid = tl.load(t_group_ptr + pid_m).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < rows
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + (row0 + offs_m)[:, None] * K
    if DEC == 0:
        lut_reg = tl.load(lut_ptr + tl.arange(0, 16))
    else:
        c0 = tl.load(lut_ptr + 0)
        c1 = tl.load(lut_ptr + 1)
        c2 = tl.load(lut_ptr + 2)
        c3 = tl.load(lut_ptr + 3)
        c4 = tl.load(lut_ptr + 4)
        c5 = tl.load(lut_ptr + 5)
        c6 = tl.load(lut_ptr + 6)
        c7 = tl.load(lut_ptr + 7)
        c8 = tl.load(lut_ptr + 8)
        c9 = tl.load(lut_ptr + 9)
        c10 = tl.load(lut_ptr + 10)
        c11 = tl.load(lut_ptr + 11)
        c12 = tl.load(lut_ptr + 12)
        c13 = tl.load(lut_ptr + 13)
        c14 = tl.load(lut_ptr + 14)
        c15 = tl.load(lut_ptr + 15)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(tl.int32)
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        if DEC == 0:
            w = tl.reshape(_TL_GATHER(lut_reg, tl.reshape(nib, [BLOCK_N * BLOCK_K]), 0), [BLOCK_N, BLOCK_K])
        else:
            b0 = (nib & 1) != 0
            b1 = (nib & 2) != 0
            b2 = (nib & 4) != 0
            b3 = (nib & 8) != 0
            s0 = tl.where(b0, c1, c0)
            s1 = tl.where(b0, c3, c2)
            s2 = tl.where(b0, c5, c4)
            s3 = tl.where(b0, c7, c6)
            s4 = tl.where(b0, c9, c8)
            s5 = tl.where(b0, c11, c10)
            s6 = tl.where(b0, c13, c12)
            s7 = tl.where(b0, c15, c14)
            u0 = tl.where(b1, s1, s0)
            u1 = tl.where(b1, s3, s2)
            u2 = tl.where(b1, s5, s4)
            u3 = tl.where(b1, s7, s6)
            v0 = tl.where(b2, u1, u0)
            v1 = tl.where(b2, u3, u2)
            w = tl.where(b3, v1, v0)
        g0 = k0 // 64
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an + g0, mask=n_mask, other=0.0)
        w = w * am[:, None]
        a = tl.load(a_base + kk[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(w))
    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def main(out_path, families, quick=False):
    import torch
    from int4_b32 import build_group_tiles_fused
    from nf4_grouped import _lut, dequant_ref, gemm_4bit_grouped_captured
    from nf4_smallm import gemm_nf4_grouped_smallm, pair_lut
    dev = "cuda"
    k26 = _k26
    lut = pair_lut(dev)
    lut16 = _lut(dev)

    def ablate(x, packed, absmax, row0, rows, grp, order, dec):
        R = order.numel() if order is not None else x.shape[0]
        E_, N, kh = packed.shape
        assert (kh * 2) % PLAN["kc"] == 0, "the ablation copy is EVEN_K only"
        out = torch.empty(R, N, dtype=torch.bfloat16, device=dev)
        k26[(row0.numel(), triton.cdiv(N, PLAN["block_n"]))](
            x, order if order is not None else row0, packed, absmax, lut, lut16, row0, rows, grp, out,
            N, packed.stride(0), packed.stride(1), absmax.stride(0), absmax.stride(1),
            K=kh * 2, BLOCK_M=16, BLOCK_N=PLAN["block_n"], KC=PLAN["kc"], GATHER=order is not None, DEC=dec,
            num_warps=PLAN["warps"], num_stages=PLAN["stages"])
        return out

    def served_copy(xs, packed, absmax, row0, rows, grp, dec):
        E_, N, kh = packed.shape
        out = torch.empty(xs.shape[0], N, dtype=torch.bfloat16, device=dev)
        _k26s[(row0.numel(), triton.cdiv(N, 128))](
            xs, packed, absmax, out, lut16, row0, rows, grp, kh * 2, N, packed.stride(0), packed.stride(1),
            absmax.stride(0), absmax.stride(1), BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, DEC=dec, num_warps=4, num_stages=3)
        return out

    def capture(fn, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        torch.cuda.synchronize()
        return gr

    def graph_ms(fn):
        gr = capture(fn)
        ts = []
        for _ in range(ITERS):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            gr.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        del gr
        torch.cuda.empty_cache()
        return statistics.median(ts)

    def copy_gbps():
        src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
        dst = torch.empty_like(src)
        return 2 * src.numel() / (graph_ms(lambda: dst.copy_(src)) / 1e3) / 1e9

    t0 = time.time()
    rec = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "plan": PLAN, "load_plan": LOAD_PLAN,
           "B": B, "steps": STEPS, "quick": quick, "copy_gbps": copy_gbps(), "families": {}}
    print(f"copy {rec['copy_gbps']:.0f} GB/s", flush=True)
    for fam in families:
        c = FAMILIES[fam]
        E, k, L = c["E"], c["k"], (2 if quick else c["L"])
        g = torch.Generator(device=dev).manual_seed(26)
        WL = []
        for _li in range(L):
            Wl = {}
            for proj in ("gu", "dn"):
                N, K = c[proj]
                Wl[proj] = (torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                            torch.rand(E, N, K // 64, generator=g, device=dev) * 0.05 + 0.01)
            WL.append(Wl)
        xg = (torch.randn(B, c["gu"][1], generator=g, device=dev) / 4).to(torch.bfloat16).repeat_interleave(k, 0)
        xd = (torch.randn(B * k, c["dn"][1], generator=g, device=dev) / 4).to(torch.bfloat16)
        gcpu = torch.Generator().manual_seed(2026)
        steps = []
        for _s in range(STEPS):
            per_layer = []
            for _li in range(L):
                ids = torch.stack([torch.randperm(E, generator=gcpu)[:k] for _ in range(B)]).reshape(-1).to(torch.int32).to(dev)
                row0, rows, grp, order, _c = build_group_tiles_fused(ids, E, 16)
                per_layer.append((ids, row0, rows, grp, order, xg.index_select(0, order).contiguous()))
            steps.append(per_layer)

        def arm(name, sl):
            def f():
                for li, (ids, row0, rows, grp, order, xs) in enumerate(sl):
                    (gp, ga), (dp, da) = WL[li]["gu"], WL[li]["dn"]
                    if name == "pair":
                        gemm_nf4_grouped_smallm(xg, gp, ga, row0, rows, grp, order, **PLAN)
                        gemm_nf4_grouped_smallm(xd, dp, da, row0, rows, grp, None, **PLAN)
                    elif name == "load":
                        gemm_nf4_grouped_smallm(xg, gp, ga, row0, rows, grp, order, lut="load", **LOAD_PLAN)
                        gemm_nf4_grouped_smallm(xd, dp, da, row0, rows, grp, None, lut="load", **LOAD_PLAN)
                    elif name == "served":
                        gemm_4bit_grouped_captured(xs, gp, ga, row0, rows, grp, 16)
                        gemm_4bit_grouped_captured(xd, dp, da, row0, rows, grp, 16)
                    elif name in ("scopy", "stree"):
                        served_copy(xs, gp, ga, row0, rows, grp, 0 if name == "scopy" else 1)
                        served_copy(xd, dp, da, row0, rows, grp, 0 if name == "scopy" else 1)
                    else:
                        dec = {"copy": 0, "affine": 1, "noscale": 2, "bytes": 3, "tree": 4}[name]
                        ablate(xg, gp, ga, row0, rows, grp, order, dec)
                        ablate(xd, dp, da, row0, rows, grp, None, dec)
            return f

        # numerics on step 0, layer 0, gate_up: the product against the fp32 dequant oracle; the copy against the product
        ids, row0, rows, grp, order, xs = steps[0][0]
        gp, ga = WL[0]["gu"]
        y = gemm_nf4_grouped_smallm(xg, gp, ga, row0, rows, grp, order, **PLAN)
        yc = ablate(xg, gp, ga, row0, rows, grp, order, 0)
        yt = ablate(xg, gp, ga, row0, rows, grp, order, 4)
        yd = ablate(xd, *WL[0]["dn"], row0, rows, grp, None, 4)
        sids = ids.index_select(0, order)
        N, K = c["gu"]
        oracle = torch.stack([xg[int(order[i])].float().cpu() @ dequant_ref(gp[int(sids[i])].cpu(), ga[int(sids[i])].cpu(), N, K).t()
                              for i in range(8)])
        rel = float((y[:8].float().cpu() - oracle).abs().max() / oracle.abs().max())
        f = {"E": E, "k": k, "layers": L, "numerics": {"product_vs_oracle_rel": rel}, "numerics_ok": rel <= NUM_TOL,
             "copy_bit_equal": bool(torch.equal(y, yc)),
             "tree_bit_equal": bool(torch.equal(y, yt))
             and bool(torch.equal(yd, gemm_nf4_grouped_smallm(xd, *WL[0]["dn"], row0, rows, grp, None, **PLAN)))}
        ys = gemm_4bit_grouped_captured(xs, gp, ga, row0, rows, grp, 16)
        f["scopy_bit_equal"] = bool(torch.equal(ys, served_copy(xs, gp, ga, row0, rows, grp, 0)))
        f["stree_bit_equal"] = bool(torch.equal(ys, served_copy(xs, gp, ga, row0, rows, grp, 1)))
        bpe = sum(N_ * (K_ // 2) + N_ * (K_ // 64) * 4 for N_, K_ in (c["gu"], c["dn"]))
        f["floor_ms"] = {str(s): sum(torch.unique(pl[0]).numel() for pl in steps[s]) * bpe / (rec["copy_gbps"] * 1e9) * 1e3
                         for s in range(STEPS)}
        f["ms"] = {}
        for name in ARMS:
            try:
                f["ms"][name] = {str(s): graph_ms(arm(name, steps[s])) for s in range(STEPS)}
            except Exception as e:                                     # recorded, never dropped
                f["ms"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        f["median_ms"] = {a: statistics.median(v.values()) for a, v in f["ms"].items() if "error" not in v}
        f["median_floor_ms"] = statistics.median(f["floor_ms"].values())
        f["efficiency"] = {a: f["median_floor_ms"] / m for a, m in f["median_ms"].items()}
        med = f["median_ms"]
        f["ratios"] = {kk: med[a] / med[b] for kk, (a, b) in {
            "affine_over_pair": ("affine", "pair"), "noscale_over_affine": ("noscale", "affine"),
            "bytes_over_pair": ("bytes", "pair"), "served_over_pair": ("served", "pair"),
            "copy_over_pair": ("copy", "pair"), "load_over_pair": ("load", "pair"),
            "tree_over_pair": ("tree", "pair"), "scopy_over_served": ("scopy", "served"),
            "stree_over_served": ("stree", "served")}.items() if a in med and b in med}
        rec["families"][fam] = f
        WL = steps = None                                         # free this family's stores before the next
        torch.cuda.empty_cache()
        print(f"K26 {fam}: " + ", ".join(f"{a} {m:.3f}" for a, m in med.items()) + f" ms/step; floor {f['median_floor_ms']:.3f}; "
              + ", ".join(f"{kk} {v:.3f}" for kk, v in f["ratios"].items()), flush=True)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["verdict"], rec["verdict_reason"] = verdict(rec) if set(families) == set(FAMILIES) else ("VOID", "a subset of families ran")
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K26_VERDICT {rec['verdict']}: {rec['verdict_reason']}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--families", default="granite,olmoe")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    if not a.out:
        ap.error("out is required")
    sys.exit(main(a.out, a.families.split(","), quick=a.quick))
