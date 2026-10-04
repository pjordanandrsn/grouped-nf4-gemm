"""Lane K28 bench (``PREREG-k28-pdl-decode-chain.md``): does programmatic dependent launch (``GNF4_PDL=1``) shorten a
CUDA-graph replay of the served B=1 decode layer's gnf4 kernels on an RTX 5090, bit-identically?

experts4bit-qlora's SC1b census (#846) read e4b's B=1 decode step on an RTX 5090 at the same summed kernel work as
llama.cpp's (4.09 vs 4.14 ms) with none of its 1,550 in-graph kernels overlapping, where llama.cpp overlaps 95.5 % of
consecutive pairs and hides about 1.38 ms a step. 913 of e4b's kernels are this repository's decode-row kernels. This
bench replays those kernels, in the served layer's order and at Qwen3-30B-A3B's B=1 shapes, over 48 layers in one
CUDA graph, with the switch off and on.

**The layer** (the served step's order, from SC1b's node census; e4b's own attention kernels and the cuBLAS router
GEMM are not gnf4's and are left out):

    rmsnorm_rows -> [quant -> gemv -> reduce](qkv) -> rope_norm_heads(q) -> rope_norm_heads(k)
      -> [quant -> gemv -> reduce](o) -> rmsnorm_resid_rows -> router_epilogue -> index_select (ATen, the expert rows)
      -> [quant -> gemv -> reduce](gate_up, R = 8) -> swiglu_rows -> [quant -> gemv -> reduce](down, R = 8)
      -> combine_rows -> residual add (ATen)

19 gnf4 kernels and 2 ATen kernels per layer; 912 gnf4 kernels over 48 layers. Each layer reads the one before, so a
stale read would change the final residual stream.

**Graphs**, each captured once from its own build:
- ``chain_off``, ``chain_on``, ``chain_off2``: the layer above. ``chain_off2`` repeats ``chain_off`` (the instrument).
- ``glued_off``, ``glued_on``: the same plus two ATen kernels per layer where the served step leaves gnf4: an attention
  stand-in (a copy) and a router GEMM (``torch.mm``), whose output the router epilogue reads. Reported, not ruled on.

**Timing.** ``ROUNDS`` rounds after ``WARM``; each round replays all five graphs once, in an order that reverses every
round, each timed by CUDA events and synchronised. Medians per graph.

**Checks.**
- *Bitwise:* every recorded output (each layer's residual stream, k, the router's three outputs, and the glued router
  logits) of ``chain_on`` and ``chain_off2`` equals ``chain_off``'s; ``glued_on`` equals ``glued_off``; the eager
  chain equals itself with the switch off and on, and equals ``chain_off``.
- *Launch accounting:* a Triton launch hook counts each capture's gnf4 launches and how many carried ``launch_pdl``.
- *Engagement probe:* a 20 us spin kernel releases its dependents at once and stamps the clock last; a dependent stamps
  the clock before and after its wait. Five graph replays with PDL and five without.

    python k28_bench.py out.json
    python k28_bench.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

H, I, E, TOP, D, QH, KH, L = 2048, 768, 16, 8, 128, 32, 4, 48
NQ = (QH + 2 * KH) * D
GNF4_PER_LAYER = 19
GNF4_LAUNCHES = GNF4_PER_LAYER * L                     # 912
CHAIN_KERNELS = ("_rmsnorm_rows", "_quant_x_rows", "_gemv_int4_b32", "_reduce_partials", "_rope_norm_heads",
                 "_rmsnorm_resid_rows", "_router_epilogue", "_swiglu_rows", "_combine_rows")
GRAPHS = ("chain_off", "chain_on", "chain_off2", "glued_off", "glued_on")
WARM, ROUNDS = 20, 200
SPIN_NS = 20_000
PROBE_REPS = 5
SAVE_MIN_US = 0.25          # chain saving per gnf4 kernel, us
SELF_LO, SELF_HI = 0.98, 1.02
MIN_CC = (9, 0)

try:
    import triton
    import triton.language as tl
except ImportError:                                    # the self-test needs neither
    triton = tl = None

if triton is not None:
    @triton.jit
    def _probe_spin(out_ptr, NS: tl.constexpr, PDL: tl.constexpr):
        """Releases its dependents at once, spins NS ns on the global clock, stamps the clock last."""
        t0 = tl.extra.cuda.globaltimer()
        if PDL:
            tl.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=tl.int32,
                                      is_pure=False, pack=1)
        t = t0
        while t - t0 < NS:
            t = tl.extra.cuda.globaltimer()
        tl.store(out_ptr, t)

    @triton.jit
    def _probe_after(out_ptr, PDL: tl.constexpr):
        """Stamps the clock on entry and again after its wait."""
        t_pre = tl.extra.cuda.globaltimer()
        if PDL:
            tl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False,
                                      pack=1)
        t_post = tl.extra.cuda.globaltimer()
        tl.store(out_ptr + 1, t_pre)
        tl.store(out_ptr + 2, t_post)


def verdict(r):
    """The registered rule. The first that applies:
    VOID           a graph's timing is missing; the card is below sm_90; a capture's gnf4 launches are not 912, or an on
                   capture launched any without PDL, or an off capture any with it; the probe's dependent did not start
                   at least half the spin early in every PDL replay, or started early in any replay without PDL.
    FUNCTION_FAIL  a bitwise check failed, or a probe dependent left its wait before its primary's last stamp.
    NOISY          chain_off2 / chain_off outside [0.98, 1.02].
    LEVER          the chain saves >= 0.25 us per gnf4 kernel: (chain_off - chain_on) / 912.
    NO_LEVER       otherwise."""
    ms = r.get("median_ms") or {}
    if any(not ms.get(g) for g in GRAPHS):
        return "VOID", f"graph timings missing: {[g for g in GRAPHS if not ms.get(g)]}"
    if tuple(r.get("cc") or (0, 0)) < MIN_CC:
        return "VOID", f"compute capability {r.get('cc')} is below sm_90: PDL cannot engage"
    acct = r.get("launches") or {}
    for g in GRAPHS:
        a = acct.get(g) or {}
        want_pdl = GNF4_LAUNCHES if g.endswith("_on") else 0
        if a.get("gnf4") != GNF4_LAUNCHES or a.get("pdl") != want_pdl:
            return "VOID", f"{g}: launched {a.get('gnf4')} gnf4 kernels, {a.get('pdl')} with PDL (registered {GNF4_LAUNCHES}, {want_pdl})"
    pr = r.get("probe") or {}
    on, off = pr.get("on") or [], pr.get("off") or []
    if len(on) != PROBE_REPS or len(off) != PROBE_REPS:
        return "VOID", "probe replays missing"
    if not all(x["early_ns"] >= SPIN_NS // 2 for x in on):
        return "VOID", f"PDL did not engage: the dependent started {[x['early_ns'] for x in on]} ns before the spin ended"
    if any(x["early_ns"] > 0 for x in off):
        return "VOID", f"instrument: without PDL the dependent started early {[x['early_ns'] for x in off]}"
    bad = [k for k, v in (r.get("bitwise") or {}).items() if v is not True]
    if bad or not r.get("bitwise"):
        return "FUNCTION_FAIL", f"bitwise checks failed: {bad or 'none recorded'}"
    if not all(x["wait_ok"] for x in on + off):
        return "FUNCTION_FAIL", "a probe dependent left its wait before its primary's last stamp"
    n = ms["chain_off2"] / ms["chain_off"]
    if not SELF_LO <= n <= SELF_HI:
        return "NOISY", f"chain_off2 / chain_off = {n:.4f}, outside [{SELF_LO}, {SELF_HI}]"
    s = (ms["chain_off"] - ms["chain_on"]) * 1000.0 / GNF4_LAUNCHES
    sg = (ms["glued_off"] - ms["glued_on"]) * 1000.0 / GNF4_LAUNCHES
    txt = (f"chain {ms['chain_off']:.4f} -> {ms['chain_on']:.4f} ms (x{ms['chain_off'] / ms['chain_on']:.3f}), "
           f"{s:.3f} us per gnf4 kernel; glued {ms['glued_off']:.4f} -> {ms['glued_on']:.4f} ms, {sg:.3f} us; "
           f"instrument {n:.4f}")
    if s >= SAVE_MIN_US:
        return "LEVER", txt + f" -- >= {SAVE_MIN_US} us: next, a served lane reads GNF4_PDL=1 on e4b's decode step"
    return "NO_LEVER", txt + f" -- < {SAVE_MIN_US} us: GNF4_PDL stays opt-in"


def self_test():
    def rec(off=3.0, on=2.7, off2=3.01, goff=3.4, gon=3.2, cc=(12, 0), acct=None, probe_on=15000, probe_off=-300,
            wait=True, bit=True, drop=None):
        ms = {"chain_off": off, "chain_on": on, "chain_off2": off2, "glued_off": goff, "glued_on": gon}
        if drop:
            ms.pop(drop)
        a = {g: {"gnf4": GNF4_LAUNCHES, "pdl": GNF4_LAUNCHES if g.endswith("_on") else 0} for g in GRAPHS}
        a.update(acct or {})
        return {"median_ms": ms, "cc": cc, "launches": a,
                "probe": {"on": [{"early_ns": probe_on, "wait_ok": wait}] * PROBE_REPS,
                          "off": [{"early_ns": probe_off, "wait_ok": True}] * PROBE_REPS},
                "bitwise": {"chain_on": bit, "chain_off2": True, "glued_on": True, "eager_on": True, "eager_graph": True}}
    cases = [
        ("lever", verdict(rec())[0] == "LEVER"),
        ("boundary", verdict(rec(on=3.0 - 0.25 * GNF4_LAUNCHES / 1000.0))[0] == "LEVER"),
        ("no lever", verdict(rec(on=2.9))[0] == "NO_LEVER"),
        ("below sm_90", verdict(rec(cc=(8, 6)))[0] == "VOID"),
        ("missing graph", verdict(rec(drop="glued_on"))[0] == "VOID"),
        ("accounting", verdict(rec(acct={"chain_on": {"gnf4": GNF4_LAUNCHES, "pdl": 900}}))[0] == "VOID"),
        ("not engaged", verdict(rec(probe_on=200))[0] == "VOID"),
        ("instrument early", verdict(rec(probe_off=5000))[0] == "VOID"),
        ("bitwise", verdict(rec(bit=False))[0] == "FUNCTION_FAIL"),
        ("wait broken", verdict(rec(wait=False))[0] == "FUNCTION_FAIL"),
        ("noisy", verdict(rec(off2=3.1))[0] == "NOISY"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"k28_bench self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(out_path):
    import torch

    import int4_b32 as m
    dev = "cuda"
    t_start = time.time()
    rec = {"device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()),
           "torch": torch.__version__, "triton": triton.__version__,
           "shapes": {"H": H, "I": I, "E": E, "top_k": TOP, "D": D, "q_heads": QH, "kv_heads": KH, "layers": L},
           "rounds": ROUNDS, "warm": WARM, "gnf4_per_layer": GNF4_PER_LAYER}

    # ---- weights and inputs: per-layer synthetic int4 stores (random bytes, small scales), seeded
    g = torch.Generator(device=dev).manual_seed(28)

    def store(e, n, k):
        return (torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, generator=g, device=dev),
                (torch.rand(e, n, k // 32, generator=g, device=dev) * 0.004 + 0.002).to(torch.float16))

    def bf(*s, scale=1.0):
        return (torch.randn(*s, generator=g, device=dev) * scale).to(torch.bfloat16)
    LY = []
    for _l in range(L):
        LY.append({"qkv": store(1, NQ, H), "o": store(1, H, QH * D), "gu": store(E, 2 * I, H), "dn": store(E, H, I),
                   "ln1": 1 + bf(H, scale=0.1), "ln2": 1 + bf(H, scale=0.1), "qn": 1 + bf(D, scale=0.1),
                   "kn": 1 + bf(D, scale=0.1), "logits": torch.randn(1, E, generator=g, device=dev),
                   "rw": bf(H, E, scale=H ** -0.5)})
    x0 = bf(1, H)
    cos, sin = bf(1, D), bf(1, D)
    e0 = torch.zeros(1, dtype=torch.int32, device=dev)
    row_token = torch.zeros(TOP, dtype=torch.long, device=dev)

    def trio(x, st, eids, n, k):
        xq, xs = m.quant_x_rows(x)
        return m.gemv_int4_b32(xq, xs, st[0], st[1], eids, n, k)

    def layers(glued):
        outs = []
        x = x0
        for ly in LY:
            h = m.rmsnorm_rows(x, ly["ln1"], 1e-6)
            qkv = trio(h, ly["qkv"], e0, NQ, H)
            q = m.rope_norm_heads(qkv[:, :QH * D].view(1, QH, D), ly["qn"], cos, sin, 1e-6)
            k = m.rope_norm_heads(qkv[:, QH * D:(QH + KH) * D].view(1, KH, D), ly["kn"], cos, sin, 1e-6)
            a = q.view(1, QH * D)
            if glued:
                a = torch.empty_like(a).copy_(a)                        # the attention stand-in (ATen)
            o = trio(a, ly["o"], e0, H, QH * D)
            normed, resid = m.rmsnorm_resid_rows(o, x, ly["ln2"], 1e-6)
            logits = torch.mm(normed, ly["rw"]) if glued else ly["logits"]   # the router GEMM stand-in (ATen)
            first, w, idx = m.router_epilogue(logits, TOP, True)
            xr = normed.index_select(0, row_token)                       # the expert rows (ATen, as served)
            eids = idx.view(-1)
            gu = trio(xr, ly["gu"], eids, 2 * I, H)
            sw = m.swiglu_rows(gu)
            dn = trio(sw, ly["dn"], eids, H, I)
            moe = m.combine_rows(dn, w, TOP)
            x = resid + moe                                              # the residual add (ATen, as served)
            outs += [x, k, first, w, idx] + ([logits] if glued else [])
        return outs

    rec["pdl_active"] = {}

    def setenv(on, tag):
        """Sets the switch and records whether gnf4 reads it as active here (below sm_90 it never is: the
        accounting check then reads VOID rather than this bench refusing, so a rehearsal runs to the end)."""
        os.environ[m.PDL_ENV] = "1" if on else "0"
        m.pdl_refresh()                                    # the switch is read once; re-read it for this build
        rec["pdl_active"][tag] = m.pdl_active(dev)

    # ---- launch accounting: a Triton launch hook records each capture's gnf4 launches
    seen = []

    def hook(md):
        seen.append((md.data["name"], md.data["function"]))
    fn_pdl = {}

    def count(launches):
        for name in CHAIN_KERNELS:
            for _d, (cache, *_r) in getattr(m, name).device_caches.items():
                for ck in cache.values():
                    fn_pdl[ck.function] = bool(getattr(ck.metadata, "launch_pdl", False))
        ours = [(n, f) for n, f in launches if n in CHAIN_KERNELS]
        return {"gnf4": len(ours), "pdl": sum(fn_pdl.get(f, False) for _n, f in ours),
                "other_triton": len(launches) - len(ours)}

    # ---- eager references (they also compile both variants outside any capture)
    setenv(False, "eager_off")
    eager_off = layers(False)
    layers(True)
    setenv(True, "eager_on")
    eager_on = layers(False)
    layers(True)
    torch.cuda.synchronize()

    graphs, outs = {}, {}
    triton.knobs.runtime.launch_enter_hook = hook
    try:
        rec["launches"] = {}
        for gname in GRAPHS:
            setenv(gname.endswith("_on"), gname)
            glued = gname.startswith("glued")
            layers(glued)                                                # warm this build
            torch.cuda.synchronize()
            seen.clear()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                outs[gname] = layers(glued)
            torch.cuda.synchronize()
            rec["launches"][gname] = count(list(seen))
            graphs[gname] = gr
    finally:
        triton.knobs.runtime.launch_enter_hook = None
    os.environ[m.PDL_ENV] = "0"
    m.pdl_refresh()
    for gr in graphs.values():
        gr.replay()
    torch.cuda.synchronize()

    def same(a, b):
        return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    rec["bitwise"] = {"chain_on": same(outs["chain_on"], outs["chain_off"]),
                      "chain_off2": same(outs["chain_off2"], outs["chain_off"]),
                      "glued_on": same(outs["glued_on"], outs["glued_off"]),
                      "eager_on": same(eager_on, eager_off),
                      "eager_graph": same(eager_off, outs["chain_off"])}
    rec["final_resid_abs_mean"] = float(outs["chain_off"][-5].float().abs().mean())
    print("K28 launches " + json.dumps(rec["launches"]) + " bitwise " + json.dumps(rec["bitwise"]), flush=True)

    # ---- timing
    times = {g: [] for g in GRAPHS}
    order = list(GRAPHS)
    for rnd in range(WARM + ROUNDS):
        for gname in (order if rnd % 2 == 0 else order[::-1]):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            graphs[gname].replay()
            b.record()
            b.synchronize()
            if rnd >= WARM:
                times[gname].append(a.elapsed_time(b))
    rec["median_ms"] = {g: statistics.median(v) for g, v in times.items()}
    rec["quartiles_ms"] = {g: statistics.quantiles(v, n=4) for g, v in times.items()}
    rec["times_ms"] = {g: [round(t, 5) for t in v] for g, v in times.items()}
    print("K28 median ms " + json.dumps({g: round(v, 4) for g, v in rec["median_ms"].items()}), flush=True)

    # ---- engagement probe (its kernels are module-level, above)
    def probe(pdl):
        buf = torch.zeros(3, dtype=torch.int64, device=dev)
        kw = {"launch_pdl": True} if pdl else {}

        def step():
            _probe_spin[(1,)](buf, NS=SPIN_NS, PDL=pdl)
            _probe_after[(1,)](buf, PDL=pdl, **kw)
        step()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            step()
        rows = []
        for _ in range(PROBE_REPS):
            gr.replay()
            torch.cuda.synchronize()
            end, pre, post = buf.tolist()
            rows.append({"early_ns": end - pre, "wait_ok": post >= end, "end": end, "pre": pre, "post": post})
        return rows
    # griddepcontrol does not assemble below sm_90, so a rehearsal there records no PDL probe (and reads VOID by cc)
    can = tuple(rec["cc"]) >= MIN_CC
    rec["probe"] = {"on": probe(True) if can else [], "off": probe(False), "spin_ns": SPIN_NS}
    print("K28 probe early_ns on " + str([x["early_ns"] for x in rec["probe"]["on"]]) + " off "
          + str([x["early_ns"] for x in rec["probe"]["off"]]), flush=True)

    rec["wall_s"] = round(time.time() - t_start, 1)
    rec["verdict"], rec["verdict_reason"] = verdict(rec)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K28_VERDICT {rec['verdict']}: {rec['verdict_reason']}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.out:
        ap.error("out is required")
    sys.exit(main(a.out))
