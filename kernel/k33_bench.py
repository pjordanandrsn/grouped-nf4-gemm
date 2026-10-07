"""Lane K33 bench (``PREREG-k33-nf4-decode-gemv-bw.md``): does ``_gemv_nf4_bw`` (``GNF4_GEMV_BW=1``) run the NF4
single-row decode GEMV near the streaming ceiling on an RTX 5090, faster than the served route?

experts4bit-qlora's SV2 census put the served NF4 expert GEMV (``_gemv_nf4_dotpad``) at 2.469 ms of a 6.46 ms graphed
B=1 step on Qwen3-30B-A3B, 3.8x its byte floor; the NF4 families whose shapes dot-pad does not cover (Granite, OLMoE)
run the scalar GEMV at ~18 % of the ceiling. This bench replays one expert projection's decode launches over every
layer of a family in one CUDA graph, at the family's served shape (8 activation rows: top-8 of the layer's experts).

**Stores.** Per layer its own synthetic NF4 stack (random packed bytes, absmax in [0.5, 1.5)), seeded, so the graph
streams every layer's bytes and nothing stays in L2 (K24's lesson); 16 experts per layer, top-8 routed per layer from a
seeded draw. Families (``FAMILIES``): Qwen3-30B-A3B (48 layers; gate_up 1536 x 2048, down 2048 x 768),
Granite-3.1-3b-a800m (32; 1024 x 1536, 1536 x 512), OLMoE-1B-7B (16; 2048 x 2048, 2048 x 1024).

**Arms**, one graph each per (family, projection), every one with ``GNF4_PDL=0`` except ``bw_pdl``:
- ``incumbent``, ``incumbent2``: today's route (``GNF4_GEMV_BW=0``: dot-pad at Qwen3's shapes on >= 160-SM parts, the
  scalar GEMV elsewhere); the repeat is the instrument;
- ``scalar``: the certified scalar GEMV (``GNF4_GEMV_DOTPAD=0``);
- ``bw_tree``, ``bw_prmt32``: ``_gemv_nf4_bw`` with each decode, at the plan this run selected;
- ``bw_pdl``: ``bw_prmt32`` with ``GNF4_PDL=1`` (reported);
- ``int4``: ``quant_x_rows`` + ``gemv_int4_b32`` on int4-b32 stores of the same shape, the same bytes per parameter in
  another format (the named comparator; reported, not ruled).

**Plan selection** (K20's discipline), per projection: every plan in ``PLANS`` is captured and timed for ``SELECT_ROUNDS``;
plans that fail to compile, spill, or miss the numerics are not selectable; the fastest median wins, and the arms are
then timed afresh for ``ROUNDS`` (order reversed every round, CUDA events). **Numerics before timing**, at the selected
plan: ``bw_prmt32`` bitwise ``bw_tree`` on 8 rows, and both within the tolerance contract of the fp32 reference (error no
more than 5 % above the scalar route's, or <= 2^-9 of the output scale). **Engagement:** ``dispatch_counts()`` across
each capture shows the registered route. **Floor:** active bytes / the box's own copy bandwidth (a 512 MiB copy).

    python k33_bench.py out.json
    python k33_bench.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

FAMILIES = {"qwen3": {"layers": 48, "gate_up": (1536, 2048), "down": (2048, 768)},
            "granite": {"layers": 32, "gate_up": (1024, 1536), "down": (1536, 512)},
            "olmoe": {"layers": 16, "gate_up": (2048, 2048), "down": (2048, 1024)}}
PROJ = ("gate_up", "down")
E_PER_LAYER, ROWS = 16, 8
ARMS = ("incumbent", "incumbent2", "scalar", "bw_tree", "bw_prmt32", "bw_pdl", "int4")
PLANS = [(16, 256, 4, 1), (16, 512, 4, 1), (16, 1024, 4, 1), (16, 256, 8, 1), (16, 512, 8, 1),
         (32, 256, 4, 1), (32, 512, 4, 1), (32, 512, 8, 1), (32, 1024, 8, 1), (16, 256, 4, 2), (16, 512, 4, 2),
         (32, 256, 4, 2)]
WARM, ROUNDS, SELECT_ROUNDS = 20, 200, 40
SELF_LO, SELF_HI = 0.98, 1.02
SHAPE_MAX, PAIR_LEVER, PAIR_PARTIAL, FLOOR_MIN = 0.77, 0.667, 0.80, 0.45
TOL_X, TOL_ABS = 1.05, 2.0 ** -9
CARD = "5090"


def nf4_bytes(n, k):
    """Bytes one row reads per launch: packed nibbles + fp32 absmax."""
    return n * (k // 2 + (k // 64) * 4)


def verdict(r):
    """The registered rule, on the Qwen3 projections. The first that applies:
    VOID           a timing or the copy floor is missing; the card is not the registered RTX 5090; a capture's tally is not
                   its registered route; no plan was selectable for a Qwen3 projection.
    NOISY          incumbent2 / incumbent outside [0.98, 1.02] on either Qwen3 projection.
    FUNCTION_FAIL  at any family's selected plan, bw_prmt32 is not bitwise bw_tree, or a bw decode misses the tolerance.
    LEVER          with bw = the faster contract-exact decode on the pair: bw / incumbent <= 0.77 on each Qwen3 projection,
                   <= 0.667 on the pair, and floor / bw >= 0.45 on each.
    PARTIAL        the pair <= 0.80, but not LEVER.
    NO_LEVER       otherwise."""
    if CARD not in str(r.get("device", "")):
        return "VOID", f"card {r.get('device')!r} is not the registered RTX {CARD}"
    if not r.get("copy_gbps"):
        return "VOID", "the copy floor was not measured"
    ms = r.get("median_ms") or {}
    q = {p: ms.get(f"qwen3/{p}") or {} for p in PROJ}
    missing = [f"qwen3/{p}/{a}" for p in PROJ for a in ARMS if not q[p].get(a)]
    if missing:
        return "VOID", f"timings missing: {missing}"
    if not all((r.get("plan") or {}).get(f"qwen3/{p}") for p in PROJ):
        return "VOID", "no plan was selectable for a Qwen3 projection"
    bad_tally = [k for k, v in (r.get("tally_ok") or {}).items() if v is not True]
    if bad_tally or not r.get("tally_ok"):
        return "VOID", f"a capture did not run its registered route: {bad_tally or 'none recorded'}"
    noisy = {p: q[p]["incumbent2"] / q[p]["incumbent"] for p in PROJ}
    if any(not SELF_LO <= v <= SELF_HI for v in noisy.values()):
        return "NOISY", f"incumbent2 / incumbent = {noisy}, outside [{SELF_LO}, {SELF_HI}]"
    nb = [k for k, v in (r.get("numerics") or {}).items() if v.get("bitwise") is not True or v.get("tolerance") is not True]
    if nb or not r.get("numerics"):
        return "FUNCTION_FAIL", f"numerics failed at {nb or 'none recorded'}"
    pair = {d: q["gate_up"][d] + q["down"][d] for d in ("bw_tree", "bw_prmt32")}
    best = min(pair, key=pair.get)
    inc_pair = q["gate_up"]["incumbent"] + q["down"]["incumbent"]
    shape = {p: q[p][best] / q[p]["incumbent"] for p in PROJ}
    floor = {p: (r["floor_ms"]["qwen3/" + p]) / q[p][best] for p in PROJ}
    pr = pair[best] / inc_pair
    txt = (f"bw={best}: gate_up x{shape['gate_up']:.3f}, down x{shape['down']:.3f}, pair x{pr:.3f} of the incumbent; "
           f"floor/bw {floor['gate_up']:.2f} / {floor['down']:.2f}; instrument {noisy['gate_up']:.4f} / {noisy['down']:.4f}")
    if all(v <= SHAPE_MAX for v in shape.values()) and pr <= PAIR_LEVER and all(v >= FLOOR_MIN for v in floor.values()):
        return "LEVER", txt + " -- next: the served lane (experts4bit-qlora P116)"
    if pr <= PAIR_PARTIAL:
        return "PARTIAL", txt + " -- opt-in ships; a second round is registered"
    return "NO_LEVER", txt + " -- GNF4_GEMV_BW stays opt-in"


def self_test():
    def rec(inc=(27.0, 18.8), inc2=None, bw=(14.0, 8.0), tree=(22.0, 12.0), floor=(9.0, 4.5), card="NVIDIA GeForce RTX 5090",
            gbps=1570.0, tally=True, bit=True, tol=True, plan=True, drop=None):
        inc2 = inc2 or inc
        ms = {}
        for i, p in enumerate(PROJ):
            ms[f"qwen3/{p}"] = {"incumbent": inc[i], "incumbent2": inc2[i], "scalar": 60.0, "bw_tree": tree[i],
                                "bw_prmt32": bw[i], "bw_pdl": bw[i] * 0.98, "int4": 12.0}
        if drop:
            ms["qwen3/gate_up"].pop(drop)
        return {"device": card, "copy_gbps": gbps, "median_ms": ms,
                "floor_ms": {"qwen3/gate_up": floor[0], "qwen3/down": floor[1]},
                "plan": {"qwen3/gate_up": [16, 256, 4, 1] if plan else None, "qwen3/down": [16, 256, 4, 1]},
                "tally_ok": {"qwen3/gate_up/bw_prmt32": tally},
                "numerics": {"qwen3/gate_up": {"bitwise": bit, "tolerance": tol}}}
    cases = [
        ("lever", verdict(rec())[0] == "LEVER"),
        ("tree is the faster decode", "bw=bw_tree" in verdict(rec(bw=(22.0, 12.0), tree=(14.0, 8.0)))[1]),
        ("partial", verdict(rec(bw=(21.5, 14.5)))[0] == "PARTIAL"),
        ("no lever", verdict(rec(bw=(26.0, 17.0), tree=(26.5, 17.5)))[0] == "NO_LEVER"),
        ("one shape misses", verdict(rec(bw=(12.0, 15.0)))[0] == "PARTIAL"),
        ("floor too far", verdict(rec(floor=(5.0, 2.0)))[0] == "PARTIAL"),
        ("wrong card", verdict(rec(card="NVIDIA RTX A2000 12GB"))[0] == "VOID"),
        ("no floor", verdict(rec(gbps=None))[0] == "VOID"),
        ("missing arm", verdict(rec(drop="int4"))[0] == "VOID"),
        ("no plan", verdict(rec(plan=False))[0] == "VOID"),
        ("tally", verdict(rec(tally=False))[0] == "VOID"),
        ("noisy", verdict(rec(inc2=(28.0, 18.8)))[0] == "NOISY"),
        ("bitwise", verdict(rec(bit=False))[0] == "FUNCTION_FAIL"),
        ("tolerance", verdict(rec(tol=False))[0] == "FUNCTION_FAIL"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"k33_bench self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(out_path, quick=False):
    import torch
    import triton

    import int4_b32
    import nf4_grouped as ng
    dev = "cuda"
    t_start = time.time()
    rounds, select_rounds, plans = (20, 5, PLANS[:2]) if quick else (ROUNDS, SELECT_ROUNDS, PLANS)
    rec = {"device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "triton": triton.__version__, "families": FAMILIES, "rows": ROWS, "experts_per_layer": E_PER_LAYER,
           "rounds": rounds, "select_rounds": select_rounds, "plans": plans, "quick": quick,
           "median_ms": {}, "quartiles_ms": {}, "floor_ms": {}, "plan": {}, "plan_ms": {}, "plan_refused": {},
           "tally_ok": {}, "numerics": {}}
    g = torch.Generator(device=dev).manual_seed(33)

    # ---- the copy floor: this box's streaming bandwidth
    src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        dst.copy_(src)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    rec["copy_gbps"] = 2 * src.numel() / (statistics.median(ts) * 1e6)    # read + write
    del src, dst

    def env(**kv):
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        int4_b32.pdl_refresh()

    ARM_ENV = {"incumbent": {"GNF4_GEMV_BW": "0", "GNF4_GEMV_DOTPAD": None, "GNF4_PDL": "0"},
               "incumbent2": {"GNF4_GEMV_BW": "0", "GNF4_GEMV_DOTPAD": None, "GNF4_PDL": "0"},
               "scalar": {"GNF4_GEMV_BW": "0", "GNF4_GEMV_DOTPAD": "0", "GNF4_PDL": "0"},
               "bw_tree": {"GNF4_GEMV_BW": "1", "GNF4_GEMV_BW_DECODE": "tree", "GNF4_PDL": "0"},
               "bw_prmt32": {"GNF4_GEMV_BW": "1", "GNF4_GEMV_BW_DECODE": "prmt32", "GNF4_PDL": "0"},
               "bw_pdl": {"GNF4_GEMV_BW": "1", "GNF4_GEMV_BW_DECODE": "prmt32", "GNF4_PDL": "1"}}

    def tally_ok(arm, tally, n, k, layers):
        """The capture ran its registered route on every layer and nothing else (a split-K launch counts once under its
        route's split key; the bw route also bumps ``bw_splitk`` beside its decode key)."""
        if arm.startswith("bw_"):
            d = "bw_tree" if arm == "bw_tree" else "bw_prmt32"
            return tally.get(d) == layers and all(v == 0 for r_, v in tally.items() if r_ not in (d, "bw_splitk"))
        dot = arm != "scalar" and (n, k) in ng._DOTPAD_CONFIGS and rec["sm_count"] >= 160
        routes = ("dotpad", "dotpad_splitk") if dot else ("scalar", "scalar_splitk")
        return sum(tally.get(r_, 0) for r_ in routes) == layers and sum(tally.values()) == layers

    def time_graphs(graphs, n_rounds):
        times = {a: [] for a in graphs}
        order = list(graphs)
        for rnd in range(WARM + n_rounds):
            for a_ in (order if rnd % 2 == 0 else order[::-1]):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                graphs[a_].replay()
                e.record()
                e.synchronize()
                if rnd >= WARM:
                    times[a_].append(s.elapsed_time(e))
        return times

    for fam, spec in FAMILIES.items():
        L = spec["layers"]
        for proj in PROJ:
            N, K = spec[proj]
            key = f"{fam}/{proj}"
            stores = [(torch.randint(0, 256, (E_PER_LAYER, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                       torch.rand(E_PER_LAYER, N, K // 64, generator=g, device=dev) + 0.5) for _ in range(L)]
            i4 = [(torch.randint(0, 256, (E_PER_LAYER, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                   (torch.rand(E_PER_LAYER, N, K // 32, generator=g, device=dev) * 0.004 + 0.002).to(torch.float16))
                  for _ in range(L)]
            eids = [torch.randperm(E_PER_LAYER, generator=g, device=dev)[:ROWS].to(torch.int32) for _ in range(L)]
            x = (torch.randn(ROWS, K, generator=g, device=dev)).to(torch.bfloat16)
            rec["floor_ms"][key] = L * ROWS * nf4_bytes(N, K) / (rec["copy_gbps"] * 1e6)

            def nf4_chain(plan=None):
                return [ng.gemm_4bit_grouped(x, B, A, [1] * ROWS, ids, bw_config=plan)
                        for (B, A), ids in zip(stores, eids)]

            def int4_chain():
                out = []
                for (P, S), ids in zip(i4, eids):
                    xq, xs = int4_b32.quant_x_rows(x)
                    out.append(int4_b32.gemv_int4_b32(xq, xs, P, S, ids, N, K))
                return out

            def capture(fn, arm):
                fn()                                                     # warm (compile) outside the capture
                torch.cuda.synchronize()
                ng.reset_dispatch_counts()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    outs = fn()
                torch.cuda.synchronize()
                return gr, outs, ng.dispatch_counts()

            # ---- plan selection for the bw route (prmt32), then numerics at the selected plan
            cand = {}
            for plan in plans:
                env(**ARM_ENV["bw_prmt32"])
                try:
                    gr, _o, _t = capture(lambda p=plan: nf4_chain(p), "bw_prmt32")
                except Exception as exc:                                 # noqa: BLE001 -- a plan that cannot run is not selectable
                    rec["plan_refused"].setdefault(key, []).append([list(plan), f"{type(exc).__name__}: {str(exc)[:120]}"])
                    continue
                cand[plan] = gr
            if not cand:
                continue
            sel_t = time_graphs({p: gr for p, gr in cand.items()}, select_rounds)
            rec["plan_ms"][key] = {",".join(map(str, p)): statistics.median(v) for p, v in sel_t.items()}
            plan = min(sel_t, key=lambda p: statistics.median(sel_t[p]))
            rec["plan"][key] = list(plan)
            del cand

            # numerics on layer 0 at the selected plan: prmt32 vs tree bitwise, both vs the fp32 reference
            B0, A0 = stores[0]
            ids0 = eids[0]
            outs = {}
            for d in ("bw_tree", "bw_prmt32"):
                env(**ARM_ENV[d])
                outs[d] = ng.gemm_4bit_grouped(x, B0, A0, [1] * ROWS, ids0, bw_config=plan)
            env(**ARM_ENV["scalar"])
            outs["scalar"] = ng.gemm_4bit_grouped(x, B0, A0, [1] * ROWS, ids0)
            ref = torch.stack([ng.dequant_ref(B0[e], A0[e], N, K).float() @ x[r].float()
                               for r, e in enumerate(ids0.tolist())])
            scale = ref.abs().max().clamp_min(1e-6)
            err = {d: ((outs[d].float() - ref).abs() / scale).max().item() for d in outs}
            rec["numerics"][key] = {"bitwise": bool(torch.equal(outs["bw_tree"], outs["bw_prmt32"])), "err": err,
                                    "tolerance": all(err[d] <= max(TOL_X * err["scalar"], TOL_ABS)
                                                     for d in ("bw_tree", "bw_prmt32"))}

            # ---- the arms, timed afresh
            graphs = {}
            for arm in ARMS:
                if arm == "int4":
                    env(GNF4_PDL="0")
                    gr, _o, _t = capture(int4_chain, arm)
                    rec["tally_ok"][f"{key}/{arm}"] = True
                else:
                    env(**ARM_ENV[arm])
                    gr, _o, tally = capture(lambda a=arm: nf4_chain(plan if a.startswith("bw_") else None), arm)
                    rec["tally_ok"][f"{key}/{arm}"] = tally_ok(arm, tally, N, K, L)
                graphs[arm] = gr
            t = time_graphs(graphs, rounds)
            rec["median_ms"][key] = {a: statistics.median(v) for a, v in t.items()}
            rec["quartiles_ms"][key] = {a: statistics.quantiles(v, n=4) for a, v in t.items()}
            print(f"K33 {key} plan {plan} median ms " + json.dumps({a: round(v, 4) for a, v in rec["median_ms"][key].items()})
                  + f" floor {rec['floor_ms'][key]:.4f} numerics {json.dumps(rec['numerics'][key])}", flush=True)
            graphs = stores = i4 = None                                  # release this projection's stacks
            torch.cuda.empty_cache()

    env(GNF4_GEMV_BW=None, GNF4_GEMV_BW_DECODE=None, GNF4_GEMV_DOTPAD=None, GNF4_PDL=None)
    rec["wall_s"] = round(time.time() - t_start, 1)
    rec["verdict"], rec["verdict_reason"] = verdict(rec)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K33_VERDICT {rec['verdict']}: {rec['verdict_reason']}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quick", action="store_true", help="a rehearsal: two plans, 20 rounds (never a reading)")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.out:
        ap.error("out is required")
    sys.exit(main(a.out, quick=a.quick))
