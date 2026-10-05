"""K30's reducer and decision rule (kernel/PREREG-k30-splitk-r-term-l4.md): does the int4-b32 split-K R term win on a
<= 64-SM card, read from the 48-cell `sk_sweep.py` grid run twice on one rented NVIDIA L4?

    python k30_reduce.py --self-test
    python k30_reduce.py <rows-dir> <out.json> [--check-installed-plan]

`<rows-dir>` holds `k30p1_<family>_<proj>.json` and `k30p2_<family>_<proj>.json` (two passes of `sk_sweep.py`, each
shape in its own process, OUT_PREFIX=k30p1 / k30p2). Every number the rule reads comes from those files; nothing is
derived from the code's arithmetic except WHICH sk each plan picks, and that pick is cross-checked against the
installed `int4_b32._plan` on the box (`--check-installed-plan`).

The rule, fixed before the data (the prereg's "Rule"):
  * The 24 cells at R >= SPLITK_R_FLOOR (16) are where the R term acts. Below it the two plans pick the same sk by
    construction; the reducer asserts that rather than assuming it.
  * Per cell c, pooled over both passes: r_c = (new_p1 + new_p2) / (old_p1 + old_p2), where new is the sweep's time at
    the R-aware plan's sk for this card's SM count and old is its time at the N-only plan's sk.
  * S = sum over cells of (new_p1 + new_p2) / sum of (old_p1 + old_p2): the summed-time ratio. W = max over cells of r_c.
  * VOID if: the card is not an NVIDIA L4 or has more than 64 SMs; a cell is missing or a plan's sk was not swept; the
    two passes disagree on the card; or the instrument is unstable, |S_p1 - S_p2| > 0.02 (each pass's own S).
  * KEEP if S <= 0.97 and W <= 1.02: the R term earns its cross-card-class bargain at kernel level on this card.
  * OFF otherwise: no rented reading supports the term, so it should not be on by default anywhere.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

FAMILIES = {"qwen3_moe": (2048, 768), "granitemoe": (1536, 512), "olmoe": (2048, 1024)}
PROJS = ("gate_up", "down")
RS = (1, 2, 4, 8, 16, 32, 64, 128)
SPLITK_R_FLOOR = 16
SPLITK_TARGET_BLOCKS_PER_SM = 8
SPLITK_R_TERM_MAX_SMS = 64
MAX_SMS = 64
CARD = "NVIDIA L4"
S_KEEP, W_KEEP, PASS_SPREAD = 0.97, 1.02, 0.02


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def plan_sk(N: int, K: int, R: int = 1, sm_count: int = 128) -> int:
    """int4_b32._plan's sk, restated in pure Python (checked against the installed module on the box)."""
    kb = K // 32
    ku = 4 if kb % 4 == 0 else (2 if kb % 2 == 0 else 1)
    sk = 8 if (cdiv(N, 128) * 8) >= 256 else 16
    if R >= SPLITK_R_FLOOR and sm_count <= SPLITK_R_TERM_MAX_SMS:
        programs = cdiv(N, 128) * R
        want = cdiv(SPLITK_TARGET_BLOCKS_PER_SM * sm_count, programs)
        capped, sk = sk, 1
        while sk < want and sk < capped:
            sk *= 2
    return min(sk, max(1, kb // ku))


def shape(family: str, proj: str) -> tuple[int, int]:
    hidden, inter = FAMILIES[family]
    return (2 * inter, hidden) if proj == "gate_up" else (hidden, inter)


def load(rows_dir: pathlib.Path, prefix: str) -> dict:
    """{(family, proj, R): {"ms": {sk: ms}, "gpu": .., "sms": ..}} for one pass; refuses a missing shape."""
    cells = {}
    for fam in FAMILIES:
        for proj in PROJS:
            f = rows_dir / f"{prefix}_{fam}_{proj}.json"
            if not f.is_file():
                raise SystemExit(f"VOID: {f.name} missing")
            doc = json.loads(f.read_text())
            for r in doc["rows"]:
                cells[(fam, proj, int(r["R"]))] = {"ms": {int(k): float(v) for k, v in r["ms"].items()},
                                                   "gpu": doc["gpu"], "sms": int(doc["sms"]),
                                                   "N": int(r["N"]), "K": int(r["K"])}
    return cells


def reduce(p1: dict, p2: dict) -> dict:
    """The rule over two passes. Returns the verdict and every number it read."""
    void = []
    cards = {(c["gpu"], c["sms"]) for p in (p1, p2) for c in p.values()}
    if len(cards) != 1:
        void.append(f"passes disagree on the card: {sorted(cards)}")
    gpu, sms = sorted(cards)[0]
    if gpu != CARD:
        void.append(f"card is {gpu!r}, the lane registers {CARD!r}")
    if sms > MAX_SMS:
        void.append(f"{sms} SMs is above the {MAX_SMS}-SM class the R term is gated to")
    cells, per_pass = [], {"p1": [0.0, 0.0], "p2": [0.0, 0.0]}
    for fam in FAMILIES:
        for proj in PROJS:
            N, K = shape(fam, proj)
            sk_old = plan_sk(N, K)
            for R in RS:
                key = (fam, proj, R)
                if key not in p1 or key not in p2:
                    void.append(f"cell {key} missing from a pass")
                    continue
                sk_new = plan_sk(N, K, R, sms)
                if R < SPLITK_R_FLOOR:
                    if sk_new != sk_old:
                        void.append(f"{key}: below the floor the plans differ ({sk_new} vs {sk_old})")
                    continue
                row = {"family": fam, "proj": proj, "R": R, "N": N, "K": K, "sk_new": sk_new, "sk_old": sk_old}
                for name, p in (("p1", p1), ("p2", p2)):
                    ms = p[key]["ms"]
                    if sk_new not in ms or sk_old not in ms:
                        void.append(f"{key} {name}: sk {sk_new} or {sk_old} not swept ({sorted(ms)})")
                        break
                    best = min(ms, key=ms.get)
                    row[f"new_{name}"], row[f"old_{name}"] = ms[sk_new], ms[sk_old]
                    row[f"best_{name}"], row[f"sk_best_{name}"] = ms[best], best
                    per_pass[name][0] += ms[sk_new]
                    per_pass[name][1] += ms[sk_old]
                else:
                    row["r"] = (row["new_p1"] + row["new_p2"]) / (row["old_p1"] + row["old_p2"])
                    row["self_pair_new"] = row["new_p1"] / row["new_p2"]
                    row["self_pair_old"] = row["old_p1"] / row["old_p2"]
                    cells.append(row)
    out = {"gpu": gpu, "sms": sms, "cells": cells, "void": void,
           "rule": {"S_keep": S_KEEP, "W_keep": W_KEEP, "pass_spread": PASS_SPREAD}}
    if len(cells) != 24 and not void:
        void.append(f"expected 24 cells at R >= {SPLITK_R_FLOOR}, read {len(cells)}")
    if cells:
        new = sum(c["new_p1"] + c["new_p2"] for c in cells)
        old = sum(c["old_p1"] + c["old_p2"] for c in cells)
        best = sum(c["best_p1"] + c["best_p2"] for c in cells)
        out["S"] = new / old
        out["W"] = max(c["r"] for c in cells)
        out["W_at"] = max(cells, key=lambda c: c["r"])
        out["S_p1"] = per_pass["p1"][0] / per_pass["p1"][1]
        out["S_p2"] = per_pass["p2"][0] / per_pass["p2"][1]
        out["new_vs_best"] = new / best
        out["old_vs_best"] = old / best
        out["self_pair_max"] = max(max(abs(c["self_pair_new"] - 1), abs(c["self_pair_old"] - 1)) for c in cells)
        present = sorted({c["R"] for c in cells})          # a rehearsal sweeps a subset of R; never divide by nothing
        out["by_R"] = {R: sum(c["new_p1"] + c["new_p2"] for c in cells if c["R"] == R)
                       / sum(c["old_p1"] + c["old_p2"] for c in cells if c["R"] == R) for R in present}
        if abs(out["S_p1"] - out["S_p2"]) > PASS_SPREAD:
            void.append(f"unstable instrument: S_p1 {out['S_p1']:.4f} vs S_p2 {out['S_p2']:.4f}")
    if void:
        out["verdict"] = "VOID"
    elif out["S"] <= S_KEEP and out["W"] <= W_KEEP:
        out["verdict"] = "KEEP"
    else:
        out["verdict"] = "OFF"
    return out


def check_installed_plan() -> None:
    """On the box: the installed int4_b32._plan must pick exactly the sk this reducer prices, for every cell."""
    import torch
    from int4_b32 import SPLITK_R_FLOOR as F, SPLITK_R_TERM_MAX_SMS as M, SPLITK_TARGET_BLOCKS_PER_SM as T, _plan
    assert (F, M, T) == (SPLITK_R_FLOOR, SPLITK_R_TERM_MAX_SMS, SPLITK_TARGET_BLOCKS_PER_SM), (F, M, T)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for fam in FAMILIES:
        for proj in PROJS:
            N, K = shape(fam, proj)
            assert _plan(N, K)[2] == plan_sk(N, K), (fam, proj, "N-only")
            for R in RS:
                assert _plan(N, K, R, sms)[2] == plan_sk(N, K, R, sms), (fam, proj, R, sms)
    print(f"K30 plan cross-check OK: the installed _plan matches the reducer on all 48 cells at {sms} SMs")


# ---------------------------------------------------------------------------------------------------- self-test --

def _synthetic(sms: int, gpu: str, scale_new: float, *, jitter: float = 0.0, drop: tuple | None = None,
               only_rs: tuple | None = None) -> dict:
    """A pass in load()'s shape: every swept sk costs 1.0 ms except the R-aware pick at R >= 16, which costs
    scale_new (and the N-only pick 1.0). jitter multiplies every time, for the stability arm."""
    cells = {}
    for fam in FAMILIES:
        for proj in PROJS:
            N, K = shape(fam, proj)
            for R in RS:
                if drop == (fam, proj, R) or (only_rs is not None and R not in only_rs):
                    continue
                sks = sorted({1, 2, 3, 4, 6, 8, 16, plan_sk(N, K)} & set(range(1, max(1, (K // 32) // 4) + 1))
                             | {plan_sk(N, K, R, sms), plan_sk(N, K)})
                ms = {s: 1.0 * (1 + jitter) for s in sks}
                if R >= SPLITK_R_FLOOR and plan_sk(N, K, R, sms) != plan_sk(N, K):
                    ms[plan_sk(N, K, R, sms)] = scale_new * (1 + jitter)
                cells[(fam, proj, R)] = {"ms": ms, "gpu": gpu, "sms": sms, "N": N, "K": K}
    return cells


def self_test() -> int:
    cases = [
        ("a clear win keeps the term", _synthetic(58, CARD, 0.90), _synthetic(58, CARD, 0.90), "KEEP"),
        ("a 2 % win is not enough", _synthetic(58, CARD, 0.98), _synthetic(58, CARD, 0.98), "OFF"),
        ("a loss turns it off", _synthetic(58, CARD, 1.05), _synthetic(58, CARD, 1.05), "OFF"),
        ("the wrong card is VOID", _synthetic(58, "NVIDIA L40S", 0.90), _synthetic(58, "NVIDIA L40S", 0.90), "VOID"),
        ("a missing cell is VOID", _synthetic(58, CARD, 0.90), _synthetic(58, CARD, 0.90, drop=("olmoe", "down", 64)),
         "VOID"),
        ("unstable passes are VOID", _synthetic(58, CARD, 0.90), _synthetic(58, CARD, 0.80), "VOID"),
        ("a rehearsal's two-R sweep is VOID, not a crash", _synthetic(26, "NVIDIA RTX A2000 12GB", 0.90, only_rs=(16, 128)),
         _synthetic(26, "NVIDIA RTX A2000 12GB", 0.90, only_rs=(16, 128)), "VOID"),
    ]
    # the R-aware pick differs from the N-only one somewhere at 58 SMs, or the instrument could not tell them apart
    acts = sum(plan_sk(*shape(f, p), R, 58) != plan_sk(*shape(f, p)) for f in FAMILIES for p in PROJS for R in RS)
    assert acts > 0, "at 58 SMs the R term never changes the pick: K30 would measure nothing"
    # below the floor the two plans agree on every shape at every SM count
    assert all(plan_sk(*shape(f, p), R, s) == plan_sk(*shape(f, p))
               for f in FAMILIES for p in PROJS for R in RS if R < SPLITK_R_FLOOR for s in (26, 58, 64, 128))
    bad = 0
    for name, a, b, want in cases:
        got = reduce(a, b)["verdict"]
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name}: {got} (want {want})")
    w = reduce(_synthetic(58, CARD, 0.90), _synthetic(58, CARD, 0.90))
    assert math.isclose(w["S_p1"], w["S_p2"]) and w["W"] <= 1.0 and len(w["cells"]) == 24, w.get("void")
    print(f"K30 rule self-test: {len(cases) - bad}/{len(cases)} cases; the R term changes the pick on {acts} of 48 cells at 58 SMs")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) < 3:
        print(__doc__)
        return 2
    if "--check-installed-plan" in argv:
        check_installed_plan()
    rows = pathlib.Path(argv[1])
    out = reduce(load(rows, "k30p1"), load(rows, "k30p2"))
    pathlib.Path(argv[2]).write_text(json.dumps(out, indent=1, default=str))
    if "S" in out:
        print(f"K30 S={out['S']:.4f} (p1 {out['S_p1']:.4f}, p2 {out['S_p2']:.4f}) W={out['W']:.4f} "
              f"at {out['W_at']['family']}/{out['W_at']['proj']} R={out['W_at']['R']}; "
              f"N-only {out['old_vs_best']:.4f}x and R-aware {out['new_vs_best']:.4f}x the per-cell optimum; "
              f"self-pair max {out['self_pair_max']:.4f}")
        print("K30 by R: " + ", ".join(f"R={R} {v:.4f}" for R, v in out["by_R"].items()))
    for v in out["void"]:
        print(f"K30 VOID reason: {v}")
    print(f"K30_VERDICT {out['verdict']} on {out['gpu']} ({out['sms']} SMs)")
    return 0 if out["verdict"] in ("KEEP", "OFF") else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
