"""Does popularity skew explain where a frequency-aware victim rule helps?

RESULTS-policy-headroom.md measured LFU closing 49% of the LRU-to-optimal gap
on OLMoE and Granite, 30% on Qwen and 2% on gpt-oss -- the one model every wall
measurement in this program was made on. It records the fact and not the cause.

The obvious cause is skew: LFU keeps rows that recur because they are POPULAR,
so if routing spreads evenly across experts every resident row has the same
expected future and frequency is noise. Qwen's math prompt is the degenerate
case already on record -- a period-2 alternation, frequency uniform, LFU 2.31x
worse.

This measures skew on the routed stream the cache actually sees, per cell, and
correlates it with the LFU/LRU ratios already measured. Registered in
PREREG-frequency-signal.md: CONFIRMED at Spearman |rho| >= 0.8, REFUTED below
0.5 or if gpt-oss is not the most uniform.

Skew comes from the routed (layer, expert) stream, not gate logits: the cache
can only act on what it observes.
"""
import argparse
import json
import math
import os
import statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = ("code", "math", "prose", "dialogue")


def load(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return rows[0]["meta"], rows[1:]


def keystream(recs):
    out = []
    for r in recs:
        for lay, experts in r["routed"].items():
            for e in experts:
                out.append((int(lay), int(e)))
    return out


def skew(keys):
    """Normalised entropy and Gini over the visit distribution.

    Normalised entropy is 1.0 for perfectly uniform routing and falls toward 0
    as mass concentrates, so it is directly "how little signal frequency
    carries". Gini is reported alongside because entropy is insensitive to the
    tail and Gini is not -- if they disagree the shape matters, not just the
    spread.
    """
    c = Counter(keys)
    n = len(c)
    total = sum(c.values())
    ps = [v / total for v in c.values()]
    h = -sum(p * math.log(p) for p in ps if p > 0)
    h_norm = h / math.log(n) if n > 1 else 0.0
    xs = sorted(c.values())
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    gini = (2 * cum) / (len(xs) * sum(xs)) - (len(xs) + 1) / len(xs)
    return h_norm, gini, n


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):           # average ties, or the coefficient is
            j = i                       # biased by however the sort broke them
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-receipt", required=True,
                    help="policy-headroom.json, for the measured LFU/LRU")
    ap.add_argument("--dirs", default=HERE,
                    help="comma-separated dirs holding <model>_<prompt>.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.policy_receipt) as f:
        pol = json.load(f)
    rows = pol["rows"] if isinstance(pol, dict) else pol

    cache = {}
    cells = []
    missing = set()
    for r in rows:
        m, p = r["model"], r["prompt"]
        if (m, p) not in cache:
            path = None
            for d in a.dirs.split(","):
                cand = os.path.join(d, "%s_%s.jsonl" % (m, p))
                if os.path.exists(cand):
                    path = cand
                    break
            if path is None:
                missing.add((m, p))
                continue
            _meta, recs = load(path)
            cache[(m, p)] = skew(keystream(recs))
        h, g, n = cache[(m, p)]
        lru, lfu = r.get("p_lru"), r.get("p_lfu")
        if not lru or not lfu:
            continue
        cells.append({"model": m, "prompt": p, "steps_held": r["steps_held"],
                      "entropy": h, "gini": g, "distinct": n,
                      "lfu_over_lru": lfu / lru})

    if not cells:
        raise SystemExit(
            "no cells matched: check --dirs and that the receipt carries "
            "p_lru/p_lfu. Refusing to report a correlation over nothing.")
    print(f"{len(cells)} cells\n")
    print(f"{'model':<9} {'entropy':>8} {'gini':>7} {'distinct':>9} "
          f"{'LFU/LRU':>8}")
    by_model = {}
    for c in cells:
        by_model.setdefault(c["model"], []).append(c)
    for m, cs in sorted(by_model.items(),
                        key=lambda kv: -statistics.fmean(
                            [c["entropy"] for c in kv[1]])):
        print(f"{m:<9} {statistics.fmean([c['entropy'] for c in cs]):>8.4f} "
              f"{statistics.fmean([c['gini'] for c in cs]):>7.4f} "
              f"{cs[0]['distinct']:>9} "
              f"{statistics.median([c['lfu_over_lru'] for c in cs]):>8.4f}")
    ent = [c["entropy"] for c in cells]
    gin = [c["gini"] for c in cells]
    rat = [c["lfu_over_lru"] for c in cells]
    r_ent, r_gin = spearman(ent, rat), spearman(gin, rat)
    print(f"\nSpearman across all {len(cells)} cells:")
    print(f"  entropy vs LFU/LRU : {r_ent:+.3f}   "
          f"(uniform routing -> LFU no better -> POSITIVE if skew is the story)")
    print(f"  gini    vs LFU/LRU : {r_gin:+.3f}   (opposite sign expected)")
    # The registered rule has TWO clauses, and an earlier version of this
    # harness implemented only the first: CONFIRMED needs |rho| >= 0.8 AND
    # gpt-oss to be the most uniform model, and gpt-oss NOT being most uniform
    # is an independent REFUTED path. A run with a strong rho and some other
    # model most uniform would have printed CONFIRMED where the prereg says
    # REFUTED (Bugbot, gnf4#189).
    ent_by_model = {}
    for c in cells:
        ent_by_model.setdefault(c["model"], []).append(c["entropy"])
    mean_ent = {m: statistics.fmean(v) for m, v in ent_by_model.items()}
    most_uniform = max(mean_ent, key=mean_ent.get) if mean_ent else None
    gptoss_most_uniform = most_uniform is not None and "gptoss" in most_uniform

    if missing:
        raise SystemExit(
            f"traces missing for {sorted(missing)}; a verdict written without "
            f"them would not be the registered test. Pass --dirs.")
    if not any("gptoss" in m for m in mean_ent):
        raise SystemExit(
            "no gpt-oss cells: the registered hypothesis is ABOUT gpt-oss, so "
            "there is no verdict to report without it.")

    best = max(abs(r_ent), abs(r_gin))
    if not gptoss_most_uniform:
        verdict = "REFUTED"
        why = (f"gpt-oss is not the most uniform model ({most_uniform} is, "
               f"entropy {mean_ent[most_uniform]:.4f} vs "
               f"{mean_ent[[m for m in mean_ent if 'gptoss' in m][0]]:.4f})"
               f" -- an independent REFUTED clause, regardless of rho")
    elif best >= 0.8:
        verdict, why = "CONFIRMED", f"|rho| {best:.3f} >= 0.8 and gpt-oss is most uniform"
    elif best >= 0.5:
        verdict, why = "PARTIAL", f"|rho| {best:.3f} in [0.5, 0.8)"
    else:
        verdict, why = "REFUTED", f"|rho| {best:.3f} < 0.5"
    print(f"\nregistered rule has TWO clauses: |rho| >= 0.8 AND gpt-oss most "
          f"uniform.\n  most uniform: {most_uniform}   -> {verdict}\n  {why}")
    if a.out:
        json.dump({"cells": cells, "spearman_entropy": r_ent,
                   "spearman_gini": r_gin, "verdict": verdict,
                   "most_uniform": most_uniform,
                   "gptoss_most_uniform": gptoss_most_uniform,
                   "verdict_reason": why},
                  open(a.out, "w"), indent=2)
        print("receipt ->", a.out)


if __name__ == "__main__":
    main()
