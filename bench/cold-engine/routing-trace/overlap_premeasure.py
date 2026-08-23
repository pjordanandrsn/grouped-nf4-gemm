"""The routing-overlap pre-measurement. Registered in
bench/cold-engine/PREREG-overlap-premeasure.md — the bar froze before this
ran anywhere. Offline, deterministic, on the 16 committed rank traces.
"""
import itertools
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from score_policies import load                              # noqa: E402

MODELS = ("granite", "olmoe", "qwen", "gptoss")
PROMPTS = ("prose", "code", "math", "dialogue")
FRACS = (0.50, 0.66, 0.75)
TDIR = os.path.join(HERE, "..", "rank-2026-08-22")


def routed_sets(recs):
    """per step: {layer: frozenset(experts)}"""
    out = []
    for r in recs:
        out.append({int(L): frozenset(ex) for L, ex in r["routed"].items()})
    return out


def main():
    ap_out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv \
        else "/tmp/overlap.json"
    result = {"models": {}}
    gate_hits = 0
    for model in MODELS:
        traces = {}
        for p in PROMPTS:
            f = os.path.join(TDIR, f"{model}_{p}.jsonl")
            meta, recs = load(f)
            traces[p] = routed_sets(recs)
        layers = sorted(traces[PROMPTS[0]][0].keys())
        steps = min(len(t) for t in traces.values())
        # pooled mass and per-request marginals
        pooled = defaultdict(int)
        marg = {p: defaultdict(int) for p in PROMPTS}
        for p in PROMPTS:
            for step in traces[p][:steps]:
                for L, ex in step.items():
                    for e in ex:
                        pooled[(L, e)] += 1
                        marg[p][(L, e)] += 1
        for p in PROMPTS:
            for key in marg[p]:
                marg[p][key] /= steps
        ranked = sorted(pooled, key=lambda k: (-pooled[k], k))
        mm = {"layers": len(layers), "steps": steps,
              "pairs_seen": len(ranked), "fracs": {}}
        for frac in FRACS:
            vram = set(ranked[: int(frac * len(ranked))])
            dram = [k for k in ranked if k not in vram]
            dram_set = set(dram)

            def union_emp(group):
                tot = 0
                for t in range(steps):
                    for L in layers:
                        u = set()
                        for p in group:
                            u |= traces[p][t].get(L, frozenset())
                        tot += sum(1 for e in u if (L, e) in dram_set)
                return tot / steps                    # uniques per STEP

            def union_ind(group):
                s = 0.0
                for key in dram:
                    miss = 1.0
                    for p in group:
                        miss *= 1.0 - marg[p].get(key, 0.0)
                    s += 1.0 - miss
                return s

            pairs = list(itertools.combinations(PROMPTS, 2))
            pair_R = {}
            for g in pairs:
                e_, i_ = union_emp(g), union_ind(g)
                pair_R["+".join(g)] = {"emp": e_, "ind": i_,
                                       "R": e_ / i_ if i_ else 1.0}
            e4, i4 = union_emp(PROMPTS), union_ind(PROMPTS)
            R4 = e4 / i4 if i4 else 1.0
            mm["fracs"][str(frac)] = {
                "dram_pairs": len(dram),
                "m4": {"emp": e4, "ind": i4, "R": R4},
                "m2_pairs": pair_R,
                "m2_R_spread": (min(v["R"] for v in pair_R.values()),
                                max(v["R"] for v in pair_R.values())),
                "m4_delta_uniques": i4 - e4,
                "m4_value_us_per_step": (i4 - e4) * 58.0,
            }
            if abs(frac - 0.66) < 1e-9 and R4 <= 0.90:
                gate_hits += 1
            print("%-8s f=%.2f  m4 R=%.3f (emp %.1f vs ind %.1f uniq/step)"
                  "  m2 R spread %.3f-%.3f  value %.0f us/step"
                  % (model, frac, R4, e4, i4,
                     *mm["fracs"][str(frac)]["m2_R_spread"],
                     mm["fracs"][str(frac)]["m4_value_us_per_step"]))
        result["models"][model] = mm
    verdict = "GRADUATE" if gate_hits >= 2 else "CLOSED"
    result["gate_hits_at_f066_m4"] = gate_hits
    result["verdict"] = verdict
    print("models with R<=0.90 at f=0.66 m=4: %d/4  ->  %s"
          % (gate_hits, verdict))
    with open(ap_out, "w") as f:
        json.dump(result, f, indent=1)
    print("receipt ->", ap_out)


if __name__ == "__main__":
    main()
