"""Step-to-step routing overlap, raw and chance-normalized.

Exploratory support for RESULTS-third-model.md. Two tables:

  * overlap of step i with step i-lag, for lag 1..6 -- a spike at even lags
    means the generation fell into a repetition loop and the trace is
    degenerate (greedy argmax decode does this readily);
  * overlap at lag 1 divided by `k/E`, the value expected if each layer drew
    its k experts independently and uniformly from E.

The normalization is the point. Raw lag-1 overlap differs 3x across the three
models and lines up exactly with which of them win at a one-step cache, which
looks like an explanation. It is not: `k/E` alone differs 3x, and once it is
divided out Granite (wins) and Qwen (loses) sit at 2.18x and 2.01x, closer to
each other than Granite is to OLMoE. Reporting the raw number as a finding
would have been fitting a story to arithmetic.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from score_policies import load                              # noqa: E402

PROMPTS = ("prose", "code", "math", "dialogue")


def step_sets(recs):
    return [{(int(L), e) for L, ex in r["routed"].items() for e in ex}
            for r in recs]


def overlap(S, lag):
    v = [len(S[i] & S[i - lag]) / len(S[i]) for i in range(lag, len(S))]
    return sum(v) / len(v)


def token_period(recs, max_p=8, frac=0.9):
    """Smallest p where token[i] == token[i-p] for >= `frac` of steps.

    None when the trace predates token recording, or when no period fits --
    which is the normal case. A small p is a degenerate generation: the model
    is emitting a fixed cycle and every routing statistic downstream inherits
    it.
    """
    tok = [r.get("token") for r in recs]
    if any(t is None for t in tok):
        return None
    for p in range(1, max_p + 1):
        n = len(tok) - p
        if n > 0 and sum(tok[i] == tok[i - p] for i in range(p, len(tok))) \
                >= frac * n:
            return p
    return 0                      # tokens recorded, no cycle found


def n_experts(meta, recs):
    """E, from the metadata when the capture recorded it.

    Granite's router did not expose a count and meta carries null, so fall
    back to the largest id actually routed. That is a LOWER BOUND -- an expert
    no token ever selected is invisible -- so it is reported as such.
    """
    if meta.get("n_experts"):
        return int(meta["n_experts"]), True
    mx = max(e for s in recs for L, ex in s["routed"].items() for e in ex)
    return mx + 1, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe,granite,qwen")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prompts = [x for x in a.prompts.split(",") if x]

    rows = []
    print("overlap of step i with step i-lag (mean over trace)\n")
    print("%-8s %-9s %s" % ("model", "prompt",
                            " ".join("lag%-2d" % l for l in range(1, 7))))
    for m in a.models.split(","):
        for p in prompts:
            f = os.path.join(a.dir, "%s_%s.jsonl" % (m, p))
            if not os.path.exists(f):
                continue
            meta, recs = load(f)
            S = step_sets(recs)
            E, exact = n_experts(meta, recs)
            k = meta["top_k"]
            lags = [overlap(S, l) for l in range(1, 7)]
            per = token_period(recs)
            rows.append({"model": m, "prompt": p, "top_k": k, "n_experts": E,
                         "n_experts_exact": exact, "lags": lags,
                         "token_period": per,
                         "chance": k / E, "norm": lags[0] / (k / E)})
            note = ""
            if per:
                note = "  <- GENERATION LOOPS with period %d" % per
            elif per is None and max(lags[1::2]) > 2 * min(lags[0::2]):
                note = "  <- even-lag spike; likely a loop (no tokens in trace)"
            print("%-8s %-9s %s%s"
                  % (m, p, " ".join("%5.1f" % (100 * x) for x in lags), note))
        print()

    print("lag-1 overlap against chance (k/E)\n")
    print("%-8s %-9s %3s %5s %7s %8s %10s"
          % ("model", "prompt", "k", "E", "chance", "observed", "obs/chance"))
    for r in rows:
        print("%-8s %-9s %3d %5s %6.1f%% %7.1f%% %9.2fx"
              % (r["model"], r["prompt"], r["top_k"],
                 r["n_experts"] if r["n_experts_exact"]
                 else ">=%d" % r["n_experts"],
                 100 * r["chance"], 100 * r["lags"][0], r["norm"]))
    print()
    for m in a.models.split(","):
        v = [r["norm"] for r in rows if r["model"] == m]
        if v:
            print("  %-8s mean obs/chance %.2fx" % (m, sum(v) / len(v)))

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows}, f, indent=1)
        print("\nreceipt -> %s" % a.out)


if __name__ == "__main__":
    main()
