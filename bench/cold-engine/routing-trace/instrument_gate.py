"""Is a box a usable instrument, and can it resolve the effect? Two questions.

The first version of this gate asked one question -- `effect > 3 * spread` --
and refused otherwise. That conflates two opposite situations:

  * a NOISY BOX, where the spread is large and nothing can be measured on it;
  * a GOOD BOX measuring an effect that has genuinely SHRUNK.

They look identical to a ratio test and they call for opposite responses:
re-rent, versus report the result. It fired on the second case in the
demote-heap run -- per-arm IQR 0.56% / 1.44%, among the tightest this campaign
has measured, refused because the effect had collapsed from ~8% to 1.58%. The
collapse WAS the finding.

So it now asks them separately:

  UNUSABLE   per-arm IQR exceeds --max-spread. A property of the box alone,
             independent of any effect. Re-rent.
  RESOLVED   spread fine and effect clears --factor x spread. Report the point
             estimate.
  BELOW-RES  spread fine, effect at or under the resolution. NOT a refusal:
             report an UPPER BOUND. "The effect is under X%" answers a
             dissolution prediction directly, and is the honest form of a
             measurement whose point estimate is inside its own error bar.

Spread is IQR/median, not (max-min)/median: on small n the latter is decided
by one sample, which is how an earlier revision refused a box whose four other
repeats spanned 1.1%.
"""
import argparse
import json
import random
import statistics
import sys


def iqr_pct(values):
    q = statistics.quantiles(sorted(values), n=4)
    return (q[2] - q[0]) / statistics.median(values) * 100


def effect_ci(hard, soft, reps=4000, seed=11):
    """Bootstrap CI for the wall delta.

    Per-run IQR answers "could ONE run tell these apart"; it is the wrong
    question for a 7-9 repeat median, whose uncertainty shrinks with n. Using
    IQR as the error bar put a 0.22 pt residual behind a 2.96 pt bound -- a
    limit of the statistic, not of the data. Resampling the repeats gives the
    interval that belongs to the estimator actually used.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        h = statistics.median(rng.choices(hard, k=len(hard)))
        s = statistics.median(rng.choices(soft, k=len(soft)))
        out.append((s - h) / h * 100)
    out.sort()
    return out[int(0.025 * reps)], out[int(0.975 * reps)]


def judge(point, max_spread=2.0, factor=3.0):
    hard = iqr_pct(point["hard_wall_ns"])
    soft = iqr_pct(point["soft_wall_ns"])
    spread = max(hard, soft)
    effect = point["delta_wall_pct"]
    reads = point["delta_reads_pct"]
    resolution = factor * spread
    lo, hi = effect_ci(point["hard_wall_ns"], point["soft_wall_ns"])
    out = {"hard_iqr_pct": hard, "soft_iqr_pct": soft, "spread_pct": spread,
           "effect_pct": effect, "reads_pct": reads,
           "residual_pts": effect - reads, "resolution_pct": resolution,
           "effect_ci95": [lo, hi],
           "residual_ci95_pts": [lo - reads, hi - reads]}
    if spread > max_spread:
        out["verdict"] = "UNUSABLE"
        out["note"] = (f"per-arm IQR {spread:.2f}% exceeds {max_spread}%; the "
                       f"box is the problem regardless of the effect size")
    elif abs(effect) > resolution:
        out["verdict"] = "RESOLVED"
        out["note"] = (f"effect {effect:+.2f}% clears {factor}x spread "
                       f"({resolution:.2f}%)")
    else:
        out["verdict"] = "BELOW-RES"
        # the residual cannot be pinned, but it CAN be bounded
        out["residual_upper_pts"] = hi - reads
        out["note"] = (f"effect {effect:+.2f}% is inside a single run's "
                       f"{resolution:.2f}% resolution on a CLEAN box (IQR "
                       f"{spread:.2f}%). Not a refusal. Across the repeats the "
                       f"95% CI is [{lo:+.2f}, {hi:+.2f}]%, so residual is in "
                       f"[{lo - reads:+.2f}, {hi - reads:+.2f}] pts")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--max-spread", type=float, default=2.0,
                    help="per-arm IQR%% above which the BOX is unusable")
    ap.add_argument("--factor", type=float, default=3.0,
                    help="multiple of spread an effect must clear to be a "
                         "point estimate rather than a bound")
    a = ap.parse_args()
    with open(a.receipt) as f:
        d = json.load(f)
    rc = 0
    for p in d["points"]:
        v = judge(p, a.max_spread, a.factor)
        print(f"rows={p['rows']} prot={p['protected']}  {v['verdict']}")
        print(f"  IQR hard {v['hard_iqr_pct']:.2f}%  soft "
              f"{v['soft_iqr_pct']:.2f}%   effect {v['effect_pct']:+.2f}%   "
              f"residual {v['residual_pts']:+.2f} pts")
        print(f"  {v['note']}")
        if "residual_ci95_pts" in v:
            a_, b_ = v["residual_ci95_pts"]
            print(f"  residual 95% CI: [{a_:+.2f}, {b_:+.2f}] pts")
        if v["verdict"] == "UNUSABLE":
            rc = 9
    return rc


if __name__ == "__main__":
    sys.exit(main())
