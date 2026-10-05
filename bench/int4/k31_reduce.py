"""K31's reducer (kernel/PREREG-k31-splitk-r-term-a4000.md): K30's rule, unchanged, on one rented RTX A4000.

    python k31_reduce.py --self-test
    python k31_reduce.py <rows-dir> <out.json> [--check-installed-plan]

The rule is `k30_reduce.py`'s, imported from the same commit and called as it stands: `reduce`, `load`, `plan_sk`,
`check_installed_plan` and K30's seven self-test cases. Two things are substituted, and only these:
  * the registered card, `NVIDIA RTX A4000` (48 SMs, sm_86), in place of K30's `NVIDIA L4`;
  * the pass prefixes, `k31p1` / `k31p2`.

The thresholds (S <= 0.97 and W <= 1.02 for KEEP), the 24 cells at R >= 16, the pass-spread VOID (0.02) and every
VOID condition are K30's. How a K31 verdict combines with K30's is the registration's, not this file's.
"""
from __future__ import annotations

import json
import pathlib
import sys

import k30_reduce as k30

CARD = "NVIDIA RTX A4000"
SMS = 48
k30.CARD = CARD          # the one rule-level substitution: k30.reduce() reads its module global at call time


def self_test() -> int:
    """K30's seven cases against this card, plus K31's own checks."""
    bad = k30.self_test()
    acts = sum(k30.plan_sk(*k30.shape(f, p), R, SMS) != k30.plan_sk(*k30.shape(f, p))
               for f in k30.FAMILIES for p in k30.PROJS for R in k30.RS)
    acted_above_floor = sum(k30.plan_sk(*k30.shape(f, p), R, SMS) != k30.plan_sk(*k30.shape(f, p))
                            for f in k30.FAMILIES for p in k30.PROJS for R in k30.RS if R >= k30.SPLITK_R_FLOOR)
    assert acts == acted_above_floor == 24, (acts, acted_above_floor)
    extra = [
        ("K30's card is not this lane's card", k30._synthetic(SMS, "NVIDIA L4", 0.90),
         k30._synthetic(SMS, "NVIDIA L4", 0.90), "VOID"),
        ("a clear win on this card keeps the term", k30._synthetic(SMS, CARD, 0.90),
         k30._synthetic(SMS, CARD, 0.90), "KEEP"),
        ("an RTX A4000 Ada is another card", k30._synthetic(SMS, "NVIDIA RTX 4000 Ada Generation", 0.90),
         k30._synthetic(SMS, "NVIDIA RTX 4000 Ada Generation", 0.90), "VOID"),
    ]
    for name, a, b, want in extra:
        got = k30.reduce(a, b)["verdict"]
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name}: {got} (want {want})")
    print(f"K31 rule self-test: K30's cases plus {len(extra)}; the R term changes the pick on {acts} of 48 cells at "
          f"{SMS} SMs, all of them at R >= {k30.SPLITK_R_FLOOR}")
    return 1 if bad else 0


def check_installed_plan() -> None:
    k30.check_installed_plan()


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) < 3:
        print(__doc__)
        return 2
    if "--check-installed-plan" in argv:
        check_installed_plan()
    rows = pathlib.Path(argv[1])
    out = k30.reduce(k30.load(rows, "k31p1"), k30.load(rows, "k31p2"))
    out["lane"] = "K31"
    pathlib.Path(argv[2]).write_text(json.dumps(out, indent=1, default=str))
    if "S" in out:
        print(f"K31 S={out['S']:.4f} (p1 {out['S_p1']:.4f}, p2 {out['S_p2']:.4f}) W={out['W']:.4f} "
              f"at {out['W_at']['family']}/{out['W_at']['proj']} R={out['W_at']['R']}; "
              f"N-only {out['old_vs_best']:.4f}x and R-aware {out['new_vs_best']:.4f}x the per-cell optimum; "
              f"self-pair max {out['self_pair_max']:.4f}")
        print("K31 by R: " + ", ".join(f"R={R} {v:.4f}" for R, v in out["by_R"].items()))
    for v in out["void"]:
        print(f"K31 VOID reason: {v}")
    print(f"K31_VERDICT {out['verdict']} on {out['gpu']} ({out['sms']} SMs)")
    return 0 if out["verdict"] in ("KEEP", "OFF") else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
