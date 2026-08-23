"""Verdict calculator for PREREG-lowg-split2.md. Pure arithmetic over the
four receipts lowg_cert.py emits -- committed so the gate, the
noise-derived bars, and the verdict are frozen before any box runs.

Usage:
  python lowg_cert_verdict.py --gate A1.json A2.json
      Prints per-cell noise and ACCEPT/REJECT for the A/A gate. Run
      BEFORE arm B; a REJECT means destroy the box and re-hunt.
  python lowg_cert_verdict.py --score A1.json A2.json B.json A3.json
      Prints B1/B2 verdicts against the noise-derived bars, the A3
      drift check, and the overall verdict.
"""
import json
import sys

B1 = ("1,128,256", "1,128,768")
B2 = ("16,128,256", "32,64,256", "8,128,768", "16,128,768",
      "32,128,768", "64,128,768", "29,64,768", "29,128,768")


def load(p):
    with open(p) as f:
        return json.load(f)


def noise_of(a1, a2, cell):
    m = (a1[cell] + a2[cell]) / 2.0
    return abs(a1[cell] - a2[cell]) / m, m


def gate(a1, a2):
    ok = True
    b2n = []
    for c in B1:
        n, m = noise_of(a1, a2, c)
        flag = "ok" if n <= 0.10 else "FAIL(>10%)"
        if n > 0.10:
            ok = False
        print("gate B1 %-11s mean %8.1f us  noise %5.1f%%  %s"
              % (c, m, n * 100, flag))
    for c in B2:
        n, m = noise_of(a1, a2, c)
        b2n.append(n)
        print("gate B2 %-11s mean %8.1f us  noise %5.1f%%" % (c, m, n * 100))
    med = sorted(b2n)[len(b2n) // 2]
    if med > 0.05:
        ok = False
    print("gate B2 median noise %.1f%% (bar <= 5%%)" % (med * 100))
    print("A/A GATE:", "ACCEPT" if ok else "REJECT -- destroy and re-hunt")
    return ok


def score(a1, a2, b, a3):
    if not gate(a1, a2):
        print("VERDICT: VOID (gate failed; B should never have run)")
        return
    ok = True
    for c in B1:
        n, m = noise_of(a1, a2, c)
        imp = (m - b[c]) / m
        bar = max(0.30, 3 * n)
        good = imp >= bar
        ok &= good
        print("B1 %-11s A %8.1f  B %8.1f  improvement %5.1f%%  "
              "bar >= %4.1f%%  %s" % (c, m, b[c], imp * 100, bar * 100,
                                      "PASS" if good else "FAIL"))
    for c in B2:
        n, m = noise_of(a1, a2, c)
        allow = max(0.10 * m, 3 * abs(a1[c] - a2[c]))
        dev = abs(b[c] - m)
        good = dev <= allow
        ok &= good
        print("B2 %-11s A %8.1f  B %8.1f  |dev| %7.1f  allow %7.1f  %s"
              % (c, m, b[c], dev, allow, "PASS" if good else "FAIL"))
    drift = False
    for c in B1 + B2:
        n, m = noise_of(a1, a2, c)
        bound = max(0.10 * m, 3 * abs(a1[c] - a2[c]))
        if abs(a3[c] - m) > bound:
            drift = True
            print("A3 DRIFT %-11s A3 %8.1f vs mean(A) %8.1f (bound %7.1f)"
                  % (c, a3[c], m, bound))
    if drift:
        print("VERDICT: VOID (box drifted mid-experiment)")
    else:
        print("A3 confirm: no drift at any scored cell")
        print("VERDICT:", "CERTIFIED" if ok else "REFUTED")


if __name__ == "__main__":
    mode = sys.argv[1]
    fs = [load(p) for p in sys.argv[2:]]
    if mode == "--gate":
        sys.exit(0 if gate(*fs) else 1)
    score(*fs)
