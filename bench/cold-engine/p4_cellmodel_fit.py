"""The partial-cell offline test (RESULTS-p4-rowblock's registered next
hypothesis): fit the cell model against the null on the committed
single-expert rows curve, and place the serving-call marginal beside them.

Data: e4b bench/hybrid-g9/b16close/rows_curve.json (Zen 4 TR, 8 experts,
T = rows per expert; T = 1 uses the AVX-512 single-row fast path and is
excluded — a different kernel), and this repo's p4-receipts (9V74 A/B/A).

  H_cell: med = c0 + c1 * ceil(T/8) + c2 * rows      (cells decode; rows
                                                      within a cell share it)
  H_lin:  med = a0 + a1 * rows                       (the b16close fit's form)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# committed verbatim from e4b b16close/rows_curve.json (cell-path points)
ROWS_CURVE = [(2, 1756.255), (4, 1750.476), (8, 1781.002),
              (16, 2180.056), (32, 3157.802)]
EXPERTS = 8


def fit(points):
    X_cell = np.array([[1, math.ceil(t / 8), EXPERTS * t]
                       for t, _ in points], float)
    X_lin = np.array([[1, EXPERTS * t] for t, _ in points], float)
    y = np.array([m for _, m in points])
    out = {}
    for name, X in (("H_cell", X_cell), ("H_lin", X_lin)):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        out[name] = {
            "coef": [float(b) for b in beta],
            "rms_us": float(np.sqrt(np.mean((pred - y) ** 2))),
            "max_resid_us": float(np.max(np.abs(pred - y))),
            "resid_us": [float(p - m) for (_, m), p in zip(points, pred)],
        }
    return out


def main():
    r = fit(ROWS_CURVE)
    for name, d in r.items():
        print("%s  coef %s  rms %.1f us  max|resid| %.1f us"
              % (name, [round(c, 3) for c in d["coef"]],
                 d["rms_us"], d["max_resid_us"]))
    cell = r["H_cell"]["coef"]
    print("per-expert cell decode: %.1f us/cell   within-cell row cost: "
          "%.3f us/row" % (cell[1] / EXPERTS, cell[2]))
    rec = json.load(open(os.path.join(HERE, "p4-receipts", "p4-oldA.json")))
    m64 = rec["arms"]["64"]["med_us"]
    m128 = rec["arms"]["128"]["med_us"]
    print("serving-call marginal (9V74, 29 uniques): %.2f us/row"
          % ((m128 - m64) / 64))
    with open(os.path.join(HERE, "p4-receipts", "cellmodel_fit.json"),
              "w") as f:
        json.dump(r, f, indent=1)
    print("receipt -> p4-receipts/cellmodel_fit.json")


if __name__ == "__main__":
    main()
