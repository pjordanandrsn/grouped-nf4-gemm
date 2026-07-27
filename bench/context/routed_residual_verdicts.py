"""Verdict logic for PREREG-routed-residual — stdlib only, no torch, no bnb, no GPU.

Split out of the harness deliberately. This function encodes R5, which decides
whether the expert-major coalescer gets built; a decision function that only ever
executes on a $3/hr pod is a decision function nobody has tested. Keeping it
import-light means its tests run on the laptop where it was written.
"""

import statistics

#: Arms whose predictions are registered in the stamped prereg. Everything else
#: reports under `exploratory` and may not be quoted as confirmatory.
REGISTERED = ("C", "T1")


def evaluate(records: list, ceiling_gbps: float) -> dict:
    """Reduce arm records to the prereg's verdicts. PURE — no torch, no model.

    Split out so the decision logic is unit-testable on a laptop. A verdict
    function that only ever runs on a $3/hr pod is a verdict function nobody has
    tested, and this one encodes R5, which decides whether a week of coalescer
    work happens.

    `records`: [{arm, rep, s_per_token, greedy_ids, counts, routed_gbps}]
    """
    by_arm: dict = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)

    def med(arm):
        return statistics.median(x["s_per_token"] for x in by_arm[arm])

    out: dict = {"registered": {}, "exploratory": {}, "gates_passed": None}

    missing = [a for a in REGISTERED if a not in by_arm]
    if missing:
        return {**out, "error": f"missing registered arm(s): {missing}"}

    # --- R1: bit-identity. A divergence is a STOP, not a slow result. ----------
    ids_c = by_arm["C"][0]["greedy_ids"]
    r1_ok = all(x["greedy_ids"] == ids_c for x in by_arm["C"] + by_arm["T1"])
    out["registered"]["R1_bit_identity"] = {
        "pass": r1_ok,
        "detail": "greedy ids identical across C and T1" if r1_ok
                  else "DIVERGENCE — a routed row was never staged; STOP, do not report timings",
    }

    # --- R2: engagement. Both counts, because device==0 also means "never ran". -
    def counts_for(arm):
        h = sum(x["counts"]["host"] for x in by_arm[arm])
        d = sum(x["counts"]["device"] for x in by_arm[arm])
        return h, d

    t_host, t_dev = counts_for("T1")
    c_host, c_dev = counts_for("C")
    r2_ok = t_host > 0 and t_dev == 0 and c_dev > 0 and c_host == 0
    out["registered"]["R2_engagement"] = {
        "pass": r2_ok,
        "T1": {"host": t_host, "device": t_dev},
        "C": {"host": c_host, "device": c_dev},
        "detail": "T1 host>0 & device==0; C is the mirror image, which also proves the switch flipped"
                  if r2_ok else "fast path not engaged as specified — timings are uninterpretable",
    }

    # --- R3 / R6: the pair, against the self-pair noise floor. -----------------
    c_times = sorted(x["s_per_token"] for x in by_arm["C"])
    spread = (c_times[-1] - c_times[0]) / statistics.mean(c_times) if len(c_times) > 1 else float("nan")
    ratio = med("T1") / med("C")
    out["registered"]["R3_no_regression"] = {
        "pass": ratio <= 1.0 + (spread if spread == spread else 0.0),
        "ratio": ratio, "self_pair_spread": spread,
    }
    if ratio > 1.0:
        r6 = "REGRESSION"
    elif ratio < 0.95:
        r6 = "BELOW BAND — the registered model of where the time goes is wrong; explain, do not claim"
    else:
        r6 = "IN BAND [0.95, 1.00]"
    out["registered"]["R6_t1_magnitude"] = {
        "ratio": ratio, "band": [0.95, 1.00], "verdict": r6,
        "pass": 0.95 <= ratio <= 1.00,
    }

    # --- R4 + R5: the decomposition, and the decision it forces. ---------------
    gbps = [x["routed_gbps"] for x in by_arm["C"] if x.get("routed_gbps")]
    if gbps and ceiling_gbps:
        achieved = statistics.median(gbps)
        frac = achieved / ceiling_gbps
        holds = frac <= 0.70
        out["registered"]["R4_decomposition"] = {
            "routed_gbps": achieved, "ceiling_gbps": ceiling_gbps,
            "fraction_of_ceiling": frac, "bar": 0.70, "pass": holds,
        }
        out["registered"]["R5_decision"] = {
            "build_expert_major_coalescer": holds,
            "detail": (
                "R4 holds: residual is transfer inefficiency. BUILD the coalescer; its ceiling "
                f"is the {(1 - frac) * 100:.1f}% of link left on the floor — claims beyond that "
                "gap are unsupported."
                if holds else
                "R4 FALSIFIED: transfers are already near-efficient, so the copy count is not "
                "costing what #22's phrasing implied. DO NOT build the coalescer. Record the "
                "negative and move the lane to host-side stall."
            ),
        }
    else:
        out["registered"]["R4_decomposition"] = {"pass": None, "detail": "no routed stats — was E4B_OFFLOAD_STATS=1 set?"}
        out["registered"]["R5_decision"] = {"build_expert_major_coalescer": None}

    for arm in ("T1s", "T1c"):
        if arm in by_arm:
            out["exploratory"][arm] = {
                "ratio_vs_C": med(arm) / med("C"),
                "note": "NOT REGISTERED — needs its own prereg before it can be quoted",
            }

    # Harness-level fidelity gate, NOT a registered prediction. The copy-plan arms
    # write identical bytes, so a control that quietly ran the treatment's copy loop
    # is invisible to every other check here -- R6 would measure half of T1 and
    # report it as T1. Only meaningful when the records carry `row_plan`.
    rp = [x for x in records if x.get("row_plan")]
    if rp:
        c_rp = [x["row_plan"] for x in by_arm["C"] if x.get("row_plan")]
        t_rp = [x["row_plan"] for x in by_arm["T1"] if x.get("row_plan")]
        fid = (all(p["dict"] > 0 and p["flat"] == 0 for p in c_rp)
               and all(p["flat"] > 0 and p["dict"] == 0 for p in t_rp))
        out["arm_fidelity"] = {
            "pass": fid, "C": c_rp, "T1": t_rp,
            "detail": "C on the dict plan, T1 on the flat plan" if fid
                      else "ARM LEAK — an arm ran the other arm's copy loop; the pair is void",
        }
    else:
        out["arm_fidelity"] = {"pass": None, "detail": "no row_plan counts in records"}

    gates = [out["registered"]["R1_bit_identity"]["pass"], out["registered"]["R2_engagement"]["pass"]]
    if out["arm_fidelity"]["pass"] is False:
        gates.append(False)
    out["gates_passed"] = all(gates)
    return out


