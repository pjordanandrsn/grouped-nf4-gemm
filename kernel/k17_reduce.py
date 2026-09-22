#!/usr/bin/env python3
"""K17 read: kernel/receipts-k17/5090/k17_rows.json -> the RESULTS table + the pre-registered verdicts.

Applies PREREG-k17-fused-splitk-gemv.md literally:
  P1  every row torch.equal (a single unequal row refuses the kernel);
  P2  R=1: saving 3-6 us on each of the six shapes; REFUTED if saving < 2 us on a majority of shapes
      or the fused path is slower on any shape;
  P3  R=128 on expert_gate_up / expert_down: fused/two_launch in [0.90, 1.15]; REFUTED if > 1.15 on either.
Prints markdown only; quotes nothing the rows do not carry.
"""
import json, sys

d = json.load(open(sys.argv[1]))
rows = d["rows"]; floor = d["launch_floor_us"]
print(f"device {d['device']} ({d['sm_count']} SMs), torch {d['torch']}, launch floor {floor:.2f} us, {len(rows)} rows\n")
print("| shape | N | K | R | SK | P1 equal | two-launch us | fused us | saving us | saving/floor | fused/two | A/A spread us (two/fused) | cnt re-armed |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    u = r.get("us")
    if u:
        sp = u["aa_spread"]
        print(f"| {r['shape']} | {r['N']} | {r['K']} | {r['R']} | {r['sk']} | {r['p1_equal']} | {u['two_launch']:.2f} | {u['fused']:.2f} | "
              f"{u['saving']:+.2f} | {u['saving_over_floor']:+.2f} | {u['fused_over_two_launch']:.3f} | {sp['two_launch']:.2f} / {sp['fused']:.2f} | {r.get('cnt_zero_after_replays')} |")
    else:
        print(f"| {r['shape']} | {r['N']} | {r['K']} | {r['R']} | {r['sk']} | {r['p1_equal']} | {r['ms']['two_launch']} | {r['ms']['fused']} | — | — | — | — | — |")

p1 = all(r["p1_equal"] for r in rows)
timed = [r for r in rows if r.get("us")]
untimed = [r for r in rows if not r.get("us")]
r1 = [r for r in timed if r["R"] == 1]
in_band = [r for r in r1 if 3.0 <= r["us"]["saving"] <= 6.0]
under2 = [r for r in r1 if r["us"]["saving"] < 2.0]
slower = [r for r in r1 if r["us"]["saving"] < 0.0]
p2_refuted = len(under2) > len(r1) / 2 or bool(slower)
p2 = len(in_band) == len(r1) == 6 and not p2_refuted
r128 = [r for r in timed if r["R"] == 128 and r["shape"] in ("expert_gate_up", "expert_down")]
p3_bad = [r for r in r128 if r["us"]["fused_over_two_launch"] > 1.15]
p3 = len(r128) == 2 and not p3_bad and all(0.90 <= r["us"]["fused_over_two_launch"] <= 1.15 for r in r128)
rearm_bad = [r for r in timed if not r.get("cnt_zero_after_replays")]
print()
print(f"- **P1 (bitwise identity):** {'HELD' if p1 else 'REFUTED'} — {sum(r['p1_equal'] for r in rows)}/{len(rows)} rows torch.equal"
      + (f"; untimed rows: {len(untimed)}" if untimed else ""))
print(f"- **P2 (R=1 saving 3–6 µs on each of six shapes):** {'HELD' if p2 else ('REFUTED' if p2_refuted else 'NOT HELD (under band, not refuted)')} — "
      f"in band {len(in_band)}/{len(r1)}; under 2 µs {len(under2)}/{len(r1)}; slower {len(slower)}/{len(r1)}; "
      f"savings " + ", ".join(f"{r['shape']} {r['us']['saving']:+.2f}" for r in r1))
print(f"- **P3 (R=128 expert shapes, fused/two in [0.90, 1.15]):** {'HELD' if p3 else 'REFUTED'} — "
      + ", ".join(f"{r['shape']} {r['us']['fused_over_two_launch']:.3f}" for r in r128))
print(f"- **counter re-arm after 800+ replays:** {'all zero' if not rearm_bad else 'NONZERO on ' + ', '.join(f'{r[chr(115)+chr(104)+chr(97)+chr(112)+chr(101)]} R={r[chr(82)]}' for r in rearm_bad)}")
if not p1: verdict = "¬P1 → the kernel is REFUSED; record the differing element and stop."
elif p2 and p3: verdict = "P1 ∧ P2 ∧ P3 → default ON in the next gnf4 release; consumer preallocates cnt/out; P4 read in the consumer (P57) before any position moves."
elif p2 and not p3: verdict = "P1 ∧ P2 ∧ ¬P3 → ON for R ≤ 16, OFF above, by the planner; P4 read at B=1 only."
else: verdict = "¬P2 (with P1) → ship as OPT-IN (exact, costs nothing); record that the reduce launch was not the cost."
print(f"\n**Decision rule:** {verdict}")
