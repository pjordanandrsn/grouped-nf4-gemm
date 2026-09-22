#!/usr/bin/env python3
"""K18 read: kernel/receipts-k18/5090/k18_rows.json -> the RESULTS tables + the pre-registered verdicts.

Applies PREREG-k18-grouped-expert-gemv.md literally:
  P1  grouped torch.equal served on every replayed step (both projections) and every small-R row; a single
      mismatch refuses the kernel (the box's interpreter + compiled contracts run before the bench and stop
      the lane on failure, so a rows file exists only if they passed);
  P2  served - grouped >= 0.5 ms/step on P60's recorded routing (band 0.5-0.92, the ceiling being P60's dedup
      arm); REFUTED if the saving is under 0.2 ms or grouped is slower;
  P3  at R = 8 and 16 on the six K17 shapes grouped is within +3 % of served (or faster); REFUTED if > +5 % on
      any shape.
Prints markdown only; quotes nothing the rows do not carry.
"""
import json
import sys

d = json.load(open(sys.argv[1]))
rv = d["replay"]
sv, gp, dd = (rv[k]["step_ms_median"] for k in ("served", "grouped", "dedup"))
saving = sv - gp
print(f"device {d['device']} ({d['sm_count']} SMs), torch {d['torch']}, {d['steps']} recorded steps x {d['layers']} layers, "
      f"eids sha256 {d['eids_sha256'][:12]}..., split-K plan {d['plans']}\n")
print("| arm (one CUDA graph per step, 20 replays, median) | step ms median | mean | min | max |")
print("|---|---|---|---|---|")
for k, label in (("served", "served `gemv_int4_b32`"), ("grouped", "grouped `gemv_int4_b32_grouped` (mt=4)"),
                 ("dedup", "dedup: one row per distinct expert (P60's ceiling)")):
    r = rv[k]
    print(f"| {label} | {r['step_ms_median']:.3f} | {r['step_ms_mean']:.3f} | {r['min']:.3f} | {r['max']:.3f} |")
if saving > 0 and sv > dd:
    tail = f"grouped recovers {saving / (sv - dd) * 100:.0f} % of the dedup gap"
else:
    tail = f"grouped is SLOWER than served, {gp / sv:.3f}x"
print(f"\nserved - grouped = **{saving:+.3f} ms/step**; served - dedup = {sv - dd:+.3f} ms/step ({tail})")

small = d["small_r"]
print("\n| shape | N | K | R | P1 equal | served us | grouped us | grouped/served |")
print("|---|---|---|---|---|---|---|---|")
for r in small:
    print(f"| {r['shape']} | {r['N']} | {r['K']} | {r['R']} | {r['p1_equal']} | {r['served_us']:.2f} | {r['grouped_us']:.2f} | {r['ratio']:.3f} |")

mism = d["p1_replay_mismatches"]
small_bad = [r for r in small if not r["p1_equal"]]
p1 = mism == 0 and not small_bad
p2_refuted = saving < 0.2
p2 = saving >= 0.5
over3 = [r for r in small if r["ratio"] > 1.03]
over5 = [r for r in small if r["ratio"] > 1.05]
p3 = not over3
p3_refuted = bool(over5)

print()
print(f"- **P1 (bitwise identity):** {'HELD' if p1 else 'REFUTED'} — replay mismatches {mism} of {2 * d['steps']} "
      f"(step x projection) checks; small-R rows equal {len(small) - len(small_bad)}/{len(small)}")
band = "" if not p2 else (" (inside the registered 0.5–0.92 band)" if saving <= 0.92 else
                          " (ABOVE the 0.92 ceiling P60's dedup arm set — read the dedup arm before believing it)")
print(f"- **P2 (served − grouped ≥ 0.5 ms/step):** {'HELD' if p2 else ('REFUTED' if p2_refuted else 'NOT HELD (0.2–0.5: under the bar, not refuted)')} "
      f"— {saving:+.3f} ms/step{band}")
print(f"- **P3 (grouped ≤ +3 % of served at R = 8/16):** {'HELD' if p3 else ('REFUTED' if p3_refuted else 'NOT HELD (+3–5 %: not refuted)')} "
      f"— worst {max(small, key=lambda r: r['ratio'])['shape']} R={max(small, key=lambda r: r['ratio'])['R']} "
      f"{max(r['ratio'] for r in small):.3f}"
      + ("; over +3 %: " + ", ".join(f"{r['shape']} R={r['R']} {r['ratio']:.3f}" for r in over3) if over3 else ""))
if not p1:
    verdict = "¬P1 → the kernel is REFUSED; record the differing step and stop."
elif p2:
    verdict = ("P1 ∧ P2 → the grouped GEMV ships opt-in in the next gnf4 release, and a consumer lane (experts4bit-qlora) "
               "wires it into the B=16 int4 decode branch and reads P4 before any default moves.")
else:
    verdict = ("P1 ∧ ¬P2 → recorded as exact-and-not-faster, not shipped as a lever; the headroom P60 found is not "
               "reachable by load sharing at MT = 4 (P60's P2, dedup 1.21× above the floor, is the next question).")
if p1 and not p3:
    verdict += (" ¬P3 → the consumer routes grouped by R, never a blanket switch." if p2 else
                " ¬P3 as well — moot: routing by R applies only to a kernel that is a lever, and this one is not.")
print(f"\n**Decision rule:** {verdict}")
