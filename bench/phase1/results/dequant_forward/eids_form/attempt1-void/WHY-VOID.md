# Attempt 1 at E1 — VOID, kept because the diagnosis is the evidence

Three separate invocations on one 4090: list, tensor, list. Self-pairs mostly
held (0.96–1.05) but **drift did not** — `gb_drift` ranged 0.83–1.42 — and
every cell voided under the registered Q1 rule. E1 was not adjudicable.

The cause is structural and mine: this pod ran eight ~1.4 ms cells per pass
with a CPU-bound `QuantStack` build between each and no large cell to keep the
card hot. Worse, it compared two forms **across invocations**, when adjacent
pairing is this program's entire discipline for exactly that reason.

Attempt 2 (`bench/phase1/probe_eids_form.py`) times both forms **adjacently
inside one cell** on the same fixtures — list, list, tensor, tensor, list — so
the opening pair is the self-pair, the tensor timing is taken against the list
timing immediately before it, and drift spans the comparison rather than a
process boundary.

Nothing here is a measurement. Kept only so the reason attempt 2 is shaped the
way it is can be checked rather than taken on trust.
