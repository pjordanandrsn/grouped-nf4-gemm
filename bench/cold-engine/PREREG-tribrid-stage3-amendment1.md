# PREREG amendment 1 — Stage 3, gate 2

Amends [`PREREG-tribrid-stage3.md`](PREREG-tribrid-stage3.md) (stamped
`7bf5b2be87aef56dc514f67cf90ea219ba1289004aaf0901e5c3230503a52ef5`). Filed
**before** gate 2 is run, and before the deadline estimator it requires
exists, so nothing here is chosen after seeing data.

## What is NOT amended, and a correction

Gate 2's core receipt stands **exactly as registered**:

    T_isolated_GPU(E) < T_isolated_CPU(E)   and   T_join_CPU(E) < T_join_GPU(E)

An earlier reading of the gate-1 cost attribution
([`RESULTS-tribrid-gate1.md`](RESULTS-tribrid-gate1.md), §addendum) treated
this gate as invalidated on the grounds that "storage is not the term worth
scheduling." **That was a misreading and is withdrawn.** The cost
attribution bears on gate 1's *hide-ratio* clause, which is about hiding
**storage** latency. Gate 2 is about which **compute engine** delivers a
contribution first under load. Those are different questions and the second
survives the first's result untouched.

## Amendment A1 — the dynamic arm needs a deadline estimator to exist first

Gate 2 compares destination *policies*. The arm gate 1 shipped as "dynamic"
is a **rows-per-unique-expert threshold** (`cold_dest=<rows>`), the DRAM
tier's `offload_rows` statistic read the other way round. It is a static
rule evaluated per step, not an estimate of time-to-contribution.

Running gate 2 against it would compare two static rules and answer a
question the gate does not ask. **Gate 2 is therefore blocked on
workstream 4** (the deadline-aware scheduler), and its dynamic arm is
redefined as:

> the destination chosen by a predicted time-to-contribution that
> incorporates current CPU and GPU backlog, not by a fixed threshold on
> routing shape.

The threshold becomes gate 2's **baseline arm**, which is the honest
comparator: a deadline model has to beat the cheap rule, not merely beat
fixed-GPU.

## Amendment A2 — the pressure axes are re-weighted, not dropped

As registered, gate 2 varies "disk and PCIe pressure ... independently."
Gate 1 measured storage at **5–11%** of cold-path cost at 1–10% cold mass,
so disk pressure is a weak lever there and a strong one only past the knee.
The axes are re-ordered by what the machine actually responds to:

1. **compute-side load asymmetry** (GPU-loaded, CPU-loaded, both) —
   primary, and the axis the core receipt turns on;
2. **cold mass**, since the destination question changes character across
   the knee — at 20% the tier is under real pressure and fills dominate;
3. **disk and PCIe pressure** — retained, but as a secondary sweep rather
   than a co-equal one.

No threshold or verdict changes. This is a statement about where to spend
runs.

## Amendment A3 — one receipt gate 1 already produced, carried forward

Destinations **do** flip under the threshold arm: 225/1874 cold rows
CPU/GPU at 1% cold mass, 4005/5061 at 5%, 17973/9844 at 20%
(`gate1_v2.json`). That answers "does the choice ever change" and it is
**not** re-litigated by gate 2. What gate 2 must answer is narrower and
harder: whether choosing by predicted deadline **beats** choosing by
threshold on exposed wall.

## Unchanged

Every clause, threshold, arm definition and stop condition in the original
prereg not named above. The equivalence requirement in particular is
unchanged and non-tradeable: a destination policy that is faster and
numerically wrong fails.
