### K30 registered: does the int4-b32 split-K R term win on a <=64-SM card? (bench only)

- `kernel/PREREG-k30-splitk-r-term-l4.md`: the 48-cell `bench/int4/sk_sweep.py` (unchanged) twice on one rented NVIDIA
  L4, after the compiled int4_b32 suite, a per-sk agreement check and the rule's self-test. KEEP if the R-aware pick's
  summed time is <= 0.97 of the N-only pick's and no cell is > 1.02 worse; OFF otherwise. No band is borrowed from the
  A2000 sweep that set the term.
- `bench/int4/k30_reduce.py` (the rule, a 6-case self-test, and an on-box cross-check that the installed `_plan` picks
  what it prices) and `bench/int4/k30_sk_check.py` (every swept sk agrees with sk = 1 before any timing). The box
  runner lives in experts4bit-qlora `bench/k30/`.
