### K30's OFF, the last three places: the int4 solution page, the row-invariance test's docstring, an erratum on the A2000 sweep (docs and comments only)

- `docs/solutions/int4-decode-gemv.md` still said the split-K row term's gain was unverified and a rented read was open.
  It now says the term is off on every part (`SPLITK_R_TERM_MAX_SMS = 0`) since K30 read it OFF on a rented NVIDIA L4,
  and cites `gnf4.kernel.k30-splitk-r-term.l4.2026-10-05`.
- `kernel/test_row_invariance_gpu.py`'s docstring said a <= 64-SM part's own plan could change the split count with the
  rows. With the gate at 0 no part plans from its SM count. Comment only; the test is unchanged.
- `bench/int4/RESULTS-sk-r-sweep.md`, the A2000 sweep the retired row cites, gains a dated erratum (the
  experts4bit-qlora#1128 pattern) and its record stands. The erratum makes three points:
  - its timings are not speed evidence;
  - its sk = 1 cells skipped the `reduce_partials` launch the served path pays (K30 Amendment 1);
  - K30's rented read turned the term off.
