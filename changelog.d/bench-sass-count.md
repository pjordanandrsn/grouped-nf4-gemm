### `bench/sass_count`: a static instructions-per-weight instrument for kernel PRs

- **What it is:** `bench/sass_count/sass_count.py` compiles a Triton kernel as a served launch would, for a named GPU
  target, with no GPU. It disassembles the result and reports the inner K loop's lane instructions per weight, by
  class. A kernel PR that changes a decode or a load path can quote one number per target.
- **The JIT's specialization is built in:** every 16-aligned tensor and every int divisible by 16 is marked
  `tt.divisibility = 16`. A count compiled without those hints is inflated, because loads scalarize and per-element
  address arithmetic enters the loop. The test fails if the hints go missing.
- **Recorded counts:** K25 at experts4bit-qlora's served plan reads 24.1 lane instructions per weight on sm_120 and
  25.1 on sm_86 (triton 3.4.0). The select-tree decode is 10.3 of the 24.1.
- **The toolchain is part of the count:** every record names its triton and bundled ptxas, and `compare` refuses a
  ratio across two toolchains. Under triton 3.8.0 (ptxas 12.9.86), K25's served loop reads 36.2 on sm_120 and 37.9
  on sm_86, about 50 % above 3.4.0, and the hints no longer lower it. CI checks the structure on the latest triton
  and pins the counts at 3.4.0, the served stack's.
- **Newer ptxas's loop branch:** a backward `BRA.U UP0, <target>`, with the predicate as an operand, now closes a
  loop. Without it, the counter found no loop on sm_120 under triton 3.8.0.
- **No change to any kernel.**
