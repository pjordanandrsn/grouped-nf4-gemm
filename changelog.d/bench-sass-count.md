### `bench/sass_count`: a static instructions-per-weight instrument for kernel PRs

- **What it is:** `bench/sass_count/sass_count.py` compiles a Triton kernel as a served launch would, for a named GPU
  target, with no GPU. It disassembles the result and reports the inner K loop's lane instructions per weight, by
  class. A kernel PR that changes a decode or a load path can quote one number per target.
- **The JIT's specialization is built in:** every 16-aligned tensor and every int divisible by 16 is marked
  `tt.divisibility = 16`. A count compiled without those hints is inflated, because loads scalarize and per-element
  address arithmetic enters the loop. The test fails if the hints go missing.
- **Recorded counts:** K25 at experts4bit-qlora's served plan reads 24.1 lane instructions per weight on sm_120 and
  25.1 on sm_86 (triton 3.4.0). The select-tree decode is 10.3 of the 24.1.
- **No change to any kernel.**
