### `bench/sass_count`: a load-keyed loop finder, for kernel variants with no MMA

- **The key:** `inner_loop(rows, key="load")` (CLI `--loop-key load`) finds the weight stream in a variant with no MMA,
  such as an ablation that removes it. It takes the loop whose own load bytes per iteration (less those of the loops
  nested inside it) are the most, not the smallest: a small loop that loads indices also holds a load. The pick lists
  the loops it passed over, the runner-up with its loads, and how many loops sit inside it.
- **The unroll guard for a loop with no MMA:** ptxas unrolls such loops when the trip count is a compile-time constant.
  `bytes_per_iteration` (CLI `--bytes-per-iteration`) gives k = load bytes x lanes / the stated stream, refused unless
  whole; `stream_bytes(rec)` takes that stream from a checked control. Every record names its `unroll_guard` (`"mma"`,
  `"bytes"`, or None for unchecked).
- **The loads:** every record carries `loads_per_weight` (each global-load opcode, so its width too), and
  `same_loads(a, b)` says whether a variant keeps every load of its control. It refuses across toolchains and targets,
  and where an unchecked side reads a whole k times the other, it names k instead of answering False.
  `loads_in_function` and `loads_outside_loop` give raw totals, for a loop unrolled away entirely. `shared_per_weight`
  and `same_shared(a, b)` do the same for the shared-memory traffic (LDSM / LDS / STS), the operand staging.
  `same_signature(a, b, classes=..., prefixes=...)` compares a named opcode set per weight (a math-parity check).
- **The control:** on served K25 both keys take the same loop, at triton 3.4.0 and 3.8.0, on sm_86 and sm_120. The
  fixtures (an index loop beside the weight loop, and inside it) and two compiled no-MMA kernels, one that ptxas
  unrolls 4-fold, pin the key and the guard.
- **The CLI:** `--fn` names the jit kernel in `--module`.
- **No change to any kernel.**
