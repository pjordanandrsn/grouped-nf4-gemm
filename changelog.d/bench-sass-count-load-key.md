### `bench/sass_count`: a load-keyed loop finder, for kernel variants with no MMA

- **The key:** `inner_loop(rows, key="load")` (CLI `--loop-key load`) finds the weight stream in a variant with no MMA,
  such as an ablation that removes it. Of the innermost loops whose body holds a global load, it takes the one moving
  the most load bytes per iteration, not the smallest: a small loop that loads indices also holds a load. The pick
  lists the loops it passed over and the runner-up with its loads.
- **The loads:** every record carries `loads_per_weight` (each global-load opcode, so its width too), and
  `same_loads(a, b)` says whether a variant keeps every load of its control. It refuses across toolchains and targets.
  A loop with no MMA records `unroll` None; its loads are compared with the control's instead. `shared_per_weight`
  and `same_shared(a, b)` do the same for the shared-memory traffic (LDSM / LDS / STS), the operand staging.
- **The control:** on served K25 both keys take the same loop, at triton 3.4.0 and 3.8.0, on sm_86 and sm_120. A
  fixture with an index loop beside the weight loop, and a compiled no-MMA kernel, pin the key itself.
- **The CLI:** `--fn` names the jit kernel in `--module`.
- **No change to any kernel.**
