# sass_count: static issued instructions per weight

`sass_count.py` compiles a Triton kernel as a served launch would, disassembles it and counts the instructions in its
inner K loop, per weight and by class. It needs no GPU and gives no timing: one number per target for a kernel PR to
quote.

```
python sass_count.py --kernel-dir ../../kernel --out out/           # K25 at e4b's served plan, sm_120 and sm_86
python sass_count.py --kernel-dir ../../kernel --no-specialize      # the same compile without the JIT's hints
python sass_count.py --kernel-dir DIR --module MOD --fn ARM --loop-key load   # a K25-signature variant with no MMA
```

- **What it compiles:** `triton.compile` for `GPUTarget("cuda", <arch>, 32)` with the served plan's warps and stages,
  then `cuobjdump -sass` from the triton wheel's own bin directory. The PTX, SASS, cubin and a JSON record go to
  `--out`.
- **The JIT's specialization is built in:** a served launch marks every 16-byte-aligned tensor and every int
  divisible by 16 with `tt.divisibility = 16`, and those hints are what vectorize the loads. `jit_attrs` builds them
  from the served call's own values through the backend's `parse_attr("D")`.
  - Without them the count is inflated: K25 scalarizes its weight-byte loads and adds per-element 64-bit address
    arithmetic.
  - The test fails if the hints go missing. It also checks that `--no-specialize` does show the scalar loads.
  - The rule assumes every argument is a fresh allocation, as K25's served arguments are. A tensor VIEW at an offset
    that is not a multiple of 16 bytes would get no hint from the real JIT, so a config with a sliced argument must
    drop that argument's hint, or its count is over-specialized.
- **The inner loop:** found by one of two keys (`--loop-key`). Lane instructions per weight = body x lanes /
  weights per iteration (BLOCK_N x KC), a static floor under one issue per scheduler per cycle.
  - `mma`, the default: the smallest backward branch whose body holds an MMA, so an outer persistent loop is never
    taken for it.
  - `load`: the weight stream, for a variant with no MMA (an ablation that removes it). Of the innermost loops whose
    body holds a global load, the one moving the most load bytes per iteration. It is not the smallest, because a
    small loop that loads indices (a gather, a tile-table walk) also holds a load. The record lists the loops passed
    over (`rejected`) and the next candidate with its loads (`runner_up`).
  - On served K25 both keys take the same loop (the compiled test, at both toolchains). That is necessary, not
    sufficient: the test's fixture with an index loop beside the weight loop is the case that bites.
- **The loads, per weight:** every record carries `loads_per_weight`, the loop's global-load opcodes (so each count
  carries its width: `LDG.E.128`, `LDGSTS.E.BYPASS.128`, `LDG.E.U8`). `same_loads(a, b)` says whether a variant keeps
  every load of its control, and refuses across toolchains and targets. A loop with no MMA has no unroll guard
  (`unroll` None); its loads are compared with the control's instead, and an unrolled body would show more of them.
- **The unroll guard:** one K iteration of a BLOCK_M x BLOCK_N x KC tile on W warps issues
  (BLOCK_M/m)(BLOCK_N/n)(KC/k)/W MMAs per warp, for the MMA's shape (HMMA.1688 = m16n8k8, HMMA.16816 = m16n8k16).
  - A body holding k times that is a k-fold unroll by ptxas. It counts against k x the weights per iteration, and k is
    recorded as `unroll`.
  - A count that is no whole number of iterations, or an unknown MMA shape, is refused.
  - Without the guard, an unroll would overstate the count k-fold and raise no error.
- **The toolchain is part of the count:** every record names its triton and the bundled ptxas's version. `compare(a,
  b)` gives the ratio of two counts only when both come from one toolchain, and refuses otherwise.
  - A newer triton moves counts further than a decode change does (the table below).
  - The test checks STRUCTURE on any triton: the MMA shape and count, no unroll, and loads that vectorize under the
    hints and scalarize without them. It pins the COUNTS only at triton 3.4.0, the served stack's.
- **Branches:** a loop closes on a backward `BRA`, guarded (`@P0 BRA 0x..`) or with its predicate as an operand
  (`BRA.U UP0, 0x..`, newer ptxas's form on sm_120).

## Recorded counts (triton 3.4.0, its bundled ptxas 12.8.93)

K25 (`nf4_smallm._gemm_nf4_grouped_smallm`) at experts4bit-qlora's served plan: BLOCK_N 32, KC 64, 4 warps, 3 stages,
the select-tree decode, TF32 on fp32 weights. The shapes are Qwen3-30B-A3B's W16 gate_up (K 2048) and down (K 768);
both have the same loop body.

| target | loop instructions | lane instr / weight |
|---|---|---|
| sm_120 (RTX 5090) | 386 | 24.1 |
| sm_86 (RTX A2000) | 402 | 25.1 |

On sm_120, per weight: the select tree (FSEL) 10.3, logic, shift and integer 9.9, shared memory 1.1, the absmax
multiply 1.0, MMA 0.5, global loads 0.4. Both counts are pinned in `test_sass_count.py` on triton 3.4.0.

## The same compile under triton 3.8.0 (ptxas 12.9.86): not comparable, recorded as a finding

| target | served (hints) | bare (no hints) | at 3.4.0, served |
|---|---|---|---|
| sm_120 | 579 instructions, **36.2** per weight | 32.6 | 24.1 |
| sm_86 | 606, **37.9** | 33.8 | 25.1 |

The structure is unchanged: 8 HMMA.1688 a warp, unroll 1, and vectorized loads under the hints. But the served loop
costs about 50 % more per weight, and the hints no longer lower the count. The served stack pins triton 3.4.0, so this
moves nothing served today. A triton bump on the served path would need its own read first.
