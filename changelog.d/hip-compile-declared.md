### hip-compile: GPU-recorded specializations, and the AMD default attention kernels are required

`scripts/hip_declared.json` now holds 24 specializations of four kernels the interpreter suites never launch, recorded
at `JITFunction.run` during four GPU suites on an RTX A2000 (sm_86): the f32 decode kernels `_fp8_paged_decode_split`,
`_fp8_paged_decode_packed` and `_fp8_combine`, and `nf4_route._dequant_groups_kernel`. Its `unreached` map names the
launched kernels no recorded run could reach, with the reason: `_gemv_nf4_dotpad` and `_gemv_nf4_dotpad_splitk`
engage only at >= 160 SMs, and the fp8-dot attention variants need sm_89+ and are refused on AMD.
`scripts/hip_required.json` lists the three f32 decode kernels, AMD's default attention path: the job fails if any of
them does not compile for gfx942, gfx950 and gfx1100. `hip_compile_check.py` accepts the declared file as an object
and reports `declared_unreached`.

The head_dim 256/512 shapes of the two f32 decode kernels sit under `slow_specializations` and compile only with
`--include-slow`: their AMD compiles take 160-1780 s each on 4 CPUs. They compiled OK on all three targets once
(2026-10-11, triton 3.4.0); CI compiles the D=64/128 shapes of the same kernels. The check gains `--jobs N` and the job
compiles in 4 worker processes.
