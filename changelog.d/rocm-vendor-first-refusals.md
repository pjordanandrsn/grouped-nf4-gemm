### On a ROCm build of torch, the capability-gated NVIDIA paths refuse in words that name AMD

ROCm answers `torch.cuda.get_device_capability` with gfx numbers (MI300X / gfx942 is `(9, 4)`), which pass
the sm-number rules these paths were written against. No AMD card has run this package's suites, so each gate now
asks the vendor first (`_triton_shim.rocm_refusal`, keyed on `torch.version.hip`):

- `fp8_paged_attn.fp8_compute_unsupported` returns an AMD reason before any other check, so the unset
  `GNF4_ATTN_COMPUTE` default selects the f32 path, whose e4m3 decode is integer arithmetic. An explicit
  `GNF4_ATTN_COMPUTE=fp8` reaches the same refusal.
- `fp8_kv.fp8_kv_append_t1` / `fp8_kv_append_bt1` raise on ROCm (after argument validation), pointing at the eager
  `quantize_kv_fp8` append.
- The prebound launch path (`_triton_shim.prebind`, `GNF4_TRITON_PREBIND`) stays off on ROCm even when forced: it was
  read against the CUDA launcher only. `prebind_unsupported()` says why.
- `nf4_route.route_for` takes `vendor` (`"nvidia"`, `"amd"`, or `None` for the running torch's). On AMD, `auto` never
  selects `torch._grouped_mm` from a capability tuple, and an explicit `grouped_mm` is refused in words; a route the
  caller names otherwise is used as given.

NVIDIA behaviour is unchanged. `kernel/test_rocm_refusals.py` (CPU, mocked ROCm) pairs each refusal with the same
call under an NVIDIA build at the same tuple, and runs in the CPU CI job.
