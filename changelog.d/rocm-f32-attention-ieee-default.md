### On a ROCm build, the f32 decode path defaults to ieee precision (tf32 does not compile on gfx950/gfx1100 under triton 3.4)

This is a default change on ROCm only. With `GNF4_ATTN_F32_PRECISION` unset, `fp8_paged_attn._f32_dot_precision` now
returns `ieee` on a ROCm build of torch, and `tf32` everywhere else as before; an explicit value is used as given. The
f32 split decode kernel is the attention path AMD reaches by default, since fp8 compute is refused there. Compiled for
AMD with triton 3.4.0 (the `hip-compile` job), its dots lower `tf32` on gfx942 only, `tf32x3` on none of gfx942,
gfx950 and gfx1100, and `ieee` on all three. `ieee` is exact and leaves the tensor cores; no AMD speed claim is made.
