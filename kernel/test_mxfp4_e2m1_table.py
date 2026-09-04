"""The branchless e2m1 decode used by ``_gemv_mxfp4_b32`` -- ``2*value``
as an integer: ``m`` for e == 0, else ``(2 + m) << (e - 1)`` -- equals
the codebook on all 16 nibbles. Pure python; runs on every platform (the
kernel module imports through the triton shim)."""
from mxfp4_grouped import FP4_VALUES


def test_e2m1_decode_table():
    for nib in range(16):
        e, m = (nib >> 1) & 3, nib & 1
        v2 = m if e == 0 else (2 + m) << max(e - 1, 0)
        if nib & 8:
            v2 = -v2
        assert v2 * 0.5 == FP4_VALUES[nib], (nib, v2 * 0.5, FP4_VALUES[nib])
