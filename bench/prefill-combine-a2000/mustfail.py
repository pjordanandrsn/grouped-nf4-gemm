"""Must-fail control for the PR's tests: swap `_index_add_ordered_` for the plain
`out.index_add_(0, rows, src)` the prefill path used, and run the CUDA tests that should
then fail. A test that still passed here could not see the bug. Run from the PR's kernel/.
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, os.getcwd())   # the PR's kernel/, not this script's dir
import test_mxfp4_prefill_combine as t  # noqa: E402

t._index_add_ordered_ = lambda out, rows, src: out.index_add_(0, rows, src)
cases = [("test_repeated_cuda_calls_are_bitwise_identical", (6, 96)),
         ("test_repeated_cuda_calls_are_bitwise_identical", (90, 1440)),
         ("test_equals_the_sequential_loop_bitwise", ("cuda", 6, 96)),
         ("test_equals_the_sequential_loop_bitwise", ("cuda", 90, 1440)),
         ("test_equals_the_sequential_loop_bitwise", ("cuda", 1, 7))]
print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  helper -> plain index_add_")
for name, args in cases:
    try:
        getattr(t, name)(*args)
        print(f"  PASSED (control did NOT fail)  {name}{args}")
    except AssertionError:
        print(f"  FAILED as expected             {name}{args}")
    except Exception:
        print(f"  ERROR                          {name}{args}\n{traceback.format_exc()}")
