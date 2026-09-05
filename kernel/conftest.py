# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Keep interpreter-mode tests out of a compiled-mode pytest process.

``TRITON_INTERPRET`` is read by triton when it is first imported and latches
for the life of the PROCESS. Two files here turn it on at import time
(``test_interp_contract.py``, ``test_mxfp4_interp.py``) so they can validate
kernel semantics on CPU with no GPU. Every other kernel test needs the normal
compiled path.

Both constraints are documented, but nothing enforced them, so a plain
``pytest kernel/`` collected an interpreter file, flipped the global knob, and
then FATALLY crashed a later compiled test with "Cannot call @triton.jit'd
outside of the scope of a kernel" — a process abort with a stack dump, not a
test failure. That is a confusing way to learn about a known constraint.

A fixture cannot fix this: by the time any test runs, triton has already read
the variable. So this refuses the mixed run up front and says how to split it,
which is the honest resolution of a genuinely process-global flag.

**Only when a CUDA device is actually present.** The crash needs a test that
launches a real kernel, which needs a device. With no device those tests skip,
so mixing is harmless — and that is exactly CI's situation: the
"CPU-reachable suites" step deliberately runs ``test_mxfp4_interp.py``
alongside eight compiled-path files and passes (verified: 55 passed, 9
skipped). Refusing on filenames alone would break that green step. Gating on
the device keeps CI working and still catches the real hazard.

The gate is deliberately coarser than "will this test launch a kernel", which
is not knowable at collection time. So on a GPU box this can still refuse a
combination that would have been fine — the eight CPU-only files above are the
known example. That is the conservative direction (a clear message with a
bypass, instead of a process abort), but it is why the bypass exists.

Set ``GNF4_ALLOW_MIXED_INTERP=1`` to bypass (for that case, or for debugging
the interaction itself).
"""
import os

import pytest

_INTERP_FILES = {"test_interp_contract.py", "test_mxfp4_interp.py",
                 "test_mxfp4_gemv_b32.py", "test_shape_feasibility.py"}


def _device_present():
    """True only if a CUDA device is really usable.

    torch is NOT a hard dependency of this suite (CPU CI installs just triton,
    pytest and numpy), so a bare import here would turn a missing optional dep
    into a collection error for every test. No torch and no device both mean
    the compiled tests skip, which is the safe answer either way.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if os.environ.get("GNF4_ALLOW_MIXED_INTERP") == "1":
        return
    if not _device_present():
        return
    names = {os.path.basename(str(i.fspath)) for i in items}
    interp = names & _INTERP_FILES
    compiled = names - _INTERP_FILES
    if not interp or not compiled:
        return                      # a pure run either way is fine
    raise pytest.UsageError(
        "TRITON_INTERPRET is process-global and latches when triton is first "
        f"imported, so {sorted(interp)} cannot share a pytest process with the "
        f"{len(compiled)} compiled-path test file(s). Running them together "
        "aborts the process rather than failing a test.\n\n"
        "Run them separately:\n"
        f"    pytest kernel/ --ignore-glob='*{'* --ignore-glob=*'.join(sorted(interp))}*'\n"
        f"    pytest {' '.join('kernel/' + f for f in sorted(interp))}\n\n"
        "Set GNF4_ALLOW_MIXED_INTERP=1 to bypass this check."
    )
