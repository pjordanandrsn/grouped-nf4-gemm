"""Import ``triton``, or a stand-in that keeps this package's CPU paths alive.

``triton`` is a **Linux-only** dependency here — ``pyproject.toml`` pins it as
``triton>=3.4; platform_system == 'Linux'`` — so a supported install on macOS
has no triton at all. Three shipped modules (``nf4_grouped``, ``mxfp4_grouped``,
``host_gather``) define ``@triton.jit`` kernels at import, and a bare
``import triton`` in them took the whole package down on such a box with
``ModuleNotFoundError: No module named 'triton'``.

That failure landed in the wrong place. What this package promises without a GPU
is specific and pure-torch: :func:`nf4_grouped.dequant_ref` (whose docstring
already says "no CUDA/Triton"), the README's CPU quickstart, and above all the
*taught* refusal — "requires CUDA tensors ... use dequant_ref" — that
``test_cpu_refusal`` pins as doctrine. A raw import error preempted every one of
them, so the loud death arrived one import too early to say anything useful, and
the CPU-only tests could not even be collected.

The contract here is narrow on purpose:

* **Present** — bind the real modules and change nothing. Consumers are
  unmodified below their import line, so the CUDA path is untouched by
  construction.
* **Absent** — define-time triton use resolves (kernels are *defined* at import,
  so the decorator must succeed) while *launch*-time use raises with the same
  instruction the device guard gives.

The one deliberate subtlety is that unknown attributes raise ``AttributeError``
rather than returning a stub. ``nf4_grouped`` probes ``getattr(tl, "gather",
None)`` to decide whether the register-LUT variant exists; raising is what lets
that fall back to its default, so a triton-less box reports ``HAS_TL_GATHER =
False`` — the same answer triton < 3.3 gives — instead of selecting a variant
backed by a stub.
"""

from __future__ import annotations

import functools

__all__ = ["HAS_TRITON", "tl", "triton"]

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ModuleNotFoundError:  # no triton wheel for this platform (e.g. macOS)
    HAS_TRITON = False

    _NO_TRITON = (
        "triton is not installed, so the fused kernel cannot run. triton is a "
        "Linux-only dependency of grouped-nf4-gemm; on this platform use the "
        "pure-torch path — dequant_ref(packed, absmax, N, K) — which decodes "
        "the same NF4 bytes and is what the property suite pins the kernel "
        "against."
    )

    class _UnlaunchableKernel:
        """A ``@triton.jit`` kernel on a box with no triton.

        Kernels are defined at import, so decorating must succeed; only a launch
        can fail. Triton launches are subscripted — ``kernel[grid](...)`` — so
        ``__getitem__`` returns self and the call raises, naming the same CPU
        alternative the device guard does rather than surfacing an
        ``AttributeError`` on a stub.
        """

        def __init__(self, fn):
            self._fn = fn
            functools.update_wrapper(self, fn)

        def __getitem__(self, _grid):
            return self

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError(f"{self._fn.__name__}: {_NO_TRITON}")

    class _MissingModule:
        """Stand-in for ``triton`` / ``triton.language``."""

        #: Kernel signatures annotate with ``tl.constexpr``. Every consumer has
        #: ``from __future__ import annotations``, so these are strings and are
        #: never evaluated — but binding it explicitly means dropping that
        #: future-import later cannot turn this shim into an import error.
        constexpr = object()

        def __init__(self, name):
            self._name = name

        def __getattr__(self, attr):
            # Raising is load-bearing: see the module docstring on
            # `getattr(tl, "gather", None)`.
            raise AttributeError(f"{self._name}.{attr}: {_NO_TRITON}")

        @staticmethod
        def jit(fn=None, **_kwargs):
            def wrap(f):
                return _UnlaunchableKernel(f)

            return wrap if fn is None else wrap(fn)

    triton = _MissingModule("triton")
    tl = _MissingModule("triton.language")
