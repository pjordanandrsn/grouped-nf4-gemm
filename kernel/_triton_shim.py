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
  so the decorator must succeed) while *launch*-time use raises with the CPU
  alternative **for the module that defined that kernel**.

Two details are load-bearing:

*The fallbacks are per module.* This shim is shared, so a single hard-coded
"use ``dequant_ref``" would send an MXFP4 or a gather caller to an NF4-only API.
``_CPU_PATH`` is keyed by the defining module, and ``host_gather`` gets the
honest answer — a device-side gather over UVA has no CPU equivalent at all.

*Unknown attributes raise ``AttributeError``* rather than returning another
stub. ``nf4_grouped`` probes ``getattr(tl, "gather", None)`` to decide whether
the register-LUT variant exists; raising is what lets that fall back to its
default, so a triton-less box reports ``HAS_TL_GATHER = False`` — the same
answer triton < 3.3 gives — instead of selecting a variant backed by a stub.

The stand-ins are defined unconditionally, and only *bound* in the fallback
branch, so ``test_triton_shim`` can exercise this routing on Linux CI too. Logic
that only exists on the platforms CI does not run is logic nothing checks.
"""

from __future__ import annotations

import functools
import os

__all__ = ["HAS_TRITON", "UnsupportedShapeError", "device_shared_mem_limit", "prebind", "prebind_requested", "tl", "triton"]

_NO_TRITON = (
    "triton is not installed, so this fused kernel cannot run. triton is a "
    "Linux-only dependency of grouped-nf4-gemm and has no wheel for this "
    "platform."
)

#: The CPU alternative is **per module** — see the module docstring. Keyed by
#: the defining module of the kernel that was launched, which is exactly the
#: module whose fallback applies. ``test_triton_shim`` fails if a module that
#: imports this shim is missing an entry.
_CPU_PATH = {
    "nf4_grouped":
        "For a CPU-checkable decode of the same NF4 bytes, use "
        "dequant_ref(packed, absmax, N, K) — the pure-torch reference the "
        "property suite pins the kernel against.",
    "mxfp4_grouped":
        "For a CPU-checkable decode of the same MXFP4 bytes, use "
        "mxfp4_pack_ref.dequant_mxfp4(blocks, scales) — the pure-torch "
        "reference the interpreter-parity gate gates this kernel on.",
    "host_gather":
        "This primitive has no CPU equivalent: it is a device-side gather "
        "reading pinned host memory over UVA, so the fallback is to not "
        "prefetch rather than to compute the same thing differently.",
    "fp8_paged_attn":
        "For a CPU-checkable decode attention over the same fp8 KV bytes, "
        "use fp8_paged_attn.paged_attn_ref(...) — the pure-torch oracle the "
        "shape suite pins every kernel variant against.",
    "int4_smallm":
        "For a CPU-checkable result over the same int4-b32 bytes, use "
        "x.bfloat16() @ int4_pack_ref.dequant_int4_ref(packed, scales, N, K).T "
        "— the pure-torch reference test_int4_smallm_interp pins the small-M "
        "GEMM against (the interpreter job runs that file on CPU).",
    "nf4_smallm":
        "For a CPU-checkable result over the same NF4 bytes, use "
        "x.float() @ nf4_grouped.dequant_ref(packed[e], absmax[e], N, K).T per "
        "row (rounded to bfloat16 for the compiled kernel's weight operand) — "
        "the pure-torch reference test_nf4_grouped_smallm_interp pins K25 "
        "against (the interpreter job runs that file on CPU).",
    "k26_bench":
        "A campaign instrument (lane K26), not library API: its kernels are "
        "decode ablations of nf4_smallm's, with no CPU equivalent. Only its "
        "rule runs on CPU (python k26_bench.py --self-test); for a "
        "CPU-checkable NF4 product use nf4_grouped.dequant_ref.",
    "nf4_route":
        "For a CPU-checkable result over the same NF4 bytes, use "
        "x.float() @ nf4_grouped.dequant_ref(packed[e], absmax[e], N, K).to(torch.bfloat16).float().T "
        "per group (the forward; grad_out @ that weight for the dgrad) -- the route's dequant kernel is "
        "bit-equal to dequant_ref rounded to bfloat16 (test_nf4_route), and torch._grouped_mm has no CPU path.",
    "int4_b32":
        "For a CPU-checkable result over the same int4-b32 bytes, use "
        "(xq.float() * xs.repeat_interleave(32, dim=1)) @ "
        "int4_pack_ref.dequant_int4_ref(packed[e], scales[e], N, K).T per row "
        "— the reference test_expert_offset_boundary pins every int4_b32 "
        "kernel against. int4_b32 imports triton itself (it takes only "
        "UnsupportedShapeError from this shim), so without triton it fails "
        "at import, before any launch could reach this message.",
}

#: A consumer added later without an entry gets a message that is vague but
#: never WRONG — misdirection is the failure mode being avoided here.
_CPU_PATH_UNKNOWN = (
    "That module's pure-torch reference, where it has one, is the "
    "CPU-checkable path."
)


class _UnlaunchableKernel:
    """A ``@triton.jit`` kernel on a box with no triton.

    Kernels are defined at import, so decorating must succeed; only a launch can
    fail. Triton launches are subscripted — ``kernel[grid](...)`` — so
    ``__getitem__`` returns self and the call raises, naming the CPU alternative
    for the defining module rather than surfacing an ``AttributeError`` on a
    stub.
    """

    def __init__(self, fn):
        self._fn = fn
        functools.update_wrapper(self, fn)

    def __getitem__(self, _grid):
        return self

    def __call__(self, *_args, **_kwargs):
        # `__module__` is the module that DEFINED the kernel, so a shared shim
        # still names the right fallback for the caller in hand.
        mod = (getattr(self._fn, "__module__", "") or "").rsplit(".", 1)[-1]
        raise RuntimeError(
            f"{self._fn.__name__}: {_NO_TRITON} {_CPU_PATH.get(mod, _CPU_PATH_UNKNOWN)}"
        )


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


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ModuleNotFoundError:  # no triton wheel for this platform (e.g. macOS)
    HAS_TRITON = False
    triton = _MissingModule("triton")
    tl = _MissingModule("triton.language")


class UnsupportedShapeError(ValueError):
    """A launch geometry a kernel cannot run on this device, refused BEFORE
    the launch (or converted from Triton's launch-time ``OutOfResources``),
    with the numbers a caller can act on.

    Raised where a tile configuration's shared-memory need exceeds the
    device's limit and no smaller supported configuration exists, so a
    known-unsupported shape surfaces as a typed error naming the shape and
    the limit rather than as a Triton runtime failure (gnf4#324). A
    ``ValueError`` so the existing "bad argument" handling of callers still
    catches it; the attributes carry what the message says.
    """

    def __init__(self, kernel: str, shape: dict, need_bytes: int,
                 limit_bytes: int, hint: str = ""):
        self.kernel = kernel
        self.shape = dict(shape)
        self.need_bytes = int(need_bytes)
        self.limit_bytes = int(limit_bytes)
        self.hint = hint
        geom = ", ".join(f"{k}={v}" for k, v in self.shape.items())
        msg = (f"{kernel}: geometry ({geom}) needs {self.need_bytes} bytes of "
               f"shared memory against this device's limit of "
               f"{self.limit_bytes} bytes")
        if hint:
            msg += f"; {hint}"
        super().__init__(msg)


_SHARED_MEM_LIMIT: dict = {}


def device_shared_mem_limit(device=None) -> int:
    """Per-device shared-memory (LDS) limit in bytes, queried ONCE per device
    through triton's driver and cached; 0 when unqueryable.

    NVIDIA SMs expose 100-228 KB; CDNA3 (MI300X, gfx942) exposes 64 KB.
    0 means "no answer" -- no triton, no CUDA/HIP device, interpreter mode,
    or a driver that does not expose ``max_shared_mem`` -- and every
    feasibility check treats 0 as "do not pre-refuse": the CPU and
    ``TRITON_INTERPRET`` paths are never perturbed, and a launch that then
    overflows is still caught at the launch and reported as
    :class:`UnsupportedShapeError` or a fallback by the calling wrapper.
    Any failure -> 0.
    """
    if not HAS_TRITON:
        return 0
    try:
        import torch
        idx = getattr(device, "index", None)
        if idx is None:
            if not torch.cuda.is_available():
                return 0
            idx = torch.cuda.current_device()
        if idx not in _SHARED_MEM_LIMIT:
            props = triton.runtime.driver.active.utils.get_device_properties(idx)
            _SHARED_MEM_LIMIT[idx] = int(props["max_shared_mem"])
        return _SHARED_MEM_LIMIT[idx]
    except Exception:
        return 0


# --- GNF4_TRITON_PREBIND (on by default since experts4bit-qlora TC1 amendments 26 / 30; =0 turns it off) ------
#
# A Triton launch, ``kernel[grid](...)``, spends most of its host time before the driver call: it binds the arguments to the
# signature, specializes each one (dtype, 16-byte alignment, ``== 1`` and ``% 16`` of integers), formats that into a string key,
# looks the compiled kernel up and builds launch metadata -- 30-90 us a launch under triton 3.4 on an RTX A2000 host, against a
# 4-7 us driver call. ``prebind`` wraps a kernel so that the FIRST launch of each specialization goes through Triton unchanged and
# the compiled kernel it returns is remembered under a key built from the same facts Triton specializes on; later launches with
# that key call the kernel's own launcher directly. Same compiled binary, same arguments, same stream: bit-identical outputs
# (test_triton_prebind). A Triton release this was not read against, a launch hook (profilers), a pre-run hook, a callable grid, a
# changed global, or an argument of another type takes Triton's own launch. One knob is read less often than Triton reads it:
# triton 3.4 re-reads TRITON_DEBUG at every launch, the prebound path once per kernel (3.6 itself reads it once, at import). On by
# default, read when the kernel's module is imported; GNF4_TRITON_PREBIND=0 turns it off, and off ``prebind`` returns the kernel itself.
# The default follows experts4bit-qlora's TC1 amendments 26 and 30 (one RTX 5090 each, triton 3.4): the training step at 0.973x (matched
# arm) and 0.980x (shipped arm, 60 steps) of the flag off, held-out within 0.0012.

#: Triton releases whose launch protocol (``JITFunction.run`` -> ``CompiledKernel.run``) the prebound path was read against.
PREBIND_TRITON = ((3, 4), (3, 6))
#: Launches through the prebound path, and through Triton's own (the first launch of a key, or a fallback).
PREBIND_STATS = {"prebound": 0, "triton": 0}
_PREBIND_MAX_KEYS = 4096           # integer values are keyed exactly; a sweep over sizes must not grow it forever
_PREBIND_PLAIN = (int, float, bool, type(None))
_MISSING = object()


def prebind_requested() -> bool:
    return os.environ.get("GNF4_TRITON_PREBIND", "1").strip() != "0"


def _triton_version():
    try:
        return tuple(int(v) for v in triton.__version__.split(".")[:2])
    except Exception:
        return None


def prebind(fn, force: bool = False):
    """``fn`` wrapped in :class:`Prebound` unless ``GNF4_TRITON_PREBIND=0`` (``force`` wraps regardless), when this Triton is supported; else ``fn``."""
    if not (force or prebind_requested()) or not HAS_TRITON or _triton_version() not in PREBIND_TRITON:
        return fn
    from triton.runtime.jit import JITFunction
    if not isinstance(fn, JITFunction) or not all(hasattr(fn, a) for a in ("params", "used_global_vals", "pre_run_hooks")):
        return fn                  # e.g. TRITON_INTERPRET=1's InterpretedFunction
    return Prebound(fn)


class Prebound:
    """``kernel[grid](*args, **kwargs)`` with Triton's per-launch binding done once per specialization."""

    def __init__(self, fn):
        import torch
        from triton import knobs
        self.fn, self.knobs, self.tensor = fn, knobs, torch.Tensor
        self.names = tuple(p.name for p in fn.params)
        self.defaults = {p.name: p.default for p in fn.params if p.has_default}
        self.kernels = {}
        self.device = self.stream = None       # Triton's own device / stream getters, bound at the first launch
        # 3.6 adds an instrumentation mode to every launch's options. 3.4 re-reads TRITON_DEBUG from the environment at every launch
        # (1.4 us on an RTX A2000 host), 3.6 once at import: read here once per kernel under 3.4, per launch under 3.6.
        self.compilation = getattr(knobs, "compilation", None)
        self.debug = knobs.runtime.debug if _triton_version() < (3, 6) else None

    def __getitem__(self, grid):
        if callable(grid):
            return self.fn[grid]
        return functools.partial(self.launch, grid)

    def _triton(self, grid, args, kwargs):
        PREBIND_STATS["triton"] += 1
        return self.fn[grid](*args, **kwargs)

    def launch(self, grid, *args, **kwargs):
        fn, rt = self.fn, self.knobs.runtime
        # A registered launch hook (3.4: a callable; 3.6: a non-empty HookChain) needs Triton's launch metadata: take its path.
        if getattr(rt.launch_enter_hook, "calls", rt.launch_enter_hook) or getattr(rt.launch_exit_hook, "calls", rt.launch_exit_hook) \
                or fn.pre_run_hooks:
            return self._triton(grid, args, kwargs)
        try:
            # every parameter in signature order (the launcher's argument list), constexprs and defaults included
            full = args + tuple([kwargs[n] if n in kwargs else self.defaults[n] for n in self.names[len(args):]])
            if self.device is None:            # torch's, on CUDA: bound once rather than through the driver proxy per launch
                from triton.runtime.driver import driver
                self.device, self.stream = driver.active.get_current_device, driver.active.get_current_stream
            dev, T = self.device(), self.tensor
            # Triton's key, made exact: a tensor's dtype and 16-byte alignment, an integer's value (which fixes its == 1, % 16 and
            # width), any other argument's type and value; plus the launch options, the device and the debug / instrumentation knobs.
            key = (dev, rt.debug if self.debug is None else self.debug, getattr(self.compilation, "instrumentation_mode", None),
                   tuple(kwargs.items()),
                   tuple([(a.dtype, a.data_ptr() % 16 == 0) if isinstance(a, T) else a if type(a) is int else (type(a), a)
                          for a in full]))
            hit = self.kernels.get(key)
        except (KeyError, TypeError):          # a missing argument, an unhashable one: Triton reports it
            return self._triton(grid, args, kwargs)
        if hit is None:
            kernel = self._triton(grid, args, kwargs)
            # Kept only for plain arguments (a tensor passed by keyword would sit in the key, by identity) and a launchable kernel.
            if (len(full) == len(self.names) and all(isinstance(a, (T,) + _PREBIND_PLAIN) for a in args)
                    and all(type(v) in _PREBIND_PLAIN for v in kwargs.values())
                    and all(hasattr(kernel, a) for a in ("run", "function", "packed_metadata"))):
                if len(self.kernels) >= _PREBIND_MAX_KEYS:
                    self.kernels.clear()
                self.kernels[key] = (kernel, kernel.run, kernel.function, kernel.packed_metadata)
            return kernel
        for (name, _), (val, scope) in fn.used_global_vals.items():
            cur = scope.get(name, _MISSING)
            if cur is not val and cur != val:  # Triton refuses a launch after a global it compiled against changed
                return self._triton(grid, args, kwargs)
        kernel, run, function, metadata = hit
        n = len(grid)
        PREBIND_STATS["prebound"] += 1
        # launch metadata and the enter / exit hooks are None: exactly what Triton passes when no hook is registered
        run(grid[0], grid[1] if n > 1 else 1, grid[2] if n > 2 else 1, self.stream(dev), function, metadata, None, None, None, *full)
        return kernel
