"""pytest plugin: record every kernel specialization the TRITON_INTERPRET=1 suites launch, for the HIP compile check.

Load with ``-p hip_harvest`` and ``HIP_HARVEST_OUT=<path>``. The interpreter runs the kernel's Python body on CPU, so
it sees the real arguments of each launch: tensor dtypes, integer values, constexpr values and launch options. Each
distinct (kernel, signature, constexprs, options) is written once, as data the compile step can rebuild with
``ASTSource`` -- no GPU is needed to learn what the kernels are compiled with.
"""

from __future__ import annotations

import inspect
import json
import os

_RECORDS: dict = {}
_OPTION_KEYS = ("num_warps", "num_stages", "num_ctas")


def _dtype(t) -> str:
    import torch

    names = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.float64: "fp64",
             torch.int8: "i8", torch.int16: "i16", torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8",
             torch.bool: "i1"}
    for name in ("float8_e4m3fn", "float8_e5m2"):
        if hasattr(torch, name):
            names[getattr(torch, name)] = {"float8_e4m3fn": "fp8e4nv", "float8_e5m2": "fp8e5"}[name]
    for name in ("uint16", "uint32", "uint64"):
        if hasattr(torch, name):
            names[getattr(torch, name)] = "u" + name[4:]
    return names.get(t.dtype, str(t.dtype))


def _scalar(v) -> str | None:
    if isinstance(v, bool):
        return "i1"
    if isinstance(v, int):
        return "i32" if -(2 ** 31) <= v < 2 ** 31 else "i64"
    if isinstance(v, float):
        return "fp32"
    return None


def _record(fn, args, kwargs):
    import torch

    params = list(inspect.signature(fn).parameters.values())
    bound = dict(zip((p.name for p in params), args))
    bound.update({k: v for k, v in kwargs.items() if k not in _OPTION_KEYS and k != "grid"})
    signature, constexprs = {}, {}
    for p in params:
        constexpr = "constexpr" in str(p.annotation)
        v = bound.get(p.name, p.default)
        if v is inspect.Parameter.empty:
            return                                     # a launch the harvest cannot describe; skip, never guess
        if constexpr or v is None:
            signature[p.name] = "constexpr"
            constexprs[p.name] = v if isinstance(v, (int, float, bool, str, type(None))) else repr(v)
        elif isinstance(v, torch.Tensor):
            signature[p.name] = "*" + _dtype(v)
        elif _scalar(v) is not None:
            signature[p.name] = _scalar(v)
        else:
            return
    options = {k: kwargs[k] for k in _OPTION_KEYS if k in kwargs}
    entry = {"module": fn.__module__, "name": fn.__qualname__, "signature": signature, "constexprs": constexprs,
             "options": options}
    _RECORDS.setdefault(json.dumps(entry, sort_keys=True, default=repr), entry)


def pytest_configure(config):
    if os.environ.get("TRITON_INTERPRET") != "1":
        return
    from triton.runtime import interpreter

    original = interpreter.InterpretedFunction.__getitem__

    def getitem(self, grid):
        executor = original(self, grid)
        call = executor.__call__

        def launch(*args, **kwargs):
            try:
                _record(self.fn, args, kwargs)
            except Exception:
                pass                                   # recording never changes what the test sees
            return call(*args, **kwargs)

        return launch

    interpreter.InterpretedFunction.__getitem__ = getitem


def pytest_unconfigure(config):
    out = os.environ.get("HIP_HARVEST_OUT")
    if out and _RECORDS:
        existing = json.load(open(out)) if os.path.exists(out) else []
        merged = {json.dumps(e, sort_keys=True, default=repr): e for e in existing + list(_RECORDS.values())}
        with open(out, "w") as f:
            json.dump(sorted(merged.values(), key=lambda e: (e["module"], e["name"])), f, indent=1, default=repr)
