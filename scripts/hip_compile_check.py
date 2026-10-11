"""Compile every harvested kernel specialization for AMD targets, CPU only, and fail on any assembler diagnostic.

usage: hip_compile_check.py HARVEST.json [--declared DECLARED.json] [--targets gfx942,gfx950,gfx1100]
                            [--allow ALLOW.json] [--pyproject pyproject.toml] [--cap N] [--report OUT.json]

``HARVEST.json`` comes from ``hip_harvest.py`` (the TRITON_INTERPRET=1 suites, recorded launch by launch);
``DECLARED.json`` holds specializations of kernels the interpreter suites never launch: either a list in the harvest
format, or ``{"source": ..., "specializations": [...], "unreached": {"module.name": "why", ...}}``. ``unreached``
names launched kernels no recorded run could reach, with the reason, so the gap is stated rather than silent; they are
listed under ``declared_unreached`` in the summary. ``slow_specializations`` compile only with ``--include-slow``:
shapes whose AMD compile is too slow for the CI budget, kept so a manual run can still check them.

``triton.compile`` returns normally when the AMD assembler rejects an instruction (PTX inline asm, for one): the
assembler only writes ``error: ...`` to the process's stderr. So stderr is captured at the file-descriptor level
around each compile, and a compile FAILS on an exception or on any ``error:`` line. ROCm's ``ld.lld`` is not needed:
point ``TRITON_HIP_LLD_PATH`` at a copy-only stand-in (``hip_lld_standin.sh``). The result proves lowering and
assembly, never numerics or that the code object loads.

An allowlist entry ``{"kernel": "module.name", "target": "gfx1100", "gate": "<the refusal that keeps this off AMD>"}``
turns a matching failure into ``allowed``. An entry that matches nothing that fails is reported ``stale`` and fails
the check, so the list cannot outlive the reason for it.

Two classes of result. STRICT (exit 1): an exception or assembler ``error:`` outside the allowlist, a stale allowlist
entry, and a kernel named in ``--require`` (``["module.name", ...]``) that no specialization compiled. REPORTING (printed,
never fails): ``uncovered_jit_kernels``, the shipped ``@triton.jit`` functions no specialization reached (device helpers
compiled inside their callers appear here too).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
import time

ERROR_LINE = re.compile(r"^\s*(?:<[^>]*>:\d+:\d+:\s*)?error:", re.M)


def _kernel(module, name):
    obj = importlib.import_module(module)
    if ".<locals>." in name:                              # built inside a no-argument factory (cached per process)
        factory, inner = name.split(".<locals>.", 1)
        for part in factory.split("."):
            obj = getattr(obj, part)
        obj = obj()
        if getattr(getattr(obj, "fn", obj), "__name__", None) != inner.split(".")[-1]:
            raise LookupError(f"{module}.{factory}() did not return {inner}")
        return _unwrap(obj)
    for part in name.split("."):
        obj = getattr(obj, part)
    return _unwrap(obj)


def _unwrap(obj):
    for _ in range(4):                                   # Prebound / Autotuner / Heuristics wrap a JITFunction
        if type(obj).__name__ == "JITFunction":
            return obj
        obj = getattr(obj, "fn", obj)
    return obj


def _compile_capturing(fn, entry, target):
    """``(ok, diagnostics)``: compile one specialization with stderr captured at fd 2."""
    import triton
    from triton.compiler import ASTSource

    src = ASTSource(fn=fn, signature=entry["signature"], constexprs=entry["constexprs"])
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as err:
        os.dup2(err.fileno(), 2)
        exc = None
        try:
            triton.compile(src, target=target, options=dict(entry.get("options") or {}))
        except Exception as e:  # noqa: BLE001 - every failure is a result here
            exc = e
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        err.seek(0)
        text = err.read().decode(errors="replace")
    errors = [ln.strip() for ln in text.splitlines() if ERROR_LINE.match(ln)]
    if exc is not None:
        msg = [ln.strip() for ln in str(exc).splitlines() if ln.strip() and set(ln.strip()) != {"^"}]
        errors.insert(0, f"{type(exc).__name__}: {' | '.join(msg[-2:])[:300]}")
    return not errors, errors


def _compile_job(job):
    """One (specialization, target) compile, self-contained so it can run in a worker process:
    ``(ok, errors, seconds)``."""
    from triton.backends.compiler import GPUTarget

    entry, arch, warp = job
    t0 = time.time()
    try:
        fn = _kernel(entry["module"], entry["name"])
    except Exception as e:  # noqa: BLE001
        return False, [f"import: {e}"], 0.0
    ok, errors = _compile_capturing(fn, entry, GPUTarget("hip", arch, warp))
    return ok, errors, round(time.time() - t0, 2)


def shipped_modules(pyproject):
    import tomllib

    return set(tomllib.load(open(pyproject, "rb"))["tool"]["setuptools"]["py-modules"])


def jit_kernels(kernel_dir, modules):
    """Every ``@triton.jit`` function defined in the shipped modules' source, as ``module.name``."""
    pat = re.compile(r"@triton\.jit[^\n]*\n(?:[ \t]*@[^\n]*\n)*[ \t]*def[ \t]+(\w+)")
    found = set()
    for m in modules:
        path = os.path.join(kernel_dir, m + ".py")
        if os.path.exists(path):
            found |= {f"{m}.{n}" for n in pat.findall(open(path).read())}
    return found


def select(entries, cap):
    """At most ``cap`` specializations per kernel, chosen greedily so every distinct constexpr value, argument type and
    launch option the harvest saw for that kernel is compiled at least once (then the cap is filled in harvest order)."""
    by = {}
    for e in entries:
        by.setdefault((e["module"], e["name"]), []).append(e)
    out = []
    for group in by.values():
        def facts(e):
            return ({("c", k, json.dumps(v, default=repr)) for k, v in e["constexprs"].items()}
                    | {("s", k, v) for k, v in e["signature"].items()}
                    | {("o", k, v) for k, v in (e.get("options") or {}).items()})
        seen, chosen = set(), []
        while len(chosen) < cap:
            best = max((e for e in group if e not in chosen), key=lambda e: len(facts(e) - seen), default=None)
            if best is None or not (facts(best) - seen):
                break
            chosen.append(best)
            seen |= facts(best)
        chosen += [e for e in group if e not in chosen][:max(0, cap - len(chosen))]
        out += chosen
    return out


def main(argv=None):
    from triton.backends.compiler import GPUTarget

    ap = argparse.ArgumentParser()
    ap.add_argument("harvest")
    ap.add_argument("--targets", default="gfx942,gfx950,gfx1100")
    ap.add_argument("--declared", default=None)
    ap.add_argument("--allow", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--pyproject", default=None, help="restrict to its py-modules and report uncovered jit kernels")
    ap.add_argument("--cap", type=int, default=0, help="specializations per kernel (0 = all)")
    ap.add_argument("--require", default=None, help="JSON list of kernels that must compile for every target")
    ap.add_argument("--include-slow", action="store_true", help="also compile the declared slow_specializations")
    ap.add_argument("--jobs", type=int, default=1, help="compile in this many worker processes")
    args = ap.parse_args(argv)
    declared = json.load(open(args.declared)) if args.declared else []
    if isinstance(declared, dict):
        unreached = declared.get("unreached", {})
        declared = declared.get("specializations", []) + (declared.get("slow_specializations", [])
                                                         if args.include_slow else [])
    else:
        unreached = {}
    harvested = json.load(open(args.harvest)) + declared
    modules = shipped_modules(args.pyproject) if args.pyproject else None
    entries = [e for e in harvested if modules is None or e["module"] in modules]
    excluded = sorted({f"{e['module']}.{e['name']}" for e in harvested if e not in entries})
    if args.cap:
        entries = select(entries, args.cap)
    allow = json.load(open(args.allow)) if args.allow else []
    warp = {"gfx942": 64, "gfx950": 64, "gfx90a": 64}
    targets = [GPUTarget("hip", t, warp.get(t, 32)) for t in args.targets.split(",")]
    rows, t_start = [], time.time()
    jobs = [(entry, t.arch, t.warp_size) for entry in entries for t in targets]
    if args.jobs > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(_compile_job, jobs))
    else:
        results = [_compile_job(job) for job in jobs]
    for (entry, arch, _warp), (ok, errors, seconds) in zip(jobs, results):
        kernel = f"{entry['module']}.{entry['name']}"
        status = "ok" if ok else "fail"
        if not ok and any(a["kernel"] == kernel and a["target"] in (arch, "*") for a in allow):
            status = "allowed"
        rows.append({"kernel": kernel, "target": arch, "status": status, "errors": errors[:3],
                     "seconds": seconds, "constexprs": entry["constexprs"]})
    failing = {(r["kernel"], r["target"]) for r in rows if r["status"] == "allowed"}
    stale = [a for a in allow if not any(k == a["kernel"] and (a["target"] in (t, "*")) for k, t in failing)]
    # a factory-built kernel is defined (and found by jit_kernels) as module.<inner name>
    covered = {f"{e['module']}.{e['name'].rsplit('.', 1)[-1]}" for e in entries}
    required = json.load(open(args.require)) if args.require else []
    compiled = {(r["kernel"], r["target"]) for r in rows if r["status"] == "ok"}
    missing = [f"{k} @ {t.arch}" for k in required for t in targets if (k, t.arch) not in compiled]
    summary = {"entries": len(entries), "kernels": len({r["kernel"] for r in rows}),
               "compiles": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
               "allowed": sum(r["status"] == "allowed" for r in rows),
               "failed": sum(r["status"] == "fail" for r in rows), "stale_allow": stale, "required_missing": missing,
               "excluded_not_shipped": excluded, "declared_unreached": unreached,
               "uncovered_jit_kernels": (sorted(jit_kernels(os.path.join(os.path.dirname(os.path.abspath(args.pyproject)),
                                                                     "kernel"), modules) - covered) if modules else None),
               "seconds": round(time.time() - t_start, 1)}
    if args.report:
        json.dump({"summary": summary, "rows": rows}, open(args.report, "w"), indent=1, default=repr)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['status'].upper():8s} {r['kernel']} @ {r['target']}: {' || '.join(r['errors'])[:300]}")
    print(json.dumps(summary))
    return 1 if summary["failed"] or stale or missing else 0


if __name__ == "__main__":
    sys.exit(main())
