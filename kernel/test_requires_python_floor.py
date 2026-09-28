# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The declared Python floor holds for every shipped module, statically.

CI runs 3.11 only, so nothing executes the shipped modules on the floor that
``requires-python`` declares. Two things break there without any 3.11 test
noticing:

* grammar newer than the floor (``match``, parenthesised context managers), and
* PEP 604 ``X | None`` in an annotation that is EVALUATED at import -- a
  function signature, or an annotated assignment at module or class scope --
  in a module without ``from __future__ import annotations``. Below 3.10 that
  raises ``TypeError`` when the ``def`` or ``class`` statement runs.

An annotation on an assignment inside a function body is never evaluated, so
it is not flagged. ``int4_smallm.py`` failed the second rule until its future
import was added, which made ``import int4_smallm`` raise on 3.9.
"""
import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _floor():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    m = re.fullmatch(r">=\s*3\.(\d+)", cfg["project"]["requires-python"].strip())
    assert m, "requires-python is not a plain >=3.N floor; teach this test the new form"
    return cfg, (3, int(m.group(1)))


def _shipped(cfg):
    st = cfg["tool"]["setuptools"]
    base = ROOT / st.get("package-dir", {}).get("", ".")
    files = [base / f"{m}.py" for m in st.get("py-modules", [])]
    for pkg in st.get("packages", []):
        files += sorted((ROOT / pkg.replace(".", "/")).glob("*.py"))
    return files


def _evaluated_annotations(tree):
    """Annotations Python evaluates when the module is imported."""
    def visit(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = child.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                    if arg is not None and arg.annotation is not None:
                        yield arg.annotation
                if child.returns is not None:
                    yield child.returns
                yield from visit(child, True)
            elif isinstance(child, ast.AnnAssign):
                if not in_function:
                    yield child.annotation
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, False)
            else:
                yield from visit(child, in_function)
    yield from visit(tree, False)


def _pep604_lines(tree):
    out = set()
    for ann in _evaluated_annotations(tree):
        for n in ast.walk(ann):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
                out.add(n.lineno)
    return sorted(out)


def _has_future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names)
               for n in tree.body)


def test_every_shipped_module_holds_the_declared_python_floor():
    cfg, floor = _floor()
    files = _shipped(cfg)
    assert len(files) > 10, f"found only {len(files)} shipped modules; the pyproject walk is broken"
    missing = [str(f.relative_to(ROOT)) for f in files if not f.is_file()]
    assert not missing, f"py-modules names files that do not exist: {missing}"
    problems = []
    for f in files:
        src = f.read_text()
        try:
            tree = ast.parse(src, feature_version=floor)
        except SyntaxError as e:
            problems.append(f"{f.relative_to(ROOT)}:{e.lineno}: grammar newer than {floor}: {e.msg}")
            continue
        if floor < (3, 10) and not _has_future_annotations(tree):
            for ln in _pep604_lines(tree):
                problems.append(f"{f.relative_to(ROOT)}:{ln}: `X | Y` annotation evaluated at import "
                                f"(add `from __future__ import annotations`)")
    assert not problems, "\n".join(problems)


def test_the_detector_sees_an_evaluated_annotation_and_ignores_a_local_one():
    """Calibration: a check that has only ever passed carries no information."""
    flagged = ast.parse("def f(x: int | None = None) -> None:\n    pass\n")
    local = ast.parse("def f():\n    y: int | None = None\n    self_x: list[int | None] = []\n")
    klass = ast.parse("class C:\n    x: int | None = None\n")
    future = ast.parse("from __future__ import annotations\ndef f(x: int | None): pass\n")
    assert _pep604_lines(flagged) == [1]
    assert _pep604_lines(local) == []
    assert _pep604_lines(klass) == [2]
    assert _has_future_annotations(future) and not _has_future_annotations(flagged)
