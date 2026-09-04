#!/usr/bin/env python3
"""Change-impact contract: the companion changes a class of change requires,
detected from a git diff. Standard library plus git; no network.

    python scripts/check_change_impact.py --base origin/main
    python scripts/check_change_impact.py --base <sha> --allow-claims-only

The diff is ``git merge-base <base> HEAD`` against the working tree (so a
local run sees uncommitted, tracked changes; in CI the tree is the PR head).
The classes and their companions are the contract in docs/change-impact.json;
this script implements the triggers a diff can show:

  new-kernel-capability  a module added to pyproject py-modules, or a new
                         @triton.jit kernel in a shipped module -> FAIL unless
                         docs/capabilities.json and CHANGELOG.md changed too;
                         a layout constant changed -> WARN unless
                         docs/KERNEL_CONTRACT.md changed
  public-api-change      the entrypoint set in docs/capabilities.json changed
                         -> FAIL unless CHANGELOG.md changed; WARN on README.md
  measured-result        a claim added, or its status/value/unit changed, in
                         docs/claims.json -> FAIL unless docs/STATUS.md changed
                         (--allow-claims-only downgrades to WARN)
  dependency-floor       the pyproject version changed -> FAIL unless
                         CHANGELOG.md changed; dependencies changed -> WARN on
                         README.md and docs/capabilities.json

Prints one line per triggered class (class, trigger, missing companions) and
OK when nothing is missing. Exit 1 on any FAIL, 2 when git cannot answer.
Without --base the check is skipped cleanly (exit 0), which is what a
non-pull-request CI event gets.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

LAYOUT_WORDS = re.compile(r"(BLOCK|GROUP|STRIDE|ALIGN|LAYOUT|NIBBLE|PACK|ROW|TILE|WIDTH|BYTES)")
CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::\s*[^=]+)?=\s*(.+?)\s*(?:#.*)?$", re.M)
JIT_KERNEL = re.compile(r"^@triton\.jit(?:\([^)]*\))?\s*\n(?:^@[^\n]*\n)*^def\s+(\w+)", re.M)


class GitError(Exception):
    pass


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def _show(root: Path, ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _head_text(root: Path, path: str) -> str | None:
    p = root / path
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _py_modules(pyproject_text: str | None) -> set[str]:
    if not pyproject_text:
        return set()
    st = tomllib.loads(pyproject_text).get("tool", {}).get("setuptools", {})
    return set(st.get("py-modules") or []) | set(st.get("packages") or [])


def _project(pyproject_text: str | None) -> dict:
    return tomllib.loads(pyproject_text).get("project", {}) if pyproject_text else {}


def _module_path(pyproject_text: str, module: str) -> str:
    st = tomllib.loads(pyproject_text).get("tool", {}).get("setuptools", {})
    pkg_dir = st.get("package-dir") or {}
    if module in pkg_dir:
        return f"{pkg_dir[module]}/__init__.py"
    base = pkg_dir.get("", "")
    return f"{base}/{module}.py" if base else f"{module}.py"


def _constants(text: str | None) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in CONSTANT.finditer(text or "") if LAYOUT_WORDS.search(m.group(1))}


def _kernels(text: str | None) -> set[str]:
    return set(JIT_KERNEL.findall(text or ""))


def _claims(text: str | None) -> dict[str, tuple]:
    if not text:
        return {}
    doc = json.loads(text)
    claims = doc.get("claims", doc) if isinstance(doc, dict) else doc
    return {c["id"]: (c.get("status"), json.dumps(c.get("value")), c.get("unit")) for c in claims if c.get("id")}


def _entrypoints(text: str | None) -> set[tuple[str, str]]:
    if not text:
        return set()
    doc = json.loads(text)
    return {(c["id"], ep) for c in doc.get("capabilities", []) for ep in c.get("entrypoints", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--base", default=None, metavar="REF", help="git ref to diff against (skipped cleanly when absent)")
    ap.add_argument("--contract", default="docs/change-impact.json")
    ap.add_argument("--allow-claims-only", action="store_true",
                    help="a claims.json change without docs/STATUS.md is a WARN, not a FAIL")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if not a.base:
        print("SKIP: no --base given (not a pull request); change-impact check not run")
        return 0
    contract = json.loads((root / a.contract).read_text(encoding="utf-8"))
    known = {c["id"] for c in contract["classes"]}
    try:
        mb = _git(root, "merge-base", a.base, "HEAD").strip()
        changed = set()
        for line in _git(root, "diff", "--name-only", mb).splitlines():
            if line.strip():
                changed.add(line.strip())
    except GitError as e:
        print(f"FAIL: cannot compute the diff: {e}")
        return 2
    untracked = [ln for ln in _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
                 if ln.startswith(("kernel/", "docs/", "pyproject.toml", "CHANGELOG.md"))]

    findings: list[tuple[str, str, str, list[str]]] = []       # (level, class, trigger, missing)

    def report(level: str, cls: str, trigger: str, required: list[str]) -> None:
        if cls not in known:
            findings.append(("FAIL", cls, f"class not in {a.contract}", []))
        missing = [p for p in required if p not in changed]
        findings.append((level if missing else "OK", cls, trigger, missing))

    # -- pyproject: modules, version, dependencies ------------------------------
    base_py, head_py = _show(root, mb, "pyproject.toml"), _head_text(root, "pyproject.toml")
    if "pyproject.toml" in changed:
        new_modules = sorted(_py_modules(head_py) - _py_modules(base_py))
        for mod in new_modules:
            report("FAIL", "new-kernel-capability", f"module {mod!r} added to pyproject py-modules",
                   ["docs/capabilities.json", "CHANGELOG.md"])
        bp, hp = _project(base_py), _project(head_py)
        if bp.get("version") != hp.get("version"):
            report("FAIL", "dependency-floor", f"pyproject version {bp.get('version')} -> {hp.get('version')}", ["CHANGELOG.md"])
        if (bp.get("dependencies") or []) != (hp.get("dependencies") or []):
            report("WARN", "dependency-floor", "pyproject dependencies changed", ["README.md", "docs/capabilities.json"])

    # -- shipped modules: new kernels, layout constants -------------------------
    shipped = _py_modules(head_py)
    for mod in sorted(shipped):
        path = _module_path(head_py or "", mod)
        if path not in changed:
            continue
        base_t, head_t = _show(root, mb, path), _head_text(root, path)
        if base_t is None:
            continue                                    # a new module: handled via py-modules above
        new_k = sorted(_kernels(head_t) - _kernels(base_t))
        if new_k:
            report("FAIL", "new-kernel-capability", f"new @triton.jit kernel(s) {new_k} in {path}",
                   ["docs/capabilities.json", "CHANGELOG.md"])
        bc, hc = _constants(base_t), _constants(head_t)
        moved = sorted(k for k in hc if bc.get(k) != hc[k])
        if moved:
            report("WARN", "new-kernel-capability", f"layout constant(s) {moved} changed in {path}", ["docs/KERNEL_CONTRACT.md"])

    # -- capabilities: entrypoint set -------------------------------------------
    if "docs/capabilities.json" in changed:
        b, h = _entrypoints(_show(root, mb, "docs/capabilities.json")), _entrypoints(_head_text(root, "docs/capabilities.json"))
        added, removed = sorted(h - b), sorted(b - h)
        if added or removed:
            trig = f"entrypoints added {[e for _, e in added]} removed {[e for _, e in removed]}"
            report("FAIL", "public-api-change", trig, ["CHANGELOG.md"])
            report("WARN", "public-api-change", trig + " (entry-point table)", ["README.md"])

    # -- claims: added / status / value ----------------------------------------
    if "docs/claims.json" in changed:
        b, h = _claims(_show(root, mb, "docs/claims.json")), _claims(_head_text(root, "docs/claims.json"))
        added = sorted(k for k in h if k not in b)
        moved = sorted(k for k in h if k in b and b[k] != h[k])
        if added or moved:
            report("WARN" if a.allow_claims_only else "FAIL", "measured-result",
                   f"claims added {added} / status-value-unit changed {moved}", ["docs/STATUS.md"])

    for level, cls, trigger, missing in findings:
        if missing:
            print(f"{level}: class={cls} trigger={trigger} missing={missing}")
        else:
            print(f"OK: class={cls} trigger={trigger} companions present")
    if untracked:
        print(f"WARN: untracked files are invisible to this diff (git add them): {untracked[:8]}")
    n_fail = sum(1 for f in findings if f[0] == "FAIL")
    n_warn = sum(1 for f in findings if f[0] == "WARN")
    if n_fail:
        print(f"FAIL: {n_fail} class(es) missing required companions ({n_warn} warning(s)); contract: {a.contract}")
        return 1
    print(f"OK: {len(changed)} file(s) differ from {a.base} ({mb[:10]}); {len(findings)} trigger(s), {n_warn} warning(s), "
          "no required companion missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
