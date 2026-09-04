#!/usr/bin/env python3
"""Dependency-version statements in the CURRENT documents, checked against
their canonical sources. Standard library only; no network.

This repository is the dependency, not the consumer, so the drift it can
carry is (a) a statement of this package's own version -- "version 0.29.0",
"grouped-nf4-gemm 0.29.0", "latest ... 0.29.0", a self-repository link pinned
to a release tag -- that no longer equals pyproject.toml, and (b) a statement
of the consumer floor -- "grouped-nf4-gemm>=X" beside the `fast` extra, or
"experts4bit-qlora>=Y" -- that differs from the compatibility record that is
current in docs/system-manifest.json. Both FAIL. The other hand-copied floors
the documents carry are validated the same way rather than removed: torch and
triton floors against pyproject dependencies, "Python 3.11 ... CI" against
.github/workflows/ci.yml, "pyproject says >=3.9" against requires-python, and
the README licence word against [project].license.

Scope: README.md, AGENTS.md, llms.txt, docs/SOLUTIONS.md, docs/INDEX.md,
docs/solutions/*.md, docs/capabilities.json and the docs/STATUS.md header
(up to its first `---` rule). Historical lines are exempt: CHANGELOG.md,
receipts and anchored documents are never scanned, and a line carrying
"was", "were", "raised from", "raised to", "until" or "no longer" is a record
of a past state, not a current statement. The pyproject comment ladder is not
a document.

Prints one FAIL per finding as file:line -> statement -> canonical value;
exit 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_system_manifest import current_record, floor_version, range_lower_bound, release_tuple  # noqa: E402
from discovery_common import load_pyproject, pep503_name, read_text, requirement_name, self_slug  # noqa: E402

CURRENT_DOCS = ["README.md", "AGENTS.md", "llms.txt", "docs/SOLUTIONS.md", "docs/INDEX.md", "docs/capabilities.json"]
STATUS_HEADER = "docs/STATUS.md"
SOLUTIONS_GLOB = "docs/solutions/*.md"

HISTORICAL = re.compile(r"\b(was|were|raised (?:from|to)|until|no longer)\b", re.I)
OTHER_PACKAGE = re.compile(r"(bitsandbytes|torch|triton|python|transformers|numpy|pip|cff|unsloth|vllm)", re.I)

VERSION_WORD = re.compile(r"\bversion\s+v?(\d+\.\d+\.\d+)\b", re.I)
LATEST = re.compile(r"\blatest\b[^.\n]{0,80}?\bv?(\d+\.\d+\.\d+)\b|\bv?(\d+\.\d+\.\d+)\b[^.\n]{0,80}?\blatest\b", re.I)
SELF_NAME_VERSION = re.compile(r"grouped-nf4-gemm\s+v?(\d+\.\d+\.\d+)\b")
SELF_PIN = re.compile(r"grouped-nf4-gemm\s*(==|>=|~=)\s*v?(\d+(?:\.\d+)+)")
CONSUMER_PIN = re.compile(r"experts4bit-qlora\s*(>=|==)\s*v?(\d+(?:\.\d+)+)")
BARE_FLOOR = re.compile(r"(?<![\w.-])>=\s*v?(\d+\.\d+\.\d+)\b")
TORCH_FLOOR = re.compile(r"\btorch\s*(?:>=|≥)\s*v?(\d+(?:\.\d+)*)", re.I)
TRITON_FLOOR = re.compile(r"\btriton\s*(?:>=|≥)\s*v?(\d+(?:\.\d+)*)", re.I)
PYTHON_CI = re.compile(r"\bPython\s+(\d+\.\d+)\b(?![.\d])")
PYTHON_PYPROJECT = re.compile(r"pyproject says\s*>=\s*(\d+\.\d+)")
FAST_CONTEXT = re.compile(r"\[fast\]|`fast`|fast extra|consumer floor|floors? on|floor", re.I)


def _status_header(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip() == "---":
            break
        out.append((i, line))
    return out


def _lines(rel: str, text: str) -> list[tuple[int, str]]:
    if rel == STATUS_HEADER:
        return _status_header(text)
    return list(enumerate(text.splitlines(), 1))


def _dep_floor(py: dict, name: str) -> tuple[int, ...] | None:
    for req in py.get("dependencies") or []:
        if requirement_name(req) == name:
            m = re.search(r">=\s*v?(\d+(?:\.\d+)*)", req)
            return release_tuple(m.group(1)) if m else None
    return None


def _ci_python_versions(root: Path) -> set[str]:
    ci = root / ".github" / "workflows" / "ci.yml"
    if not ci.is_file():
        return set()
    return set(re.findall(r'python-version:\s*"?(\d+\.\d+)"?', read_text(ci)))


def _truncate_eq(doc: tuple[int, ...], canon: tuple[int, ...]) -> bool:
    """A document floor is compared at its own precision: "2.8" matches a
    pyproject floor of 2.8.0.dev0; "2.9" does not."""
    return canon[:len(doc)] == doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default="docs/system-manifest.json")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    py = load_pyproject(root)
    name = pep503_name(str(py.get("name", "")))
    version = str(py.get("version", "")).strip()
    slug = self_slug(root, py)
    if slug is None:
        print("FAIL: self-repo slug unknown (no github.com URL in [project.urls] and no git origin)")
        return 2
    self_link = re.compile(rf"github\.com/{re.escape(slug)}/(?:blob|tree)/v(\d+(?:\.\d+)+)/")

    manifest = json.loads(read_text(root / a.manifest))
    rec = current_record(manifest, name)
    floor = floor_version(str(rec["floor"])) if rec else None
    consumer_low = range_lower_bound(str(rec["consumer_versions"])) if rec else None
    torch_floor = _dep_floor(py, "torch")
    triton_floor = _dep_floor(py, "triton")
    ci_py = _ci_python_versions(root)
    req_py = re.search(r">=\s*(\d+\.\d+)", str(py.get("requires-python", "")))
    lic = py.get("license")
    lic = lic.get("text") if isinstance(lic, dict) else lic

    docs = list(CURRENT_DOCS) + [STATUS_HEADER] + [p.relative_to(root).as_posix() for p in sorted(root.glob(SOLUTIONS_GLOB))]
    fails: list[str] = []
    n_statements = 0

    def check(rel: str, ln: int, what: str, stated: str, canonical: str, equal: bool) -> None:
        nonlocal n_statements
        n_statements += 1
        if not equal:
            fails.append(f"{rel}:{ln} -> {what} states {stated!r} but the canonical value is {canonical!r}")

    for rel in docs:
        p = root / rel
        if not p.is_file():
            fails.append(f"{rel}: missing")
            continue
        for ln, line in _lines(rel, read_text(p)):
            if HISTORICAL.search(line):
                continue
            for m in self_link.finditer(line):
                check(rel, ln, "self-repository link pinned to tag", "v" + m.group(1), "v" + version, m.group(1) == version)
            for m in VERSION_WORD.finditer(line):
                if OTHER_PACKAGE.search(line[max(0, m.start() - 40):m.start()]):
                    continue
                check(rel, ln, "own version", m.group(1), version, m.group(1) == version)
            for m in LATEST.finditer(line):
                v = m.group(1) or m.group(2)
                if OTHER_PACKAGE.search(line[max(0, m.start() - 40):m.end()]):
                    continue
                check(rel, ln, "'latest' version", v, version, v == version)
            for m in SELF_NAME_VERSION.finditer(line):
                check(rel, ln, "own version beside the package name", m.group(1), version, m.group(1) == version)
            for m in SELF_PIN.finditer(line):
                op, v = m.group(1), m.group(2)
                if op == ">=":
                    if floor is None:
                        fails.append(f"{rel}:{ln} -> consumer floor {name}>={v} stated but the manifest has no current record")
                    else:
                        check(rel, ln, "consumer floor for this package", ">=" + v, str(rec["floor"]),
                              release_tuple(v) == floor)
                else:
                    check(rel, ln, f"own version pin {op}", v, version, release_tuple(v) == release_tuple(version))
            for m in CONSUMER_PIN.finditer(line):
                if consumer_low is None:
                    continue
                check(rel, ln, "consumer version of the current record", m.group(1) + m.group(2), str(rec["consumer_versions"]),
                      release_tuple(m.group(2)) == consumer_low)
            if FAST_CONTEXT.search(line) and not SELF_PIN.search(line) and not CONSUMER_PIN.search(line):
                for m in BARE_FLOOR.finditer(line):
                    if OTHER_PACKAGE.search(line[max(0, m.start() - 24):m.start()]):
                        continue
                    if floor is not None:
                        check(rel, ln, "bare consumer floor beside 'fast'", ">=" + m.group(1), str(rec["floor"]),
                              release_tuple(m.group(1)) == floor)
            if torch_floor is not None:
                for m in TORCH_FLOOR.finditer(line):
                    check(rel, ln, "torch floor", m.group(1), ".".join(map(str, torch_floor)),
                          _truncate_eq(release_tuple(m.group(1)), torch_floor))
            if triton_floor is not None:
                for m in TRITON_FLOOR.finditer(line):
                    check(rel, ln, "triton floor", m.group(1), ".".join(map(str, triton_floor)),
                          _truncate_eq(release_tuple(m.group(1)), triton_floor))
            if ci_py and re.search(r"\bCI\b", line):
                for m in PYTHON_CI.finditer(line):
                    check(rel, ln, "Python version CI tests", m.group(1), "/".join(sorted(ci_py)), m.group(1) in ci_py)
            if req_py:
                for m in PYTHON_PYPROJECT.finditer(line):
                    check(rel, ln, "requires-python floor", m.group(1), req_py.group(1), m.group(1) == req_py.group(1))
    if lic:
        readme = read_text(root / "README.md")
        sec = re.search(r"^## License.*?(?=^## |\Z)", readme, re.M | re.S)
        n_statements += 1
        if sec and str(lic) not in sec.group(0):
            fails.append(f"README.md -> the License section does not name pyproject's license {lic!r}")

    for f in fails:
        print("FAIL:", f)
    if fails:
        print(f"FAIL: {len(fails)} stale version statement(s); pyproject is {version}"
              + (f", current consumer record {rec['consumer_versions']} -> {name}{rec['floor']}" if rec else ""))
        return 1
    print(f"OK: {n_statements} version statements across {len(docs)} current documents agree with pyproject {version}"
          + (f" and the manifest's current record ({rec['consumer_versions']} -> {name}{rec['floor']})" if rec else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
