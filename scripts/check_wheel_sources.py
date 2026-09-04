#!/usr/bin/env python3
"""Every module inside a built wheel must be byte-identical to its source.

A repository that ever tracked ``build/lib/*.py`` can ship those copies
instead of ``kernel/``: setuptools' ``build_py`` copies a source only when
it is strictly newer than the file already sitting in ``build/lib``, and a
fresh clone gives both the same second. On 2026-09-04 that put a 0.2.5-era
``nvme_reader.py`` into a wheel built from a pinned commit while its
sibling ``nvme_residency.py`` was current. ``build/`` is untracked now;
this check makes the class impossible to miss: it opens the wheel, maps
each shipped ``.py`` back to ``kernel/`` (or ``gnf4_native/``) through
pyproject's ``py-modules`` / ``package-dir``, and compares sha256.

    python scripts/check_wheel_sources.py dist/*.whl
"""
from __future__ import annotations

import hashlib
import sys
import tomllib
import zipfile
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_wheel_sources.py WHEEL [WHEEL ...]")
        return 2
    root = Path(__file__).resolve().parent.parent
    py = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    st = py.get("tool", {}).get("setuptools", {})
    pkg_dir = st.get("package-dir", {})
    modules = set(st.get("py-modules", []))
    src_root = root / pkg_dir.get("", ".")
    rc = 0
    for wheel in argv[1:]:
        bad = 0
        checked = 0
        with zipfile.ZipFile(wheel) as z:
            for name in z.namelist():
                if not name.endswith(".py") or ".dist-info/" in name:
                    continue
                top = name.split("/", 1)[0]
                if "/" not in name:
                    if name[:-3] not in modules:
                        print(f"FAIL {wheel}: {name} is not in py-modules -- who packaged it?")
                        bad += 1
                        continue
                    src = src_root / name
                else:
                    base = pkg_dir.get(top)
                    src = (root / base / name.split("/", 1)[1]) if base else (src_root / name)
                if not src.exists():
                    print(f"FAIL {wheel}: {name} has no source at {src}")
                    bad += 1
                    continue
                a = hashlib.sha256(z.read(name)).hexdigest()
                b = hashlib.sha256(src.read_bytes()).hexdigest()
                checked += 1
                if a != b:
                    print(f"FAIL {wheel}: {name} differs from {src.relative_to(root)} (wheel {a[:12]}, source {b[:12]}) -- a stale build/ copy shipped")
                    bad += 1
        if bad:
            rc = 1
        else:
            print(f"OK: {wheel}: {checked} modules byte-identical to their sources")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
