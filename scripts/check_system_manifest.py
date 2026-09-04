#!/usr/bin/env python3
"""Validate docs/system-manifest.json against this repository (the kernel
package of the two-repository system) and, on request, against the sibling.

Standard library only; no network. The manifest is byte-identical in
experts4bit-qlora and grouped-nf4-gemm; every value pyproject.toml also
carries (names, versions, floors) is validated here, never trusted from the
manifest. Checks:

  * the file parses and carries system / packages / compatibility /
    evidence_vocabulary / capability_ownership / invariants / router;
  * packages.kernels: package equals [project].name, repository equals the
    Source URL, and every import_name is shipped ([tool.setuptools]
    py-modules or packages);
  * the kernel-first invariant: every compatibility record for this kernel
    names a `floor` that is <= the pyproject version (a floor never names an
    unreleased kernel version) and, when v* tags are available locally, <=
    the latest tag; the record that is current for the consumer is reported;
  * evidence_vocabulary keys are a superset of docs/claims.json
    status_vocabulary keys;
  * capability_ownership.kernels equals the set of ids in docs/capabilities.json;
  * every invariant has id / statement / checked_by, ids unique;
  * the router carries its five entries.

  --sibling PATH (a local checkout of the consumer; never used in CI):
  PATH/docs/system-manifest.json is byte-identical to this one, the
  sibling's [project].name is packages.runtime.package, its version selects
  exactly one compatibility record, and its `fast` extra pins this package
  at that record's floor.

Prints OK / FAIL / SKIP lines; exit 1 on any FAIL, 2 when the check itself
cannot run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import (  # noqa: E402
    load_claims, load_pyproject, module_shipped, pep503_name, read_text, requirement_extras, requirement_name,
)

REQUIRED_TOP = ("system", "packages", "compatibility", "evidence_vocabulary", "capability_ownership",
                "invariants", "router", "authority")
RECORD_KEYS = ("consumer", "consumer_versions", "kernel", "floor", "extra", "since", "why")
ROUTER_ENTRIES = 5

_VERSION = re.compile(r"v?(\d+(?:\.\d+)*)")
_CLAUSE = re.compile(r"\s*(>=|<=|==|!=|~=|>|<)?\s*v?(\d+(?:\.\d+)*)(\.\*|\.x)?\s*$")


# ------------------------------------------------------------------ versions --

def release_tuple(v: str) -> tuple[int, ...]:
    """The release segment of a version as a tuple: ``0.30.0`` -> (0, 30, 0);
    ``2.8.0.dev0`` -> (2, 8, 0). Pre/post/dev suffixes are ignored on purpose:
    a floor here is a release floor."""
    m = _VERSION.match(str(v).strip())
    if not m:
        raise ValueError(f"not a version: {v!r}")
    return tuple(int(x) for x in m.group(1).split("."))


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a2, b2 = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a2 > b2) - (a2 < b2)


def floor_version(floor: str) -> tuple[int, ...]:
    """``">=0.30.0"`` -> (0, 30, 0). Only the >= form is a floor."""
    m = _CLAUSE.match(floor)
    if not m or m.group(1) != ">=" or m.group(3):
        raise ValueError(f"floor must be a '>=X.Y.Z' specifier, got {floor!r}")
    return release_tuple(m.group(2))


def range_contains(rng: str, version: tuple[int, ...]) -> bool:
    """Does a consumer_versions range (``">=0.35.0"``, ``"0.34.x"``,
    ``">=0.30,<0.34"``, ``"==0.33.*"``) contain ``version``?"""
    for clause in rng.split(","):
        m = _CLAUSE.match(clause)
        if not m:
            raise ValueError(f"cannot parse version range clause {clause!r} in {rng!r}")
        op, ver, wild = m.group(1), release_tuple(m.group(2)), m.group(3)
        if wild or (op in (None, "==") and len(ver) < 3 and not op):
            if version[:len(ver)] != ver:
                return False
            continue
        c = _cmp(version, ver)
        ok = {None: c == 0, "==": c == 0, "!=": c != 0, ">=": c >= 0, ">": c > 0, "<=": c <= 0, "<": c < 0,
              "~=": c >= 0 and version[:len(ver) - 1] == ver[:-1]}[op]
        if not ok:
            return False
    return True


def range_lower_bound(rng: str) -> tuple[int, ...]:
    """The lowest version a range admits, for ordering records when no
    consumer version is at hand."""
    lows = []
    for clause in rng.split(","):
        m = _CLAUSE.match(clause)
        if m and m.group(1) in (None, ">=", ">", "==", "~="):
            lows.append(release_tuple(m.group(2)))
    return max(lows) if lows else (0,)


def kernel_records(manifest: dict, kernel_pkg: str) -> list[dict]:
    return [r for r in manifest.get("compatibility", []) if pep503_name(str(r.get("kernel", ""))) == pep503_name(kernel_pkg)]


def current_record(manifest: dict, kernel_pkg: str, consumer_version: str | None = None) -> dict | None:
    """The compatibility record that is current for the consumer: with a
    consumer version, the one whose consumer_versions contains it (None when
    zero or several do); without one, the record whose range starts highest."""
    recs = kernel_records(manifest, kernel_pkg)
    if not recs:
        return None
    if consumer_version is not None:
        hits = [r for r in recs if range_contains(str(r["consumer_versions"]), release_tuple(consumer_version))]
        return hits[0] if len(hits) == 1 else None
    return max(recs, key=lambda r: range_lower_bound(str(r["consumer_versions"])))


def latest_tag(root: Path) -> tuple[int, ...] | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "tag", "-l", "v*"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tags = []
    for t in out.split():
        try:
            tags.append(release_tuple(t))
        except ValueError:
            continue
    return max(tags) if tags else None


def _norm_url(u: str) -> str:
    return str(u).strip().rstrip("/").removesuffix(".git").lower()


# ---------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--manifest", default="docs/system-manifest.json")
    ap.add_argument("--sibling", default=None, metavar="PATH",
                    help="local checkout of the consumer repository (byte-identity + fast-extra floor); never in CI")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    ok: list[str] = []
    fail: list[str] = []
    skip: list[str] = []

    raw = (root / a.manifest).read_bytes()
    try:
        m = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        print(f"FAIL: {a.manifest}: not valid JSON: {e}")
        return 1
    ok.append(f"{a.manifest} parses ({len(raw)} bytes)")
    missing = [k for k in REQUIRED_TOP if k not in m]
    if missing:
        print(f"FAIL: {a.manifest}: missing top-level keys {missing}")
        return 1

    py = load_pyproject(root)
    name = str(py.get("name", ""))
    version = str(py.get("version", ""))
    kernels = m["packages"].get("kernels", {})
    runtime = m["packages"].get("runtime", {})

    # -- names, repository, import names ------------------------------------
    if pep503_name(str(kernels.get("package", ""))) != pep503_name(name):
        fail.append(f"packages.kernels.package {kernels.get('package')!r} != pyproject name {name!r}")
    else:
        ok.append(f"packages.kernels.package == pyproject name ({name})")
    src = (py.get("urls") or {}).get("Source", "")
    if _norm_url(kernels.get("repository", "")) != _norm_url(src):
        fail.append(f"packages.kernels.repository {kernels.get('repository')!r} != pyproject urls.Source {src!r}")
    else:
        ok.append("packages.kernels.repository == pyproject urls.Source")
    unshipped = [i for i in kernels.get("import_names", []) if not module_shipped(i, py)]
    if unshipped:
        fail.append(f"packages.kernels.import_names not shipped by pyproject (py-modules / packages): {unshipped}")
    else:
        ok.append(f"packages.kernels.import_names ({len(kernels.get('import_names', []))}) are all shipped")
    direction = f"{runtime.get('package')} -> {kernels.get('package')}"
    if direction not in m["system"].get("dependency_direction", []):
        fail.append(f"system.dependency_direction lacks {direction!r}")
    else:
        ok.append(f"dependency direction {direction}")
    consumer_url = (py.get("urls") or {}).get("Consumer: experts4bit-qlora")
    if consumer_url and runtime.get("pypi") and _norm_url(consumer_url) != _norm_url(runtime["pypi"]):
        fail.append(f"pyproject's consumer URL {consumer_url!r} != packages.runtime.pypi {runtime['pypi']!r}")

    # -- compatibility: kernel-first ------------------------------------------
    recs = kernel_records(m, name)
    if not recs:
        fail.append(f"compatibility has no record whose kernel is {name!r}")
    here = release_tuple(version)
    tag = latest_tag(root)
    for r in recs:
        lacking = [k for k in RECORD_KEYS if k not in r]
        if lacking:
            fail.append(f"compatibility record for consumer {r.get('consumer_versions')!r} lacks {lacking}")
            continue
        try:
            fv = floor_version(str(r["floor"]))
        except ValueError as e:
            fail.append(f"compatibility record {r['consumer_versions']!r}: {e}")
            continue
        if pep503_name(str(r["consumer"])) != pep503_name(str(runtime.get("package", ""))):
            fail.append(f"compatibility record {r['consumer_versions']!r}: consumer {r['consumer']!r} is not packages.runtime.package")
        if _cmp(fv, here) > 0:
            fail.append(f"kernel-first: record {r['consumer_versions']!r} floors on {r['floor']} but pyproject is {version} "
                        "(a floor never names an unreleased kernel version)")
        else:
            ok.append(f"kernel-first vs pyproject: record {r['consumer_versions']!r} floor {r['floor']} <= {version}")
        if tag is not None:
            if _cmp(fv, tag) > 0:
                fail.append(f"kernel-first: record {r['consumer_versions']!r} floors on {r['floor']} but the latest local tag is "
                            f"v{'.'.join(map(str, tag))} (no such release exists yet)")
            else:
                ok.append(f"kernel-first vs tags: record {r['consumer_versions']!r} floor {r['floor']} <= v{'.'.join(map(str, tag))}")
    if tag is None:
        skip.append("no v* tags available locally; the floor-vs-released-tag comparison did not run (pyproject comparison did)")
    cur = current_record(m, name)
    if cur is not None:
        ok.append(f"current record (highest consumer range without a sibling): consumer {cur['consumer_versions']} -> "
                  f"{name}{cur['floor']} [{cur['extra']}] since {cur['since']}")

    # -- evidence vocabulary ----------------------------------------------------
    try:
        _claims, vocab = load_claims(root, "docs/claims.json")
    except Exception as e:  # noqa: BLE001 -- any malformed register is a finding, reported not raised
        fail.append(f"docs/claims.json: {e}")
        vocab = {}
    ev = set(m["evidence_vocabulary"])
    lacking = sorted(set(vocab) - ev)
    if lacking:
        fail.append(f"evidence_vocabulary lacks claims.json statuses {lacking}")
    else:
        ok.append(f"evidence_vocabulary ({len(ev)}) covers claims.json status_vocabulary ({len(vocab)})")

    # -- capability ownership ---------------------------------------------------
    caps = json.loads(read_text(root / "docs" / "capabilities.json"))
    have = {c["id"] for c in caps["capabilities"]}
    want = set(m["capability_ownership"].get("kernels", []))
    if have != want:
        fail.append(f"capability_ownership.kernels != docs/capabilities.json ids: manifest-only {sorted(want - have)}, "
                    f"capabilities-only {sorted(have - want)}")
    else:
        ok.append(f"capability_ownership.kernels == docs/capabilities.json ids ({len(have)})")
    overlap = want & set(m["capability_ownership"].get("runtime", []))
    if overlap:
        fail.append(f"capability ids owned by both packages: {sorted(overlap)}")

    # -- invariants, router -------------------------------------------------------
    ids = []
    for inv in m["invariants"]:
        lacking = [k for k in ("id", "statement", "checked_by") if not inv.get(k)]
        if lacking:
            fail.append(f"invariant {inv.get('id')!r} lacks {lacking}")
        ids.append(inv.get("id"))
    if len(set(ids)) != len(ids):
        fail.append("invariant ids are not unique")
    if not fail or all("invariant" not in f for f in fail):
        ok.append(f"{len(ids)} invariants carry id / statement / checked_by")
    if len(m["router"]) != ROUTER_ENTRIES:
        fail.append(f"router has {len(m['router'])} entries, not {ROUTER_ENTRIES}")
    else:
        ok.append("router carries five entries")

    # -- sibling ----------------------------------------------------------------
    if a.sibling:
        sib = Path(a.sibling).resolve()
        sm = sib / "docs" / "system-manifest.json"
        if not sm.is_file():
            fail.append(f"sibling manifest missing: {sm}")
        elif sm.read_bytes() != raw:
            fail.append(f"sibling manifest differs byte-for-byte: {sm}")
        else:
            ok.append("sibling manifest is byte-identical")
        try:
            spy = load_pyproject(sib)
        except OSError as e:
            fail.append(f"sibling pyproject: {e}")
            spy = None
        if spy is not None:
            sname, sver = str(spy.get("name", "")), str(spy.get("version", ""))
            if pep503_name(sname) != pep503_name(str(runtime.get("package", ""))):
                fail.append(f"sibling pyproject name {sname!r} != packages.runtime.package {runtime.get('package')!r}")
            rec = current_record(m, name, sver)
            if rec is None:
                fail.append(f"no single compatibility record contains the sibling version {sver}")
            else:
                extra = str(rec["extra"])
                reqs = (spy.get("optional-dependencies") or {}).get(extra, [])
                pins = [r for r in reqs if requirement_name(r) == pep503_name(name)]
                spec = ""
                if pins:
                    spec = re.sub(r"^[A-Za-z0-9._-]+(\[[^\]]*\])?", "", pins[0]).split(";")[0].replace(" ", "")
                if not pins:
                    fail.append(f"sibling [{extra}] extra does not pin {name}")
                elif spec != str(rec["floor"]).replace(" ", ""):
                    fail.append(f"sibling [{extra}] pins {name}{spec} but the current manifest record says {rec['floor']} "
                                f"(consumer {sver} in {rec['consumer_versions']})")
                else:
                    ok.append(f"sibling {sname} {sver} [{extra}] pins {name}{spec} == current record floor")
                if requirement_extras(pins[0]) if pins else False:
                    skip.append(f"sibling pin carries extras {requirement_extras(pins[0])}; not validated")
            sc = sib / "docs" / "claims.json"
            if sc.is_file():
                try:
                    _c, svocab = load_claims(sib, "docs/claims.json")
                    lacking = sorted(set(svocab) - ev)
                    if lacking:
                        fail.append(f"evidence_vocabulary lacks the sibling's claims.json statuses {lacking}")
                    else:
                        ok.append("evidence_vocabulary covers the sibling's claims.json status_vocabulary")
                except Exception as e:  # noqa: BLE001
                    skip.append(f"sibling docs/claims.json not checked: {e}")

    for line in ok:
        print("OK:", line)
    for line in skip:
        print("SKIP:", line)
    for line in fail:
        print("FAIL:", line)
    if fail:
        print(f"FAIL: {len(fail)} finding(s) in {a.manifest}")
        return 1
    print(f"OK: {a.manifest} agrees with pyproject.toml, docs/claims.json and docs/capabilities.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
