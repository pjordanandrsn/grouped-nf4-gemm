#!/usr/bin/env python3
"""Claims-register hygiene: the bookkeeping fields of docs/claims.json that no
other check reads. Standard library only; no network.

Every other checker keys on a claim's ``status`` (check_capabilities,
check_readme_claims, the site's claim-ref checks) and none of them opens an
``evidence`` path, so a ``measured`` claim could cite a receipt that is not in
the tree, a ``superseded`` row could name no successor, and a note could say
"pending" on a row presented as current -- all green. This check reads the
fields themselves (docs/claims-schema.md defines them):

  * ``evidence[]``: every entry resolves at HEAD --
      - a bare repository path (file or directory) that exists;
      - ``{"path": P, "section": S}``: P exists and carries a Markdown heading
        whose text starts with S (``"0.24.0"`` matches ``## 0.24.0 — …``);
      - ``{"glob": G}``: at least one file matches G;
      - ``{"repository": R, "path": P}``: R is a related project
        (docs/capabilities.json ``related_projects`` or the system manifest);
        P is checked when ``--sibling DIR`` names a checkout of R, else listed
        as SKIP;
      - an ``https://`` URL under this repository's own GitHub URL (the issue
        tracker form) -- checked syntactically, never fetched.
    Annotated strings (``"receipts-m3/"``, ``"kernel/RESULTS-*.md (0.7.0)"``,
    ``"CHANGELOG.md 0.24.0"``) are exactly what this refuses: use the forms.
  * ``measured`` / ``confirmed`` / ``verified`` rows carry a non-empty
    ``evidence``; ``measured-private`` rows a non-empty ``evidence_private``
    (paths outside the repository; listed, never resolved).
  * ``measured_on`` is required on measured / measured-private / confirmed /
    verified rows and is an ISO calendar date (``YYYY-MM-DD``); when present
    on any other row it is ISO too.
  * ``superseded`` rows name a ``superseded_by`` that exists, is ACTIVE, and
    lists this row in its ``supersedes`` (the manifest's vocabulary: "replaced
    by a later claim that names it"); every ``supersedes`` entry exists and is
    ``superseded``.
  * ``retired`` rows carry a non-empty ``retired_reason``.
  * ``quoted_in[]`` entries resolve: ``path``, ``path#L<n>`` / ``path:<n>``
    (a line the file has), or ``path#<heading>`` (a heading prefix, as above).
    A location, not a containment check: the file need not spell the id.
  * ``notes`` of an ACTIVE row (measured / measured-private / confirmed /
    verified) carry no placeholder word -- ``pending``, ``TBD``, ``TODO`` --
    since an active row is presented as current; ``open`` rows may.

    python scripts/check_claims_register.py                      # CI gate
    python scripts/check_claims_register.py --sibling ../experts4bit-qlora

Exit 0 when clean, 1 on findings, 2 when the check itself cannot run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import (  # noqa: E402
    ACTIVE_STATUSES, ContractError, load_claims, load_pyproject, read_text, self_slug,
)

CLAIMS = "docs/claims.json"
CAPABILITIES = "docs/capabilities.json"
MANIFEST = "docs/system-manifest.json"

#: Statuses whose evidence must be public and dated.
PUBLIC_RUN = frozenset({"measured", "confirmed", "verified"})
PRIVATE_RUN = frozenset({"measured-private"})
DATED = PUBLIC_RUN | PRIVATE_RUN

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLACEHOLDER = re.compile(r"\b(pending|TBD|TODO)\b", re.I)
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
_LINE_ANCHOR = re.compile(r"^(?:L(\d+)|(\d+))$")


# ------------------------------------------------------------------ helpers --

def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


def heading_found(text: str, section: str) -> bool:
    """A Markdown heading whose text starts with ``section`` (whitespace
    folded, case-insensitive): ``"0.24.0"`` matches ``## 0.24.0 — 2026-09-03``."""
    want = _norm(section)
    return any(_norm(m.group(1)).startswith(want) for m in _HEADING.finditer(text))


def related_repositories(root: Path) -> set[str]:
    """Names a cross-repository evidence entry may cite: the canonical
    packages of docs/capabilities.json ``related_projects`` and the packages
    of docs/system-manifest.json (both optional files)."""
    names: set[str] = set()
    cap = root / CAPABILITIES
    if cap.is_file():
        doc = json.loads(read_text(cap))
        for r in (doc.get("project") or {}).get("related_projects", []) or []:
            if r.get("canonical_package"):
                names.add(str(r["canonical_package"]))
    man = root / MANIFEST
    if man.is_file():
        doc = json.loads(read_text(man))
        for p in (doc.get("packages") or {}).values():
            if isinstance(p, dict) and p.get("package"):
                names.add(str(p["package"]))
    return names


def _is_clean_path(p: str) -> bool:
    """A bare repository path: relative, no annotation, no glob characters."""
    return bool(p) and not p.startswith(("/", "~")) and ".." not in p.split("/") \
        and not re.search(r"[\s()*?\[\]]", p)


def resolve_evidence(entry, root: Path, self_url: str | None, related: set[str],
                     sibling: Path | None) -> tuple[str | None, str | None]:
    """``(finding, skip)`` for one evidence entry; both None when it resolves."""
    if isinstance(entry, str):
        if entry.startswith("https://"):
            if self_url and entry.startswith(self_url.rstrip("/") + "/"):
                return None, None
            return f"URL evidence must be under this repository's GitHub URL ({self_url}): {entry!r}", None
        if not _is_clean_path(entry):
            return (f"evidence {entry!r} is not a bare repository path -- use {{\"path\", \"section\"}}, "
                    "{\"glob\"} or {\"repository\", \"path\"} (docs/claims-schema.md)"), None
        if not (root / entry).exists():
            return f"evidence path does not exist at HEAD: {entry!r}", None
        return None, None
    if not isinstance(entry, dict):
        return f"evidence entry is neither a string nor an object: {entry!r}", None
    keys = set(entry) - {"note"}
    if keys == {"path", "section"}:
        p, s = str(entry["path"]), str(entry["section"])
        if not _is_clean_path(p) or not (root / p).is_file():
            return f"evidence path does not exist at HEAD: {p!r}", None
        if not heading_found(read_text(root / p), s):
            return f"evidence {p!r} has no heading starting with {s!r}", None
        return None, None
    if keys == {"glob"}:
        g = str(entry["glob"])
        if g.startswith("/") or ".." in g.split("/"):
            return f"evidence glob must be relative to the repository: {g!r}", None
        if not any(root.glob(g)):
            return f"evidence glob matches nothing at HEAD: {g!r}", None
        return None, None
    if keys == {"repository", "path"}:
        r, p = str(entry["repository"]), str(entry["path"])
        if r not in related:
            return f"evidence repository {r!r} is not a related project ({sorted(related)})", None
        if not _is_clean_path(p):
            return f"cross-repository evidence path is not a bare path: {p!r}", None
        if sibling is None:
            return None, f"{r} {p}: not resolved (no --sibling checkout given)"
        if not (sibling / p).exists():
            return f"cross-repository evidence missing in --sibling {sibling}: {p!r}", None
        return None, None
    return (f"evidence object with keys {sorted(entry)} is not a known form "
            "(path+section, glob, repository+path; optional note)"), None


def resolve_quoted_in(entry, root: Path) -> str | None:
    """A finding for one ``quoted_in`` entry, or None when it resolves."""
    if not isinstance(entry, str) or not entry.strip():
        return f"quoted_in entry is not a non-empty string: {entry!r}"
    path, anchor = entry, None
    if "#" in entry:
        path, anchor = entry.split("#", 1)
    elif re.search(r":\d+$", entry):
        path, anchor = entry.rsplit(":", 1)
    if not _is_clean_path(path) or not (root / path).is_file():
        return f"quoted_in path does not exist at HEAD: {entry!r}"
    if anchor is None:
        return None
    text = read_text(root / path)
    m = _LINE_ANCHOR.match(anchor.strip())
    if m:
        n = int(m.group(1) or m.group(2))
        total = text.count("\n") + (0 if text.endswith("\n") else 1)
        if n < 1 or n > total:
            return f"quoted_in {entry!r}: line {n} is outside the file ({total} lines)"
        return None
    if not heading_found(text, anchor):
        return f"quoted_in {entry!r}: no heading starting with {anchor!r} in {path}"
    return None


def iso_date(value) -> bool:
    if not isinstance(value, str) or not _ISO.match(value):
        return False
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------- check --

def check_register(root: Path, sibling: Path | None = None) -> tuple[list[str], list[str], int]:
    """``(findings, skips, n_claims)`` for the register at ``root``."""
    claims, _vocab = load_claims(root, CLAIMS)
    by_id = {c["id"]: c for c in claims}
    py = load_pyproject(root)
    slug = self_slug(root, py)
    self_url = f"https://github.com/{slug}" if slug else None
    related = related_repositories(root)
    findings: list[str] = []
    skips: list[str] = []

    def fail(cid: str, msg: str) -> None:
        findings.append(f"{cid}: {msg}")

    for c in claims:
        cid, st = c["id"], c["status"]
        ev = c.get("evidence")
        if ev is None:
            ev = []
        if not isinstance(ev, list):
            fail(cid, "evidence is not a list")
            ev = []
        for e in ev:
            f, s = resolve_evidence(e, root, self_url, related, sibling)
            if f:
                fail(cid, f)
            if s:
                skips.append(f"{cid}: {s}")
        evp = c.get("evidence_private") or []
        if not isinstance(evp, list) or any(not isinstance(x, str) or not x.strip() for x in evp):
            fail(cid, "evidence_private must be a list of non-empty strings")
        if st in PUBLIC_RUN and not ev:
            fail(cid, f"status {st!r} needs a public receipt: evidence is empty")
        if st in PRIVATE_RUN and not evp:
            fail(cid, "status 'measured-private' needs evidence_private (where the receipt lives)")
        if st in DATED:
            if "measured_on" not in c:
                fail(cid, f"status {st!r} needs measured_on (ISO date of the run)")
            elif not iso_date(c["measured_on"]):
                fail(cid, f"measured_on {c['measured_on']!r} is not an ISO calendar date (YYYY-MM-DD)")
        elif "measured_on" in c and not iso_date(c["measured_on"]):
            fail(cid, f"measured_on {c['measured_on']!r} is not an ISO calendar date (YYYY-MM-DD)")
        if st == "superseded":
            succ = c.get("superseded_by")
            if not succ:
                fail(cid, "status 'superseded' needs superseded_by (the claim to quote instead)")
            elif succ not in by_id:
                fail(cid, f"superseded_by {succ!r} is not in {CLAIMS}")
            elif by_id[succ]["status"] not in ACTIVE_STATUSES:
                fail(cid, f"superseded_by {succ!r} has status {by_id[succ]['status']!r}, not an active claim")
            elif cid not in (by_id[succ].get("supersedes") or []):
                fail(cid, f"superseded_by {succ!r} does not list {cid!r} in its supersedes")
        elif c.get("superseded_by"):
            fail(cid, f"superseded_by is set but status is {st!r}, not 'superseded'")
        for old in c.get("supersedes") or []:
            if old not in by_id:
                fail(cid, f"supersedes {old!r} is not in {CLAIMS}")
            elif by_id[old]["status"] != "superseded":
                fail(cid, f"supersedes {old!r} whose status is {by_id[old]['status']!r}, not 'superseded'")
        if st == "retired":
            rr = c.get("retired_reason")
            if not isinstance(rr, str) or not rr.strip():
                fail(cid, "status 'retired' needs retired_reason (one sentence, with the measurement that retired it)")
        for q in c.get("quoted_in") or []:
            f = resolve_quoted_in(q, root)
            if f:
                fail(cid, f)
        if st in ACTIVE_STATUSES:
            notes = c.get("notes")
            if isinstance(notes, str):
                m = _PLACEHOLDER.search(notes)
                if m:
                    fail(cid, f"notes of an active ({st}) row carry the placeholder word {m.group(0)!r}; "
                              "an active row is presented as current -- state the fact, or make the row open")
    return findings, skips, len(claims)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--sibling", default=None, metavar="DIR",
                    help="local checkout of the related repository; resolves {repository, path} evidence")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sibling = Path(a.sibling).resolve() if a.sibling else None
    try:
        findings, skips, n = check_register(root, sibling)
    except (ContractError, OSError, ValueError) as e:
        print(f"FAIL: {e}")
        return 2
    for s in skips:
        print("SKIP:", s)
    for f in findings:
        print("FAIL:", f)
    if findings:
        print(f"FAIL: {len(findings)} finding(s) in {CLAIMS}; the field forms are in docs/claims-schema.md")
        return 1
    print(f"OK: {CLAIMS}: {n} claims -- every evidence entry resolves at HEAD"
          f"{' (' + str(len(skips)) + ' cross-repository path(s) not resolved without --sibling)' if skips else ''}, "
          "measured rows are dated, superseded/retired rows carry their bookkeeping, quoted_in resolves, "
          "no placeholder word on an active row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
