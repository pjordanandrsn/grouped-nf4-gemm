#!/usr/bin/env python3
"""The prose quotes the register. Two mechanical facts about the documents that
state numbers -- README.md, docs/STATUS.md and docs/solutions/*.md -- checked
offline (standard library only, no network). A port of experts4bit-qlora's
scripts/check_readme_claims.py to this repository's tables (a ``tier`` column,
the result before the id) and its wider document set; this repository has no
generated release block (its tag pins are checked by check_dependency_floor.py),
so that part of the original is not here.

1. **Every headline number is the register's current value.** In README.md
   and docs/STATUS.md, every markdown table whose header has a ``status`` or
   ``tier`` column is a results table: each row names its claim ids in
   backticks (``gnf4.…``; a ``*`` makes it a glob over docs/claims.json).
   Every number in the row's result cell must be a number of one of those
   claims (its ``value``, ``unit`` or ``claim`` sentence -- never its
   ``notes``, which carry the unlicensed arms), read at the document's own
   precision (``155`` for 154.9, ``1.70`` for 1.7); every named claim's
   ``value`` must appear in the row; a ``superseded`` or ``retired`` id fails
   outright (the message names what to quote instead); and the tier cell
   must name the weakest status among the row's claims, so a
   ``measured-private`` receipt is never presented as ``measured``.

2. **Every backticked id exists, and an inactive one says so.** In all three
   document sets, every backticked ``gnf4.…`` id (globs allowed) is in
   docs/claims.json, and a ``superseded`` or ``retired`` id may only appear on
   a line that says ``superseded``, ``retired`` or ``historical``. Ids of the
   sibling register (``e4b.…``) are not resolved here.

    python scripts/check_readme_claims.py            # CI gate

Exit 0 when clean, 1 on findings, 2 when the check itself cannot run.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import ContractError, KNOWN_STATUSES, load_claims, read_text  # noqa: E402

CLAIMS = "docs/claims.json"
#: Results tables are read here; the id rules apply to every document.
TABLE_DOCS = ("README.md", "docs/STATUS.md")
ID_DOCS_GLOB = "docs/solutions/*.md"

#: Never a current number, whatever the row says.
INACTIVE = frozenset({"superseded", "retired"})
#: A line may cite an inactive id when it says so with one of these words.
INACTIVE_WORDS = ("superseded", "retired", "historical")
#: Weakest first: the tier cell must name the weakest one present in the row.
_WEAKNESS = ("open", "projected", "measured-private", "measured", "confirmed", "verified")
#: Header cells that mark the status column, the id column and the result column.
STATUS_HEADERS = ("status", "tier")
RESULT_HEADERS = ("result", "measured", "value")

_STATUS_WORD = re.compile(r"measured-private|measured|verified|confirmed|projected|superseded|retired|open")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def id_pattern(claims: dict[str, dict]) -> re.Pattern:
    """Backticked ids of THIS register: the prefixes before the first ``.`` of
    every claim id (``gnf4``), so a sibling id quoted in prose is left alone."""
    prefixes = sorted({k.split(".", 1)[0] for k in claims if "." in k})
    if not prefixes:
        raise ContractError(f"{CLAIMS}: no dotted claim ids to derive a prefix from")
    alt = "|".join(re.escape(p) for p in prefixes)
    return re.compile(rf"`((?:{alt})\.[A-Za-z0-9._+*-]+)`")


# ------------------------------------------------------------------ numbers --

def _normalise(text: str) -> str:
    """Typography the documents use, folded to ASCII: minus signs, en/em
    dashes, ``≈``/``~`` approximations, ``±``/``+-`` tolerances (``±4.2%`` is
    the unsigned 4.2 on both sides), ``×`` to a space (``6.40×`` is a number,
    not a name); links reduced to their text so a URL's digits are never read
    as a result; dates blanked."""
    text = _LINK.sub(r"\1", text)
    for a, b in (("+-", " "), ("±", " "), ("−", "-"), ("–", "-"), ("—", "-"), ("×", " "), ("≈", ""), ("~", ""),
                 ("→", " "), (" ", " ")):
        text = text.replace(a, b)
    return _DATE.sub(" ", text)


def _to_decimal(tok: str) -> Decimal | None:
    try:
        return Decimal(tok.replace(",", ""))
    except InvalidOperation:
        return None


def result_numbers(cell: str) -> list[Decimal]:
    """The numbers a result cell states, at the cell's own precision.

    Skipped, because they are names and not results: digits glued to a letter
    on either side (``30B``, ``fp8``, ``64k``, ``sm_86``), a hyphenated name
    (``Gemma-4``, ``round-1``), a key (``B=16``, ``E=256``), an issue
    (``#359``) and ``144/144``-style fractions' second half. A range
    ``1.52-1.81`` yields both ends; a leading ``-``/``+`` after whitespace is
    a sign.
    """
    text = _normalise(cell)
    out: list[Decimal] = []
    for m in _NUMBER.finditer(text):
        i, j = m.start(), m.end()
        before = text[i - 1] if i else " "
        before2 = text[i - 2] if i >= 2 else " "
        after = text[j] if j < len(text) else " "
        if after.isalnum() or after == "_" or (after == "." and j + 1 < len(text) and text[j + 1].isdigit()):
            continue
        if before.isalnum() or before in "._=#":
            continue
        if before == "/" and before2.isdigit():
            continue
        sign = ""
        if before in "+-":
            if before2.isalpha() or before2 == "_":
                continue                                     # Gemma-4, round-1: a name
            if not before2.isdigit():
                sign = before                                # -0.053, +37.1: a sign
            # else 1.52-1.81: a range, this is its unsigned upper end
        d = _to_decimal(sign + m.group(0))
        if d is not None:
            out.append(d)
    return out


def _signed_numbers(text: str) -> list[Decimal]:
    """Every number in ``text`` with its sign (a leading + or - not preceded by a digit)."""
    out: list[Decimal] = []
    for m in _NUMBER.finditer(text):
        i = m.start()
        before = text[i - 1] if i else " "
        before2 = text[i - 2] if i >= 2 else " "
        sign = before if before in "+-" and not before2.isdigit() else ""
        d = _to_decimal(sign + m.group(0))
        if d is not None:
            out.append(d)
    return out


def claim_numbers(claim: dict) -> list[Decimal]:
    """Every number the claim states in its ``value``, ``unit`` and ``claim``
    fields. ``notes`` are excluded on purpose: they carry the measured-but-
    unlicensed arms, which must never become a headline by matching here."""
    parts = []
    v = claim.get("value")
    if isinstance(v, bool):
        v = None
    if isinstance(v, (int, float)):
        parts.append(repr(v))
    elif isinstance(v, str):
        parts.append(v)
    for k in ("unit", "claim"):
        if isinstance(claim.get(k), str):
            parts.append(claim[k])
    text = _normalise(" ".join(parts))
    return _signed_numbers(text)


def value_numbers(claim: dict) -> list[Decimal]:
    """The numbers of the claim's ``value`` alone (a range string gives both ends)."""
    v = claim.get("value")
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [Decimal(repr(v))]
    return _signed_numbers(_normalise(str(v)))


def number_matches(n: Decimal, pool: list[Decimal]) -> bool:
    """``n`` equals a pool number at ``n``'s own precision: 155 for 154.9,
    1.70 for 1.7, 0.046 for 0.04645. Rounding only -- 100 never matches 98.3."""
    exp = n.as_tuple().exponent
    q = Decimal(1).scaleb(exp) if isinstance(exp, int) else None
    for p in pool:
        if p == n:
            return True
        if q is None:
            continue
        try:
            if p.quantize(q, rounding=ROUND_HALF_UP) == n or p.quantize(q, rounding=ROUND_HALF_EVEN) == n:
                return True
        except InvalidOperation:
            continue
    return False


# ------------------------------------------------------------------- tables --

def _split_row(line: str) -> list[str]:
    cells, cur, esc = [], [], False
    for ch in line.strip():
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur))
    cells = [c.strip() for c in cells]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def parse_tables(text: str) -> list[dict]:
    """``[{"line": int, "header": [...], "rows": [(line, [...]), ...]}]`` for
    every pipe table; ``line`` numbers are 1-based."""
    tables, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            header = _split_row(lines[i])
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append((j + 1, _split_row(lines[j])))
                j += 1
            tables.append({"line": i + 1, "header": header, "rows": rows})
            i = j
        else:
            i += 1
    return tables


def table_columns(header: list[str]) -> tuple[int, int | None, int] | None:
    """``(status_col, id_col, result_col)`` for a results table, or None when
    the header has no status/tier column. The id column is the one whose
    header names ``claim``; the result column is the one headed ``result`` /
    ``measured`` / ``value``, else the last column that is neither."""
    low = [h.strip().lower() for h in header]
    status = next((k for k, h in enumerate(low) if h in STATUS_HEADERS), None)
    if status is None:
        return None
    id_col = next((k for k, h in enumerate(low) if "claim" in h), None)
    result = next((k for k, h in enumerate(low) if h in RESULT_HEADERS), None)
    if result is None:
        rest = [k for k in range(len(low)) if k != status and k != id_col]
        result = max(rest) if rest else status
    return status, id_col, result


def claim_tables(tables: list[dict]) -> list[tuple[dict, tuple[int, int | None, int]]]:
    """The tables this check owns, with their column roles."""
    out = []
    for t in tables:
        cols = table_columns(t["header"])
        if cols is not None:
            out.append((t, cols))
    return out


def resolve_ids(ids: list[str], claims: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Literal ids and ``*`` globs -> ``{id: claim}``; the second item lists
    the ids and globs that matched nothing."""
    found, missing = {}, []
    for ref in ids:
        if "*" in ref:
            hits = [k for k in claims if fnmatch.fnmatchcase(k, ref)]
            if not hits:
                missing.append(ref)
            for k in hits:
                found[k] = claims[k]
        elif ref in claims:
            found[ref] = claims[ref]
        else:
            missing.append(ref)
    return found, missing


def _weakest(statuses: set[str]) -> str | None:
    for s in _WEAKNESS:
        if s in statuses:
            return s
    return None


def check_row(doc: str, line: int, cells: list[str], cols: tuple[int, int | None, int],
              claims: dict[str, dict], pat: re.Pattern) -> list[str]:
    """Findings for one table row (empty when clean)."""
    status_col, _id_col, result_col = cols
    findings = []
    if len(cells) <= max(status_col, result_col):
        return [f"{doc}:{line}: row has {len(cells)} cell(s); the header promises more"]
    status_cell = cells[status_col]
    ids = [i for k, c in enumerate(cells) if k != status_col for i in pat.findall(c)]
    if not ids:
        return [f"{doc}:{line}: row names no claim id (backticked id or glob) -- every headline number "
                "must name the claim it quotes"]
    found, missing = resolve_ids(ids, claims)
    for ref in missing:
        findings.append(f"{doc}:{line}: `{ref}` is not in {CLAIMS}")
    for cid, c in found.items():
        if c.get("status") in INACTIVE:
            instead = c.get("superseded_by")
            hint = f"; quote `{instead}` instead" if instead else "; it is not a current number"
            findings.append(f"{doc}:{line}: `{cid}` is {c.get('status')}{hint}")
    if not found:
        return findings
    statuses = {str(c.get("status")) for c in found.values()}
    named = set(_STATUS_WORD.findall(status_cell))
    unknown = named - KNOWN_STATUSES
    if unknown:
        findings.append(f"{doc}:{line}: status cell names {sorted(unknown)}, not in the register's vocabulary")
    weakest = _weakest(statuses - INACTIVE)
    if weakest and weakest not in named:
        findings.append(f"{doc}:{line}: status cell says {status_cell!r} but the row's claims include "
                        f"{weakest!r} ({', '.join(sorted(k for k, c in found.items() if c.get('status') == weakest))})"
                        " -- the weakest status in the row is the one the cell must name")
    pool = [n for c in found.values() for n in claim_numbers(c)]
    row_numbers = result_numbers(cells[result_col])
    for n in row_numbers:
        if not number_matches(n, pool):
            findings.append(f"{doc}:{line}: {n} is not a current value of any claim this row names "
                            f"({', '.join(sorted(found))}) -- drift, or a hand-typed number")
    for cid, c in sorted(found.items()):
        for v in value_numbers(c):
            if not any(number_matches(n, [v]) for n in row_numbers):
                findings.append(f"{doc}:{line}: `{cid}` has value {c.get('value')!r} and the row does not quote it")
    return findings


def check_tables(doc: str, text: str, claims: dict[str, dict], pat: re.Pattern) -> list[str]:
    findings, tables = [], claim_tables(parse_tables(text))
    if not tables:
        return [f"{doc}: no table with a `status`/`tier` column -- the results table is missing"]
    for t, cols in tables:
        for line, cells in t["rows"]:
            findings += check_row(doc, line, cells, cols, claims, pat)
    return findings


def check_ids(doc: str, text: str, claims: dict[str, dict], pat: re.Pattern) -> list[str]:
    """Every backticked id anywhere in the document exists; an inactive one is
    cited only on a line that says so (``superseded`` / ``retired`` /
    ``historical``)."""
    findings = []
    for ln, line in enumerate(text.splitlines(), 1):
        for ref in pat.findall(line):
            found, missing = resolve_ids([ref], claims)
            if missing:
                findings.append(f"{doc}:{ln}: `{ref}` is not in {CLAIMS}")
            prose = re.sub(r"`[^`]*`", " ", line).lower()   # the words must come from the prose, not from an id like gnf4.retired.x
            for cid, c in found.items():
                st = c.get("status")
                if st in INACTIVE and not any(w in prose for w in INACTIVE_WORDS):
                    findings.append(f"{doc}:{ln}: `{cid}` is {st} and the line does not say so")
    return findings


# --------------------------------------------------------------------- main --

def load_claim_map(root: Path) -> dict[str, dict]:
    claims, _vocab = load_claims(root, CLAIMS)
    out = {c["id"]: c for c in claims}
    if not out:
        raise ContractError(f"{CLAIMS}: no claims")
    return out


def documents(root: Path) -> list[str]:
    return list(TABLE_DOCS) + [p.relative_to(root).as_posix() for p in sorted(root.glob(ID_DOCS_GLOB))]


def check(root: Path) -> tuple[list[str], int, int]:
    """``(findings, n_docs, n_tables)``."""
    claims = load_claim_map(root)
    pat = id_pattern(claims)
    findings: list[str] = []
    n_tables = 0
    docs = documents(root)
    for rel in docs:
        p = root / rel
        if not p.is_file():
            findings.append(f"{rel}: missing")
            continue
        text = read_text(p)
        if rel in TABLE_DOCS:
            findings += check_tables(rel, text, claims, pat)
            n_tables += len(claim_tables(parse_tables(text)))
        findings += check_ids(rel, text, claims, pat)
    return findings, len(docs), n_tables


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    try:
        findings, n_docs, n_tables = check(root)
    except (ContractError, OSError, ValueError) as e:
        print(f"FAIL: {e}")
        return 2
    for f in findings:
        print(f"FAIL: {f}")
    if findings:
        print(f"FAIL: {len(findings)} finding(s); numbers and ids come from {CLAIMS}")
        return 1
    print(f"OK: {n_docs} documents quote the register -- {n_tables} results table(s) in {', '.join(TABLE_DOCS)} hold "
          "the register's current values, every backticked id exists, no superseded/retired id is cited as current, "
          "no private receipt presented as public")
    return 0


if __name__ == "__main__":
    sys.exit(main())
