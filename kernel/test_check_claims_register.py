"""The register's bookkeeping, mechanically (scripts/check_claims_register.py).

A 2026-09-05 audit of the 0.30.1 register found receipt paths no file matched,
a month where a date belonged, a retired row without its reason and a
``quoted_in`` pointing at a section the README no longer had -- all green,
because every check keyed on ``status``. These tests pin each rule on a small
fake register, then run the real check on this repository's own
docs/claims.json (no git, no network).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ccr = _load("check_claims_register")

VOCAB = {"confirmed": "c", "measured": "m", "measured-private": "mp", "projected": "p",
         "retired": "r", "superseded": "s", "open": "o"}


def _repo(tmp_path: pathlib.Path, claims: list[dict]) -> pathlib.Path:
    """A minimal repository; idempotent, so one test can call it several times."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "grouped-nf4-gemm"\nversion = "0.0.0"\n'
        '[project.urls]\nSource = "https://github.com/o/r"\n', encoding="utf-8")
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "claims.json").write_text(
        json.dumps({"status_vocabulary": VOCAB, "claims": claims}), encoding="utf-8")
    (tmp_path / "docs" / "capabilities.json").write_text(json.dumps(
        {"project": {"related_projects": [{"canonical_package": "experts4bit-qlora"}]}}), encoding="utf-8")
    (tmp_path / "kernel").mkdir(exist_ok=True)
    (tmp_path / "kernel" / "RESULTS-a.md").write_text("# a\n\n## 0.24.0 — 2026-09-03\n\nbody\n", encoding="utf-8")
    (tmp_path / "kernel" / "receipts").mkdir(exist_ok=True)
    (tmp_path / "kernel" / "receipts" / "r1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("# t\n\n## What was retired\n\nline\n", encoding="utf-8")
    return tmp_path


def _findings(tmp_path, claims, sibling=None):
    f, _s, _n = ccr.check_register(_repo(tmp_path, claims), sibling)
    return f


GOOD = {"id": "gnf4.x.good", "status": "measured", "claim": "x", "measured_on": "2026-09-05",
        "evidence": ["kernel/RESULTS-a.md", "kernel/receipts/", {"path": "kernel/RESULTS-a.md", "section": "0.24.0"},
                     {"glob": "kernel/receipts/*.json"}, "https://github.com/o/r/issues/1"],
        "quoted_in": ["README.md", "README.md#What was retired", "README.md#L3", "README.md:5"]}


def test_a_clean_register_passes(tmp_path):
    assert _findings(tmp_path, [GOOD]) == []


@pytest.mark.parametrize("entry, needle", [
    ("kernel/RESULTS-missing.md", "does not exist at HEAD"),
    ("receipts-m3/", "does not exist at HEAD"),
    ("kernel/RESULTS-*.md (0.7.0)", "not a bare repository path"),
    ("CHANGELOG.md 0.24.0", "not a bare repository path"),
    ({"path": "kernel/RESULTS-a.md", "section": "0.99.0"}, "no heading starting with '0.99.0'"),
    ({"glob": "kernel/nothing/*.json"}, "glob matches nothing"),
    ({"repository": "someone-else", "path": "x.md"}, "not a related project"),
    ("https://github.com/other/repo/issues/1", "must be under this repository"),
    ({"thing": 1}, "not a known form"),
])
def test_each_unresolvable_evidence_form_is_named(tmp_path, entry, needle):
    c = dict(GOOD, evidence=[entry])
    f = _findings(tmp_path, [c])
    assert any(needle in x for x in f), f


def test_cross_repository_evidence_is_skipped_without_a_sibling_and_resolved_with_one(tmp_path):
    c = dict(GOOD, evidence=[{"repository": "experts4bit-qlora", "path": "bench/x/RESULTS.md"}])
    root = _repo(tmp_path / "a", [c])
    f, s, _ = ccr.check_register(root)
    assert f == [] and any("not resolved (no --sibling" in x for x in s)
    sib = tmp_path / "sib"
    (sib / "bench" / "x").mkdir(parents=True)
    f, s, _ = ccr.check_register(root, sib)
    assert any("missing in --sibling" in x for x in f)
    (sib / "bench" / "x" / "RESULTS.md").write_text("r", encoding="utf-8")
    f, s, _ = ccr.check_register(root, sib)
    assert f == [] and s == []


def test_measured_rows_need_an_iso_date_and_a_receipt(tmp_path):
    f = _findings(tmp_path, [{k: v for k, v in GOOD.items() if k != "measured_on"}])
    assert any("needs measured_on" in x for x in f)
    f = _findings(tmp_path, [dict(GOOD, measured_on="2026-08")])
    assert any("'2026-08' is not an ISO calendar date" in x for x in f)
    f = _findings(tmp_path, [dict(GOOD, measured_on="2026-02-30")])
    assert any("not an ISO calendar date" in x for x in f)
    f = _findings(tmp_path, [dict(GOOD, evidence=[])])
    assert any("needs a public receipt" in x for x in f)
    priv = dict(GOOD, id="gnf4.x.priv", status="measured-private", evidence=[])
    assert any("needs evidence_private" in x for x in _findings(tmp_path, [priv]))
    assert _findings(tmp_path, [dict(priv, evidence_private=["INT4B16/P1"])]) == []
    # an open row needs no date, but a date it carries must still be a date
    op = {"id": "gnf4.open.x", "status": "open", "claim": "OPEN", "evidence": ["https://github.com/o/r/issues"]}
    assert _findings(tmp_path, [op]) == []
    assert any("not an ISO" in x for x in _findings(tmp_path, [dict(op, measured_on="soon")]))


def test_superseded_needs_an_active_successor_that_names_it_back(tmp_path):
    old = {"id": "gnf4.x.old", "status": "superseded", "claim": "old", "evidence": ["kernel/RESULTS-a.md"]}
    f = _findings(tmp_path, [old, GOOD])
    assert any("needs superseded_by" in x for x in f)
    f = _findings(tmp_path, [dict(old, superseded_by="gnf4.x.nope"), GOOD])
    assert any("'gnf4.x.nope' is not in" in x for x in f)
    f = _findings(tmp_path, [dict(old, superseded_by="gnf4.x.good"), GOOD])
    assert any("does not list 'gnf4.x.old' in its supersedes" in x for x in f)
    gone = dict(GOOD, id="gnf4.x.gone", status="retired", retired_reason="why", supersedes=["gnf4.x.old"])
    f = _findings(tmp_path, [dict(old, superseded_by="gnf4.x.gone"), gone])
    assert any("has status 'retired', not an active claim" in x for x in f)
    ok = dict(GOOD, supersedes=["gnf4.x.old"])
    assert _findings(tmp_path, [dict(old, superseded_by="gnf4.x.good"), ok]) == []
    # supersedes must point at a superseded row, and superseded_by needs the status
    f = _findings(tmp_path, [dict(GOOD, supersedes=["gnf4.x.other"]), dict(GOOD, id="gnf4.x.other")])
    assert any("whose status is 'measured', not 'superseded'" in x for x in f)
    assert any("superseded_by is set but status is 'measured'" in x
               for x in _findings(tmp_path, [dict(GOOD, superseded_by="gnf4.x.good")]))


def test_retired_needs_its_reason(tmp_path):
    r = {"id": "gnf4.retired.x", "status": "retired", "claim": "REFUTED", "evidence": ["kernel/RESULTS-a.md"]}
    assert any("needs retired_reason" in x for x in _findings(tmp_path, [r]))
    assert _findings(tmp_path, [dict(r, retired_reason="K7 measured it flat")]) == []


@pytest.mark.parametrize("entry, needle", [
    ("README.md#Status / roadmap", "no heading starting with 'Status / roadmap'"),
    ("CHANGELOG.md 0.18.0", "does not exist at HEAD"),
    ("README.md#L400", "line 400 is outside the file"),
    ("docs/nothing.md", "does not exist at HEAD"),
])
def test_quoted_in_entries_resolve(tmp_path, entry, needle):
    f = _findings(tmp_path, [dict(GOOD, quoted_in=[entry])])
    assert any(needle in x for x in f), f


def test_placeholder_words_are_refused_on_active_rows_only(tmp_path):
    f = _findings(tmp_path, [dict(GOOD, notes="the roofline term is pending")])
    assert any("placeholder word 'pending'" in x for x in f)
    assert any("'TBD'" in x for x in _findings(tmp_path, [dict(GOOD, notes="number TBD")]))
    assert _findings(tmp_path, [dict(GOOD, notes="pendingly is not the word; independent is fine")]) == []
    op = {"id": "gnf4.open.x", "status": "open", "claim": "OPEN", "notes": "confirmation pending",
          "evidence": ["https://github.com/o/r/issues"]}
    assert _findings(tmp_path, [op]) == []


def test_heading_prefix_match_folds_case_and_whitespace():
    text = "# T\n\n## What changed — retired, superseded, corrected\n\n## 0.24.0 — 2026-09-03\n"
    assert ccr.heading_found(text, "What changed")
    assert ccr.heading_found(text, "what   CHANGED")
    assert ccr.heading_found(text, "0.24.0")
    assert ccr.heading_found(text, "0.24")          # a prefix: 0.24 matches 0.24.0
    assert not ccr.heading_found(text, "0.25.0")


def test_the_repository_register_passes():
    findings, skips, n = ccr.check_register(ROOT)
    assert findings == [], findings
    assert n >= 30
    # the cross-repository dgrad-gate receipt is listed, never silently passed
    assert any("experts4bit-qlora" in s for s in skips), skips
