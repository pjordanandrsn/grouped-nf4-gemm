"""The prose quotes the register, mechanically (scripts/check_readme_claims.py).

Ported from experts4bit-qlora's tests/test_check_readme_claims.py to this
repository's tables (a ``tier`` column, the result before the id) and its
wider document set. These tests pin the parts that decide a pass: the number
tokeniser (what is a result and what is a name), the rounding rule, the
status rule, the column roles, and the id rules outside tables. The last test
runs the real check on this repository's own README, STATUS and solution
pages (no git, no network).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from decimal import Decimal

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


crc = _load("check_readme_claims")


def D(*xs):
    return [Decimal(x) for x in xs]


# ------------------------------------------------------------- the tokeniser --

def test_result_numbers_reads_results_and_skips_names():
    cell = ("1.70× (H100, their TMA live), 2.79× (4090); Qwen3-30B-A3B at B=16, E=256 step; sm_86; fp32 accumulate; "
            "top_k=1 cells; 4-bit-storage regime; ([#319](https://github.com/o/r/issues/319)); 2026-09-05; "
            "7.37 ms/step ±4.2% (≈130–142 tok/s); 144/144 hashes; 1,044 GB/s; 403.7 → 26.5 ms")
    got = crc.result_numbers(cell)
    for present in ("1.70", "2.79", "4090", "7.37", "4.2", "130", "142", "144", "1044", "403.7", "26.5"):
        assert Decimal(present) in got, f"{present} was not read as a result"
    for absent in ("3", "30", "16", "256", "86", "32", "1", "319", "2026", "9", "5"):
        assert Decimal(absent) not in got, f"{absent} was read as a result"
    assert Decimal("-4.2") not in got, "a ± tolerance is unsigned"


def test_result_numbers_signs_ranges_and_thousands():
    got = crc.result_numbers("−0.0528 ppl on wikitext and −0.0662 on C4; 1.52–1.81× per step at 0.75–0.81× VRAM; "
                             "+37.1% over by-index; ≈1,238 tok/s; 2.56× / 3.80× / 6.40× less; 1.4813 → 1.0290")
    assert got == D("-0.0528", "-0.0662", "1.52", "1.81", "0.75", "0.81", "+37.1", "1238", "2.56", "3.80", "6.40",
                    "1.4813", "1.0290")


def test_claim_numbers_come_from_value_unit_and_claim_never_notes():
    c = {"value": 7.37, "unit": "ms/step +-4.2%", "claim": "7.37 ms/step +-4.2%, about 130-142 tok/s",
         "notes": "an unlicensed arm measured 487.8 (x2.146)"}
    got = crc.claim_numbers(c)
    for present in ("7.37", "4.2", "130", "142"):
        assert Decimal(present) in got
    assert Decimal("-4.2") not in got
    assert Decimal("487.8") not in got and Decimal("2.146") not in got, "an unlicensed arm must never match a headline"


def test_value_numbers_handles_ranges_strings_and_null():
    assert crc.value_numbers({"value": "1.16-2.73"}) == D("1.16", "2.73")
    assert crc.value_numbers({"value": "403.7 -> 26.5"}) == D("403.7", "26.5")
    assert crc.value_numbers({"value": 1044}) == D("1044")
    assert crc.value_numbers({"value": None}) == []


# ----------------------------------------------------------- the rounding rule --

@pytest.mark.parametrize("doc, register, ok", [
    ("155", "154.9", True),          # the document's own precision
    ("1.70", "1.7", True),
    ("0.046", "0.04645", True),
    ("-0.053", "-0.0528", True),
    ("100", "98.3", False),          # rounding only: an approximation is not the value
    ("204", "204.6", False),         # truncation is not rounding
    ("238.1", "240.3", False),       # drift
])
def test_number_matches_at_the_document_precision(doc, register, ok):
    assert crc.number_matches(Decimal(doc), [Decimal(register)]) is ok


# ------------------------------------------------------------ the row checks --

CLAIMS = {
    "gnf4.x.a": {"status": "confirmed", "value": 1.7, "unit": "x decode, H100", "claim": "decode 1.70x on H100 and 2.79x on RTX 4090"},
    "gnf4.x.b": {"status": "measured-private", "value": 1044, "unit": "GB/s", "claim": "1,044 GB/s"},
    "gnf4.x.old": {"status": "superseded", "value": 4.67, "superseded_by": "gnf4.x.a", "claim": "4.67x"},
    "gnf4.x.gone": {"status": "retired", "claim": "REFUTED: ~14% worse"},
    "gnf4.x.q": {"status": "confirmed", "value": None, "claim": "never less accurate in any cell"},
    "gnf4.g.a": {"status": "measured", "value": 154.9, "unit": "tok/s", "claim": "154.9 tok/s"},
    "gnf4.g.b": {"status": "measured-private", "value": 204.6, "unit": "tok/s", "claim": "204.6 tok/s"},
}
PAT = crc.id_pattern(CLAIMS)
#: description | result | tier | claim ID  (this repository's README/STATUS shape)
COLS = (2, 3, 1)


def _row(desc, result, tier, cid):
    return crc.check_row("README.md", 7, [desc, result, tier, cid], COLS, CLAIMS, PAT)


def test_a_clean_row_passes():
    assert _row("vs Unsloth, decode", "1.70× (H100), 2.79× (4090)", "confirmed", "`gnf4.x.a`") == []


def test_a_glob_resolves_and_every_matched_value_must_be_quoted():
    assert _row("speed", "155 and 204.6 tok/s", "measured-private", "`gnf4.g.*`") == []
    f = _row("speed", "155 tok/s", "measured-private", "`gnf4.g.*`")
    assert any("`gnf4.g.b` has value 204.6" in x for x in f)


def test_a_number_that_is_not_the_claims_value_is_drift():
    f = _row("vs Unsloth", "1.80× (H100)", "confirmed", "`gnf4.x.a`")
    assert any("1.80 is not a current value" in x for x in f)
    assert any("`gnf4.x.a` has value 1.7 and the row does not quote it" in x for x in f)


def test_superseded_and_retired_ids_fail_and_name_the_replacement():
    f = _row("old", "4.67×", "confirmed", "`gnf4.x.old`")
    assert any("`gnf4.x.old` is superseded; quote `gnf4.x.a` instead" in x for x in f)
    f = _row("gone", "~14% worse", "measured", "`gnf4.x.gone`")
    assert any("`gnf4.x.gone` is retired" in x for x in f)


def test_a_private_receipt_is_never_presented_as_measured():
    f = _row("gemv", "1.70× and 1,044 GB/s", "confirmed", "`gnf4.x.a`, `gnf4.x.b`")
    assert any("'measured-private'" in x and "weakest" in x for x in f)
    assert _row("gemv", "1.70× and 1,044 GB/s", "measured-private", "`gnf4.x.a`, `gnf4.x.b`") == []


def test_a_row_without_an_id_or_with_an_unknown_id_fails():
    assert any("names no claim id" in x for x in _row("speed", "155 tok/s", "measured", ""))
    assert any("`gnf4.x.nope` is not in" in x for x in _row("s", "1.70", "confirmed", "`gnf4.x.nope`"))
    assert any("`gnf4.x.*.zzz` is not in" in x for x in _row("s", "1.70", "confirmed", "`gnf4.x.*.zzz`"))


def test_a_qualitative_claim_needs_no_number():
    assert _row("fidelity", "not less accurate in any cell (fp32 accumulate)", "confirmed", "`gnf4.x.q`") == []


def test_column_roles_follow_the_header():
    assert crc.table_columns(["", "result", "tier", "claim ID in `docs/claims.json`"]) == (2, 3, 1)
    assert crc.table_columns(["", "measured", "tier", "claim ID"]) == (2, 3, 1)
    assert crc.table_columns(["what", "status"]) == (1, None, 0)          # the last non-status column is the result
    assert crc.table_columns(["a", "b"]) is None


def test_tables_are_found_by_their_tier_column_and_escaped_pipes_survive():
    text = ("# t\n\n| | result | tier | claim ID |\n|---|---|---|---|\n"
            "| a | \\|Δ\\| 1.70× (H100), 2.79× | confirmed | `gnf4.x.a` |\n\n| x | y |\n|---|---|\n| 1 | 2 |\n")
    tables = crc.claim_tables(crc.parse_tables(text))
    assert len(tables) == 1 and tables[0][1] == (2, 3, 1)
    (_, cells), = tables[0][0]["rows"]
    assert cells[1] == "|Δ| 1.70× (H100), 2.79×"
    assert crc.check_tables("README.md", text, CLAIMS, PAT) == []


def test_ids_outside_tables_must_exist_and_inactive_ones_must_say_so():
    text = ("see `gnf4.x.a` and `gnf4.x.old` (superseded)\nbut `gnf4.x.gone` is quoted as current\n`gnf4.x.nope`\n"
            "the historical `gnf4.x.old` number\n`e4b.train.something` is the sibling's, left alone\n")
    f = crc.check_ids("docs/STATUS.md", text, CLAIMS, PAT)
    assert any("docs/STATUS.md:2: `gnf4.x.gone` is retired" in x for x in f)
    assert any("docs/STATUS.md:3: `gnf4.x.nope` is not in" in x for x in f)
    assert not any("gnf4.x.old" in x for x in f)
    assert not any("e4b." in x for x in f)


def test_the_id_pattern_is_derived_from_the_register():
    assert crc.id_pattern({"gnf4.a.b": {}, "gnf4.c": {}}).pattern.startswith("`((?:gnf4)")
    assert crc.id_pattern({"e4b.a": {}, "gnf4.b": {}}).findall("`e4b.a` `gnf4.b` `x.y`") == ["e4b.a", "gnf4.b"]


def test_the_repository_documents_pass():
    findings, n_docs, n_tables = crc.check(ROOT)
    assert findings == [], findings
    assert n_docs >= 8 and n_tables >= 2


# ------------------------------------------------ the prose rule (Bugbot, PR #344) --

def _retired_register():
    return {"gnf4.retired.old-thing": {"status": "retired", "retired_reason": "x"},
            "gnf4.kernel.live": {"status": "measured"}}


def test_an_inactive_id_needs_the_word_in_the_prose_not_in_its_own_name():
    claims = _retired_register(); pat = crc.id_pattern(claims)
    bad = crc.check_ids("README.md", "see `gnf4.retired.old-thing` for the number", claims, pat)
    assert bad and "does not say so" in bad[0], bad   # the id's own `retired` segment must not satisfy the rule
    ok = crc.check_ids("README.md", "`gnf4.retired.old-thing` was retired on 2026-08-01", claims, pat)
    assert ok == []
    assert crc.check_ids("README.md", "`gnf4.kernel.live` is the position", claims, pat) == []


def test_the_contract_error_is_the_one_load_claims_raises():
    # the class the script catches must be the one load_claims raises (a local shadow class would pass the name test)
    assert crc.ContractError is crc.load_claims.__globals__["ContractError"]
    assert "class ContractError" not in (ROOT / "scripts" / "check_readme_claims.py").read_text()
