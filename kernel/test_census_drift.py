# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The shape census must not drift from its generator, and must say what it covers.

gnf4#353. ``census/shape_census.json`` drives which shapes ~20 harnesses under
``bench/`` and ``roofline/roofline.py`` sweep, so a family missing from it is a
family nothing exercises. Before this, the census was a hardcoded list whose
fetch date lived in a comment, with nothing that fails when the matrix moves
past it: editing ``MODELS`` without regenerating left the committed JSON stale
and silent.

What these tests deliberately do NOT check: whether the census covers what
experts4bit-qlora claims to support. That comparison belongs on the side making
the architecture claim -- ``ci.yml`` already states the principle for the
mirror case, that a consumer's floor is "the consumer's metadata, asserted in
its CI, never in this wheel". Asserting a consumer's support matrix from inside
the kernel package would invert that.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "census" / "make_census.py"
CENSUS = ROOT / "census" / "shape_census.json"


def _gen_namespace() -> dict:
    """Import the generator without running __main__."""
    ns: dict = {}
    src = GEN.read_text().split('if __name__ ==')[0]
    exec(compile(src, str(GEN), "exec"), ns)  # noqa: S102 — our own file
    return ns


def test_the_committed_census_is_what_the_generator_produces():
    """Regenerating must be a no-op. If it is not, someone edited MODELS (or the
    derived-field maths) and did not re-run the generator, so every harness has
    been sweeping a stale matrix."""
    before = CENSUS.read_text()
    r = subprocess.run([sys.executable, str(GEN)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    after = CENSUS.read_text()
    if after != before:
        CENSUS.write_text(before)   # leave the tree as we found it
        raise AssertionError(
            "census/shape_census.json is not what census/make_census.py produces "
            "-- re-run `python census/make_census.py` and commit the result (gnf4#353)"
        )


def test_every_model_in_the_generator_reaches_the_census():
    ns = _gen_namespace()
    declared = [m[0] for m in ns["MODELS"]]
    rows = [m["model"] for m in json.loads(CENSUS.read_text())["models"]]
    assert rows == declared, (
        f"MODELS and the census disagree: generator {declared}, file {rows}"
    )


def test_the_fetch_date_is_data_not_a_comment():
    """A date only a human reads cannot be checked. It is now a field, so a
    reviewer can see how old the shapes are without opening the generator."""
    d = json.loads(CENSUS.read_text())
    assert "fetched" in d, "the census no longer records when its shapes were read (gnf4#353)"
    ns = _gen_namespace()
    assert d["fetched"] == ns["FETCHED"], "the file's fetch date and FETCHED disagree"
    # ISO, and not a placeholder
    y, m, day = d["fetched"].split("-")
    assert len(y) == 4 and 1 <= int(m) <= 12 and 1 <= int(day) <= 31, d["fetched"]


def test_the_census_states_its_coverage_boundary():
    """The count of MODELS is not a support matrix, and #353 was filed partly
    because that had to be inferred.

    The boundary must say that an absent (N, K) still executes -- it takes the
    universal-constant decode plan, measured at median regret 1.000 -- so a
    reader does not conclude that missing from the census means unsupported.
    """
    d = json.loads(CENSUS.read_text())
    assert "coverage" in d, "the census does not state what it covers (gnf4#353)"
    cov = d["coverage"].lower()
    assert "sweep" in cov or "cover" in cov, d["coverage"]
    assert "universal" in cov and "regret" in cov, (
        "the coverage note must record that absent shapes take the measured "
        "universal-constant plan, not that they are unsupported"
    )


def test_the_fallback_the_coverage_note_describes_still_exists():
    """The note above is only true while _decode_plan actually falls through to
    the universal constant. If that changes, the note becomes the wrong kind of
    reassuring."""
    src = (ROOT / "kernel" / "nf4_grouped.py").read_text()
    assert "universal constant" in src, (
        "_decode_plan no longer describes a universal-constant fallback -- "
        "re-check the census coverage note (gnf4#353)"
    )
