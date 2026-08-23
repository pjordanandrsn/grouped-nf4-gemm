"""The phase-2 receipt parse is the provenance link the preregistration
names, and its first version silently read a key that does not exist
(Bugbot, gnf4#202). Pinned here so the link cannot quietly become a
stdout scrape again."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from elastic_e3 import parse_phase2_receipt          # noqa: E402


def test_reads_the_schema_the_harness_writes():
    best, d = parse_phase2_receipt(
        {"sweep": [{"threads": 16, "gbs": 90.0}, {"threads": 32, "gbs": 134.4}],
         "best_gbs": 134.4})
    assert best == 134.4


def test_missing_keys_are_a_hard_error_not_a_fallback():
    with pytest.raises(SystemExit):
        parse_phase2_receipt({"results": [{"gbs": 134.4}]})


def test_an_inconsistent_receipt_is_refused():
    """best_gbs must equal max(sweep); anything else means the receipt and
    the sweep came from different places."""
    with pytest.raises(SystemExit):
        parse_phase2_receipt(
            {"sweep": [{"threads": 16, "gbs": 90.0}], "best_gbs": 134.4})
