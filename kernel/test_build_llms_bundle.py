"""scripts/build_llms_bundle.py writes every URL in the plain-text-safe form and refuses a bundle with an unsafe one.

GitHub's traffic (2026-10-20) showed visitors landing on 404 paths ending in ")." and "))." -- markdown links inside
parentheses, read as plain text by a tool whose autolinker took the URL up to the next whitespace."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("build_llms_bundle", ROOT / "scripts" / "build_llms_bundle.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

U = "https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase1/README.md"


@pytest.mark.parametrize("src,want", [
    (f"about 8.1x ([phase 1 notes]({U})).", f"about 8.1x (phase 1 notes <{U}>)."),
    (f"Full list: [CHANGELOG]({U}).", f"Full list: CHANGELOG <{U}>."),
    (f"[![PyPI](https://img.shields.io/x)]({U})", f"PyPI <{U}>"),                 # a badge: alt text and the link
    (f"see {U}).", f"see <{U}>)."),                                               # bare: punctuation stays outside
    (f"see ({U})", f"see (<{U}>)"),
    (f'"repository": "{U}",', f'"repository": "{U}",'),                          # nothing to swallow: byte-identical
    ("https://en.wikipedia.org/wiki/Foo_(bar) is fine", "https://en.wikipedia.org/wiki/Foo_(bar) is fine"),
    (f"already <{U}>.", f"already <{U}>."),
])
def test_every_url_is_written_plain_text_safe(src, want):
    out = B.plain_text_links(src)
    assert out == want and B.unsafe_urls(out) == []


def test_unsafe_urls_finds_what_an_autolinker_would_swallow():
    assert B.unsafe_urls(f"x {U}). y {U}. z <{U}>.") == [f"{U}).", f"{U}."]


def test_the_build_refuses_a_bundle_with_an_unsafe_url(monkeypatch, tmp_path):
    """The rendering cannot make every URL safe (a URL ending in '.' inside a quoted string, say); the build then fails
    rather than ship one, and the check says how many and which."""
    monkeypatch.setattr(B, "plain_text_links", lambda text: text)                # a renderer that misses one
    monkeypatch.setattr(B, "bundle_sources", lambda root, cfg: [{"path": "x.md"}])
    monkeypatch.setattr(B, "read_text", lambda p: f"see ({U}).")
    with pytest.raises(B.ContractError, match=r"1 URL\(s\) in the bundle are directly followed"):
        B.build(tmp_path, {"canonical_url_base": "https://example.invalid/", "title": "t", "preamble": "p"})


def test_the_committed_bundle_is_plain_text_safe():
    text = (ROOT / "llms-full.txt").read_text(encoding="utf-8")
    assert B.unsafe_urls(text) == []                       # (a relative link, with no scheme, is never autolinked)
    assert ".md (https://" not in text                      # every section heading's source URL is in angle brackets
