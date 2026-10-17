"""Unit tests for the two repository checkers' pure helpers (no git, no network)."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_jit_kernel_regex_sees_indented_kernels():
    ci = _load("check_change_impact")
    src = (
        "def build(x):\n"
        "    @triton.jit\n"
        "    def inner_k(a, b):\n"
        "        pass\n"
        "    return inner_k\n"
        "\n"
        "@triton.jit\n"
        "def top_k(a):\n"
        "    pass\n"
        "\n"
        "class K:\n"
        "    @staticmethod\n"
        "    @triton.jit\n"
        "    def method_k(a):\n"
        "        pass\n"
    )
    assert ci._kernels(src) == {"inner_k", "top_k", "method_k"}


def test_final_release_tag_ignores_prereleases():
    cm = _load("check_system_manifest")
    assert cm.final_release_tag("v0.30.0") == (0, 30, 0)
    assert cm.final_release_tag("v0.31.0rc1") is None
    assert cm.final_release_tag("v0.31.0.dev2") is None
    assert cm.final_release_tag("v0.31.0.post1") is None
    assert cm.final_release_tag("0.31.0") is None


def test_changelog_fragments_only_where_the_repository_keeps_the_directory(tmp_path):
    # check_change_impact accepts a changelog.d/ fragment as the CHANGELOG.md
    # companion only where the directory exists (experts4bit-qlora since
    # 2026-10-05); without it, CHANGELOG.md itself stays required.
    ci = _load("check_change_impact")
    changed = {"changelog.d/1.md", "changelog.d/README.md", "changelog.d/deleted.md", "changelog.d/x.txt", "CHANGELOG.md"}
    assert ci.changelog_fragments(tmp_path, changed) == []
    (tmp_path / "changelog.d").mkdir()
    for name in ("1.md", "README.md", "x.txt"):
        (tmp_path / "changelog.d" / name).write_text("### t\n", encoding="utf-8")
    assert ci.changelog_fragments(tmp_path, changed) == ["changelog.d/1.md"]


def test_lockstep_record_accepts_a_manifest_record_that_landed_first():
    # The kernel-first lockstep (experts4bit-qlora#1692 -> #1696): the 0.55.0 [fast] floor record was synced into both
    # manifests before the consumer's pyproject moved, so the floor move's diff does not touch the manifest.
    ci = _load("check_change_impact")
    manifest = json.dumps({"compatibility": [
        {"consumer": "experts4bit-qlora", "consumer_versions": ">=0.55.0", "kernel": "grouped-nf4-gemm",
         "floor": ">=0.43.0", "extra": "fast"},
        {"consumer": "experts4bit-qlora", "consumer_versions": ">=0.35.0,<0.55.0", "kernel": "grouped-nf4-gemm",
         "floor": ">=0.30.0", "extra": "fast"}]})
    rec = ci.lockstep_record(manifest, "experts4bit-qlora", "0.55.0", ">=0.43.0")
    assert rec is not None and rec["consumer_versions"] == ">=0.55.0"
    assert ci.lockstep_record(manifest, "experts4bit-qlora", "0.55.0", " >= 0.43.0 ") == rec       # spacing is not a floor
    # the real manifest carries that record too (this repository's copy is byte-identical to the consumer's)
    assert ci.lockstep_record((ROOT / "docs" / "system-manifest.json").read_text(encoding="utf-8"),
                              "experts4bit-qlora", "0.55.0", ">=0.43.0") is not None


def test_lockstep_record_refuses_everything_else():
    ci = _load("check_change_impact")
    manifest = json.dumps({"compatibility": [
        {"consumer": "experts4bit-qlora", "consumer_versions": ">=0.55.0", "kernel": "grouped-nf4-gemm",
         "floor": ">=0.43.0", "extra": "fast"}]})
    assert ci.lockstep_record(manifest, "experts4bit-qlora", "0.55.0", ">=0.44.0") is None      # floor mismatch
    assert ci.lockstep_record(manifest, "experts4bit-qlora", "0.54.1", ">=0.43.0") is None      # range misses the version
    assert ci.lockstep_record(json.dumps({"compatibility": []}), "experts4bit-qlora", "0.55.0", ">=0.43.0") is None
    assert ci.lockstep_record(json.dumps({}), "experts4bit-qlora", "0.55.0", ">=0.43.0") is None  # no record at all
    assert ci.lockstep_record(None, "experts4bit-qlora", "0.55.0", ">=0.43.0") is None           # no base manifest
    assert ci.lockstep_record("{not json", "experts4bit-qlora", "0.55.0", ">=0.43.0") is None
    assert ci.lockstep_record(manifest, "another-consumer", "0.55.0", ">=0.43.0") is None
    assert ci.lockstep_record(manifest.replace('"fast"', '"train"'), "experts4bit-qlora", "0.55.0", ">=0.43.0") is None
    assert ci.lockstep_record(manifest.replace('">=0.55.0"', '"not a range"'), "experts4bit-qlora", "0.55.0",
                              ">=0.43.0") is None                                                # unsupported range
