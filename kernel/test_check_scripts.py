"""Unit tests for the two repository checkers' pure helpers (no git, no network)."""
from __future__ import annotations

import importlib.util
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
