"""scripts/check_change_impact.py never passes when it was not given a diff to check.

It used to print SKIP and exit 0 without --base, so a local review that ran it bare read as clean while CI, which
passes the pull request's base, failed the same diff (experts4bit-qlora#1909, 2026-10-20). Now it diffs against
origin/main by default and FAILs (exit 2) when there is no base to diff against."""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("check_change_impact", ROOT / "scripts" / "check_change_impact.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_change_impact"] = mod
    spec.loader.exec_module(mod)
    return mod


def _repo(path: pathlib.Path, *, origin_main: bool) -> pathlib.Path:
    def git(*a):
        subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True)
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "c")
    if origin_main:
        git("update-ref", "refs/remotes/origin/main", "HEAD")
    return path


def test_an_empty_base_fails_instead_of_skipping(tmp_path, capsys):
    rc = _load().main(["--root", str(tmp_path), "--base", ""])
    out = capsys.readouterr().out
    assert rc == 2 and out.startswith("FAIL: --base is empty") and "SKIP" not in out


def test_no_base_and_no_origin_main_fails_and_says_how_to_fix_it(tmp_path, capsys):
    rc = _load().main(["--root", str(_repo(tmp_path, origin_main=False))])
    out = capsys.readouterr().out
    assert rc == 2 and "origin/main does not resolve" in out and "git fetch origin main" in out and "SKIP" not in out


def test_no_base_diffs_against_origin_main_and_says_so(tmp_path, capsys):
    # Past the base choice the check carries on. This bare repository has no pyproject, so it stops at the role with
    # its own exit 2, but only after it has said which base it took.
    _load().main(["--root", str(_repo(tmp_path, origin_main=True))])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "base: origin/main (no --base given)" and "SKIP" not in out
