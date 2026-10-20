"""The CI and release workflows' shape, pinned (2026-10-20): no required job can SKIP on a pull_request (GitHub counts a
skipped required check as passing), a pull request's superseded run is cancelled but push runs never share a group,
the guards' triggers cover every path to main, and publishing needs a green CI run at the TAGGED commit."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml  # a hard import: a skipped module would pass every assertion below by never running them

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
GATED_JOBS = ("anchor-gate", "example-runs", "interp-contract", "hip-compile", "wheel-smoke", "discoverability")
GATE_IF = ("${{ github.event_name == 'pull_request' && github.event.action != 'ready_for_review' && "
           "!contains(github.event.pull_request.labels.*.name, 'ready-to-merge') }}")
GROUP = ("${{ github.event_name == 'pull_request' && format('ci-pr-{0}', github.event.pull_request.number) || "
         "format('ci-run-{0}', github.run_id) }}")


def _load(name: str) -> dict:
    d = yaml.safe_load((WF / name).read_text(encoding="utf-8"))
    d["on"] = d.pop(True, d.get("on"))   # PyYAML reads the bare key `on` as the boolean True
    return d


@pytest.mark.parametrize("job", GATED_JOBS)
def test_a_gated_ci_job_cannot_skip(job):
    """No job-level if: or needs: (either makes the job SKIP, which passes), and the first step is the failing gate."""
    j = _load("ci.yml")["jobs"][job]
    assert "if" not in j, f"{job} has a job-level if: again; a skip would pass the required check"
    assert "needs" not in j, f"{job} needs another job: if that job fails or skips, {job} SKIPS, which passes"
    first = j["steps"][0]
    assert first.get("if") == GATE_IF and "exit 1" in first.get("run", "")


def test_pull_requests_cancel_their_superseded_run_and_push_runs_never_share_a_group():
    c = _load("ci.yml")["concurrency"]
    assert c == {"group": GROUP, "cancel-in-progress": "${{ github.event_name == 'pull_request' }}"}
    assert "ci-main" not in c["group"], "push runs must not be collapsed here: every main commit keeps its own run"


def test_ci_still_runs_on_main_pushes_labels_and_dispatch():
    on = _load("ci.yml")["on"]
    assert on["push"] == {"branches": ["main"]}
    assert set(on["pull_request"]["types"]) == {"labeled", "ready_for_review"}
    assert "workflow_dispatch" in on


def test_the_guards_still_scan_every_path_to_main():
    conflict = _load("conflict-marker-guard.yml")["on"]
    assert conflict["push"] == {"branches": ["main"]} and conflict["pull_request"] in (None, {})
    assert _load("private-marker-guard.yml")["on"] == ["push", "pull_request"]


def _publish_job():
    jobs = _load("publish.yml")["jobs"]
    assert set(jobs) == {"publish"}, "a second job in publish.yml would publish without the gate"
    return jobs["publish"]


def _gate(job):
    return next(st for st in job["steps"] if st.get("name", "").startswith("CI is green at the tagged commit"))


GATE_JQ = '[.workflow_runs[] | select(.event == "push" or .event == "workflow_dispatch")] | length'

FAKE_GH = """#!/usr/bin/env python3
# A stand-in for `gh api URL --jq FILTER`. It serves (event, conclusion) runs per head_sha from FAKE_GH_RUNS, applies the
# URL's status=success the way the API does, and only the gate's reviewed filter: anything else exits 97.
import json, os, re, sys
if os.environ.get("FAKE_GH_FAIL"):
    sys.exit(1)
if "FAKE_GH_RAW" in os.environ:
    print(os.environ["FAKE_GH_RAW"]); sys.exit(0)
args = sys.argv[1:]
url = args[1] if len(args) > 1 else ""
if (args[:1] != ["api"] or "--jq" not in args or args[args.index("--jq") + 1] != os.environ["WANT_JQ"]
        or "/actions/workflows/ci.yml/runs?" not in url or "status=success" not in url):
    sys.exit(97)
sha = re.search(r"head_sha=([0-9a-f]+)", url).group(1)
runs = json.loads(os.environ["FAKE_GH_RUNS"]).get(sha, [])
print(sum(1 for event, conclusion in runs if conclusion == "success" and event in ("push", "workflow_dispatch")))
"""


@pytest.fixture
def tagged_checkout(tmp_path):
    """A repo whose tag is one commit behind the branch head, checked out at the tag (as the release checkout is), so
    a branch head (what GITHUB_SHA is on a manual dispatch) and HEAD (the tag) differ."""
    git = shutil.which("git")
    if git is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    # hermetic: no user or system config (a signing or hook setting there would break or slow the fixture)

    def g(*a):
        return subprocess.run([git, *a], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout.strip()
    g("init", "-q")
    g("commit", "-q", "--allow-empty", "-m", "tagged")
    tag = g("rev-parse", "HEAD")
    g("tag", "v0.0.1")
    g("commit", "-q", "--allow-empty", "-m", "later on the branch")
    head = g("rev-parse", "HEAD")
    g("checkout", "-q", "v0.0.1")
    return repo, tag, head


GATE_CASES = [
    ("push run green at the tag", True),
    ("dispatch run green at the tag", True),
    ("a FAILED push run at the tag", False),
    ("green only at the branch head", False),            # the dispatch trap: GITHUB_SHA is not the tag
    ("pull_request runs only at the tag", False),         # they tested refs/pull/N/merge, not this commit
    ("no runs", False),
    ("gh fails", False),                                  # refuse, never assume
    ("gh prints something unreadable", False),            # fail closed: `-lt 1` on a non-integer used to PASS
]


def _run_gate(tmp_path, tagged_checkout, case, gate_run):
    repo, tag, head = tagged_checkout
    extra = {"FAKE_GH_RUNS": json.dumps({
        "push run green at the tag": {tag: [["push", "success"]]},
        "dispatch run green at the tag": {tag: [["workflow_dispatch", "success"]]},
        "a FAILED push run at the tag": {tag: [["push", "failure"]]},
        "green only at the branch head": {head: [["push", "success"]]},
        "pull_request runs only at the tag": {tag: [["pull_request", "success"], ["pull_request", "success"]]},
    }.get(case, {tag: [["push", "success"]]} if case == "gh fails" else {}))}
    if case == "gh fails":
        extra["FAKE_GH_FAIL"] = "1"
    if case == "gh prints something unreadable":
        extra["FAKE_GH_RAW"] = "null"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "gh").write_text(FAKE_GH, encoding="utf-8")
    (bindir / "gh").chmod(0o755)
    script = tmp_path / "gate.sh"
    script.write_text(gate_run, encoding="utf-8")
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r",
           "GITHUB_SHA": head, "WANT_JQ": GATE_JQ, **extra}
    return subprocess.run([shutil.which("bash"), str(script)], cwd=repo, env=env, capture_output=True, text=True,
                          check=False, timeout=60)


def test_publishing_refuses_without_green_ci_at_the_tagged_commit():
    job = _publish_job()
    assert job["permissions"] == {"id-token": "write", "actions": "read"}
    assert "if" not in job and "continue-on-error" not in job
    steps = job["steps"]
    names = [st.get("name", st.get("uses", "")) for st in steps]
    checkout = next(i for i, st in enumerate(steps) if st.get("uses", "").startswith("actions/checkout@"))
    assert steps[checkout]["with"]["ref"] == "${{ inputs.tag || github.ref }}", "the gate reads the commit from this checkout"
    gate = names.index(_gate(job)["name"])
    build = next(i for i, n in enumerate(names) if n.startswith("build sdist"))
    publish = next(i for i, n in enumerate(names) if "pypi-publish" in n or n.startswith("publish to PyPI"))
    assert checkout < gate < build < publish
    for st in (steps[gate], steps[publish]):
        assert "if" not in st and "continue-on-error" not in st, "a gate that can be skipped or ignored gates nothing"
    run = _gate(job)["run"]
    assert "SHA=$(git rev-parse HEAD)" in run
    assert "GITHUB_SHA" not in run, "on a dispatch GITHUB_SHA is the branch head, not the tag: never query by it"
    assert "actions/workflows/ci.yml/runs?head_sha=${SHA}&status=success" in run and GATE_JQ in run
    assert '[ "${n:-0}" -ge 1 ] 2>/dev/null ||' in run, "fail closed on a non-integer count"


@pytest.mark.parametrize("case,ok", GATE_CASES)
def test_the_gate_script_judges_the_tagged_commit(tmp_path, tagged_checkout, case, ok):
    out = _run_gate(tmp_path, tagged_checkout, case, _gate(_publish_job())["run"])
    assert (out.returncode == 0) == ok, (case, out.returncode, out.stdout, out.stderr)
    if not ok:
        assert out.returncode != 97 and "::error::" in out.stdout, (case, out.stdout, out.stderr)
