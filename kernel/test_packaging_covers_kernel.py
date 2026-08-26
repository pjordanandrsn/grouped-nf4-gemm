"""Every shippable kernel module must be in pyproject's py-modules allowlist.

0.3.0 shipped a CHANGELOG announcing the NVMe residency tier and the MXFP4
residency engine -- `K3_RESIDENCY_KINDS`, `fuse_gate_up_segments`,
`Mxfp4NvmeResidency` -- while `mxfp4_residency.py` and `nvme_residency.py` were
absent from `py-modules`. They were never in any wheel. A user who followed the
release notes got `ModuleNotFoundError`.

`py-modules` is an explicit allowlist, so adding a file to `kernel/` does not
package it and nothing complained. This test is the thing that complains: it
diffs the directory against the allowlist and names what is missing. CI runs it,
so the failure lands on the PR that adds the module rather than on a user.

Excluded by design: `test_*` and `conftest` (tests are not part of the
distribution). A module that genuinely should not ship must be added to
`_DELIBERATELY_UNPACKAGED` with a reason, which keeps the decision visible
instead of silent.
"""
import pathlib
import re

# Modules intentionally kept out of the wheel. Empty today; entries need a reason.
_DELIBERATELY_UNPACKAGED: dict[str, str] = {
    # Campaign verdict calculators: preregistration instruments that
    # compute a cycle's verdict from committed receipts. They live in
    # kernel/ beside their PREREG/RESULTS documents for provenance, but
    # they are not library API and are meaningless outside the repo
    # (their bars are hardcoded to their campaigns). Ship the receipts,
    # not the calculator.
    "k1_verdict": "K1 decode-config campaign instrument, not library API",
    "k2_verdict": "K2 vectorized-nibbles campaign instrument, not library API",
    "k3_verdict": "K3 attribution campaign instrument, not library API",
    "k4_verdict": "K4 wide-loads campaign instrument, not library API",
    "k5_verdict": "K5 M-tile probe instrument, not library API",
    "k6_verdict": "K6 bespoke-GEMV instrument, not library API",
    "k6b_verdict": "K6-B productization instrument, not library API",
    "k7_verdict": "K7 round-2 GEMV campaign instrument, not library API",
    "k8_verdict": "K8 fp8-compute-attn instrument, not library API",
    "k9_verdict": "K9 fused-grouping instrument, not library API",
    "k10_verdict": "K10 decode-router instrument, not library API",
    "m2_verdict": "M2 anchor re-certification instrument, not library API",
    "k11_verdict": "K11 M-row feasibility instrument, not library API",
    "f2_verdict": "F2 graph-step-tail instrument, not library API",
    # Campaign BENCH harnesses: same reasoning as the calculators --
    # they measure one prereg's census cells on one box class and are
    # committed for reproducibility, not for import by downstreams.
    "k7_bench": "K7 census bench harness, not library API",
    # The certified anchor: a campaign constant that harnesses read so
    # an uncertified literal cannot gate rentals again. Downstream
    # users of the kernels have no use for it.
    "decode_anchor": "certified box-rental anchor, campaign instrument",
}

# Test files CI does not invoke, each with WHY. This started at 17 silent
# omissions -- including the tests for the very module 0.3.0 failed to ship.
# Everything CPU-reachable was wired into ci.yml instead of being listed here;
# what remains needs a GPU, which the CI runner does not have. Adding an entry
# is a decision that has to be written down, not a silent gap.
_NOT_IN_CI: dict[str, str] = {}
# Empty, and that is the point. It previously held nine "needs CUDA" files, but
# the exclusion is per FILE while every skip marker inside them is per TEST, so
# 57 CPU-reachable tests were excluded as collateral -- 17 in
# test_nvme_residency, 15 in test_mxfp4_residency, 11 in test_arena_experts, 10
# in test_nf4_qlora_grad, 3 in test_arena_equivalence, 1 in test_mxfp4_qlora.
# Nothing ran them but a developer's laptop; the arena pair among them was the
# only coverage on #74-#77.
#
# Naming a file here reads as "CI cannot run this", but the truthful statement
# was "CI cannot run PART of this", and there was no way to say so. ci.yml now
# invokes all nine: device-bound tests skip with their own reasons, which is
# both cheaper and more honest than an allowlist asserting it from outside.
#
# So an entry here now means a file pytest cannot even COLLECT on a CPU runner.
# If you add one, say why, and expect to be asked whether a skip marker inside
# the file would do the job instead.

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _listed_modules() -> set[str]:
    text = (_ROOT / "pyproject.toml").read_text()
    block = re.search(r"py-modules\s*=\s*\[(.*?)\]", text, re.S)
    assert block, "pyproject.toml has no py-modules list"
    return set(re.findall(r'"([^"]+)"', block.group(1)))


def _shippable_modules() -> set[str]:
    out = set()
    for p in (_ROOT / "kernel").glob("*.py"):
        stem = p.stem
        if stem.startswith("test_") or stem == "conftest":
            continue
        out.add(stem)
    return out


def test_every_kernel_module_is_packaged():
    listed = _listed_modules()
    on_disk = _shippable_modules()
    missing = sorted(on_disk - listed - set(_DELIBERATELY_UNPACKAGED))
    assert not missing, (
        "these kernel modules exist but are NOT in pyproject py-modules, so they "
        f"will not be in the wheel: {missing}. Add them to py-modules, or to "
        "_DELIBERATELY_UNPACKAGED with a reason. This exact gap shipped "
        "mxfp4_residency and nvme_residency out of 0.3.0 while the CHANGELOG "
        "announced them."
    )


def test_allowlist_has_no_ghosts():
    listed = _listed_modules()
    on_disk = _shippable_modules()
    ghosts = sorted(listed - on_disk)
    assert not ghosts, (
        f"pyproject py-modules names modules that do not exist on disk: {ghosts}. "
        "A build either fails or silently omits them."
    )


def test_every_test_file_is_actually_RUN_by_ci():
    """A test CI never invokes is a check that cannot fail.

    ci.yml runs pytest against NAMED files, not a directory. That is a
    deliberate choice -- the suite is split by what each step needs installed --
    but it means adding kernel/test_foo.py wires up nothing, silently, and the
    green check stays green about other code. This file was itself added that
    way and would never have run.

    The comment already in ci.yml says the arena tests "shipped with the arena
    format but nothing in CI ran them", so this has happened before. This makes
    the next one fail here instead of shipping.
    """
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    tests = {p.name for p in (_ROOT / "kernel").glob("test_*.py")}
    unrun = sorted(t for t in tests if t not in ci and t not in _NOT_IN_CI)
    assert not unrun, (
        f"these kernel test files are never invoked by ci.yml: {unrun}. Add them "
        "to a pytest step (pick the one whose dependencies they need), or the "
        "green check is green about other code."
    )


def test_conftest_knows_every_interpreter_mode_test_file():
    """conftest's interpreter-file list must match the files that really set it.

    ``TRITON_INTERPRET`` latches process-globally when triton is first
    imported, so a file that sets it at module scope poisons every compiled
    test collected alongside it -- on a device that is a process ABORT, not a
    failure. conftest refuses that mix, but only for the files it knows about.

    That list is hardcoded, so a third interpreter-mode file added later would
    be silently unguarded and the fatal crash would come back. This diffs the
    list against the files that actually assign the variable at import time.
    """
    setters = set()
    for p in sorted((_ROOT / "kernel").glob("test_*.py")):
        for line in p.read_text().splitlines():
            # Column 0 == module scope, which is what latches before any test
            # runs. An indented monkeypatch.delenv inside a test is scoped and
            # harmless, and a docstring mention is not an assignment.
            if re.match(r"""os\.environ(\[|\.setdefault\()\s*["']TRITON_INTERPRET["']""", line):
                setters.add(p.name)
                break

    conftest = (_ROOT / "kernel" / "conftest.py").read_text()
    declared = set(re.findall(r"""["'](test_\w+\.py)["']""", conftest))
    assert setters == declared, (
        f"conftest guards {sorted(declared)} but the files that actually set "
        f"TRITON_INTERPRET at import are {sorted(setters)}. Update "
        "_INTERP_FILES in kernel/conftest.py -- an unguarded interpreter-mode "
        "file crashes the whole pytest process on a GPU box."
    )


def test_the_modules_0_3_0_announced_are_packaged():
    """Named explicitly, because these are the ones that got missed."""
    listed = _listed_modules()
    for m in ("mxfp4_residency", "nvme_residency"):
        assert m in listed, f"{m} is 0.3.0 headline surface and must ship"
