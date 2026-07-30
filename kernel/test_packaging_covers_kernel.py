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
import os
import pathlib
import re

# Modules intentionally kept out of the wheel. Empty today; entries need a reason.
_DELIBERATELY_UNPACKAGED: dict[str, str] = {}

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


def test_the_modules_0_3_0_announced_are_packaged():
    """Named explicitly, because these are the ones that got missed."""
    listed = _listed_modules()
    for m in ("mxfp4_residency", "nvme_residency"):
        assert m in listed, f"{m} is 0.3.0 headline surface and must ship"
