"""scripts/check_capabilities.py in the kernel repository: the serving-position rule is the runtime's, by role.

The script is one file in both repositories (scripts/check_shared_tooling.py). Its serving-position
rule assumes one serving position per repository -- true of the runtime package, false here, where
each serving capability is a separate kernel on its own lane (deriving the id prefix and running the
rule here produced two false warnings). So the rule is gated on this repository's role in
docs/system-manifest.json. These tests pin that gate from this side.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "check_capabilities.py"
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("check_capabilities", _SCRIPT)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def test_this_repository_is_the_kernels_role():
    assert cc._system_role(ROOT, "grouped-nf4-gemm") == "kernels"
    assert cc._system_role(ROOT, "experts4bit-qlora") == "runtime"


def test_the_id_namespace_here_is_gnf4():
    pat = cc._status_id_pattern({"gnf4.kernel.a": {}, "gnf4.serve.b": {}})
    assert pat.findall("`gnf4.serve.b` beside the consumer's `e4b.serve.x`") == ["gnf4.serve.b"]


def test_the_real_register_passes_with_no_serving_warning():
    r = subprocess.run([sys.executable, str(_SCRIPT), "--root", str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARN" not in r.stdout and "::warning::" not in r.stdout, r.stdout
