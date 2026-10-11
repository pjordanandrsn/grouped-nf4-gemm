# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The HIP compile check's own rules, on CPU with the compile step stubbed: what counts as a failure, the allowlist and
its staleness, required coverage, the per-kernel cap and the shipped-module filter. The compiles themselves run in the
``hip-compile`` CI job."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("triton.backends.compiler", reason="the check builds GPUTarget objects from triton")

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hip_compile_check", _ROOT / "scripts" / "hip_compile_check.py")
hcc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hcc)


def _entry(name, module="nf4_grouped", **constexprs):
    return {"module": module, "name": name, "signature": {"x": "*fp32", "B": "constexpr"},
            "constexprs": constexprs or {"B": 64}, "options": {"num_warps": 4}}


def test_error_lines_are_the_assembler_diagnostics_and_nothing_else():
    text = ("warning: something benign\nerror: invalid instruction\n        mov.b32 s0, v2;\n"
            "<stdin>:12:3: error: operand must be a register\nterror: not a diagnostic\n")
    assert [m.group(0).strip() for m in hcc.ERROR_LINE.finditer(text)] == ["error:", "<stdin>:12:3: error:"]


def test_select_covers_every_constexpr_value_within_the_cap():
    entries = [_entry("k", B=b, W=w) for b in (16, 32, 64, 128) for w in (1, 2, 4)]
    chosen = hcc.select(entries, 4)
    assert len(chosen) == 4
    assert {e["constexprs"]["B"] for e in chosen} == {16, 32, 64, 128}
    assert {e["constexprs"]["W"] for e in chosen} == {1, 2, 4}
    assert len(hcc.select(entries, 0)) == 0 and len(hcc.select(entries, 100)) == len(entries)


def _run(tmp_path, monkeypatch, entries, outcomes, allow=(), require=(), cap=0):
    """Run main() with the import and compile stubbed: ``outcomes[(kernel, arch)] = errors`` (empty = ok)."""
    monkeypatch.setattr(hcc, "_kernel", lambda module, name: f"{module}.{name}")
    monkeypatch.setattr(hcc, "_compile_capturing",
                        lambda fn, entry, target: (not outcomes.get((fn, target.arch)), outcomes.get((fn, target.arch), [])))
    paths = {}
    for name, data in (("harvest", entries), ("allow", list(allow)), ("require", list(require))):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data))
    report = tmp_path / "report.json"
    rc = hcc.main([str(paths["harvest"]), "--targets", "gfx942,gfx1100", "--allow", str(paths["allow"]),
                   "--require", str(paths["require"]), "--report", str(report), "--cap", str(cap)])
    return rc, json.loads(report.read_text())["summary"]


def test_an_assembler_error_fails_and_an_allowlisted_one_does_not(tmp_path, monkeypatch):
    entries = [_entry("k")]
    bad = {("nf4_grouped.k", "gfx1100"): ["error: invalid instruction"]}
    rc, summary = _run(tmp_path, monkeypatch, entries, bad)
    assert rc == 1 and summary["failed"] == 1 and summary["ok"] == 1
    allow = [{"kernel": "nf4_grouped.k", "target": "gfx1100", "gate": "refused on AMD by X"}]
    rc, summary = _run(tmp_path, monkeypatch, entries, bad, allow=allow)
    assert rc == 0 and summary["allowed"] == 1 and summary["failed"] == 0


def test_a_stale_allowlist_entry_fails(tmp_path, monkeypatch):
    allow = [{"kernel": "nf4_grouped.k", "target": "gfx1100", "gate": "refused on AMD by X"}]
    rc, summary = _run(tmp_path, monkeypatch, [_entry("k")], {}, allow=allow)
    assert rc == 1 and summary["stale_allow"] == allow


def test_a_required_kernel_that_nothing_compiled_fails(tmp_path, monkeypatch):
    rc, summary = _run(tmp_path, monkeypatch, [_entry("k")], {}, require=["fp8_paged_attn._fp8_combine"])
    assert rc == 1 and summary["required_missing"] == ["fp8_paged_attn._fp8_combine @ gfx942",
                                                       "fp8_paged_attn._fp8_combine @ gfx1100"]
    rc, summary = _run(tmp_path, monkeypatch, [_entry("k")], {}, require=["nf4_grouped.k"])
    assert rc == 0 and summary["required_missing"] == []


def test_a_required_kernel_that_fails_on_one_target_is_missing_there(tmp_path, monkeypatch):
    bad = {("nf4_grouped.k", "gfx1100"): ["error: x"]}
    allow = [{"kernel": "nf4_grouped.k", "target": "gfx1100", "gate": "g"}]
    rc, summary = _run(tmp_path, monkeypatch, [_entry("k")], bad, allow=allow, require=["nf4_grouped.k"])
    assert rc == 1 and summary["required_missing"] == ["nf4_grouped.k @ gfx1100"]


def test_only_shipped_modules_are_compiled_and_uncovered_kernels_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(hcc, "_kernel", lambda module, name: f"{module}.{name}")
    monkeypatch.setattr(hcc, "_compile_capturing", lambda fn, entry, target: (True, []))
    harvest = tmp_path / "h.json"
    harvest.write_text(json.dumps([_entry("_gemm_nf4_grouped"), _entry("_k", module="test_something")]))
    report = tmp_path / "r.json"
    rc = hcc.main([str(harvest), "--targets", "gfx942", "--pyproject", str(_ROOT / "pyproject.toml"),
                   "--report", str(report)])
    summary = json.loads(report.read_text())["summary"]
    assert rc == 0 and summary["excluded_not_shipped"] == ["test_something._k"] and summary["kernels"] == 1
    assert "nf4_grouped._gemm_nf4_grouped" not in summary["uncovered_jit_kernels"]
    assert "fp8_paged_attn._fp8_combine" in summary["uncovered_jit_kernels"]


def test_a_declared_file_compiles_its_specializations_and_states_its_unreached_kernels(tmp_path, monkeypatch):
    monkeypatch.setattr(hcc, "_kernel", lambda module, name: f"{module}.{name}")
    monkeypatch.setattr(hcc, "_compile_capturing", lambda fn, entry, target: (True, []))
    harvest, declared, require, report = (tmp_path / n for n in ("h.json", "d.json", "r.json", "o.json"))
    harvest.write_text("[]")
    unreached = {"nf4_grouped._gemv_nf4_dotpad": "engages only at >= 160 SMs"}
    declared.write_text(json.dumps({"source": "a recorded GPU run", "unreached": unreached,
                                    "specializations": [_entry("_fp8_combine", module="fp8_paged_attn")]}))
    require.write_text(json.dumps(["fp8_paged_attn._fp8_combine"]))
    rc = hcc.main([str(harvest), "--declared", str(declared), "--require", str(require), "--targets", "gfx942",
                   "--report", str(report)])
    summary = json.loads(report.read_text())["summary"]
    assert rc == 0 and summary["required_missing"] == [] and summary["declared_unreached"] == unreached


def test_slow_specializations_compile_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(hcc, "_kernel", lambda module, name: f"{module}.{name}")
    monkeypatch.setattr(hcc, "_compile_capturing", lambda fn, entry, target: (True, []))
    harvest, declared, report = (tmp_path / n for n in ("h.json", "d.json", "o.json"))
    harvest.write_text("[]")
    declared.write_text(json.dumps({"specializations": [_entry("fast")], "slow_specializations": [_entry("slow")]}))
    base = [str(harvest), "--declared", str(declared), "--targets", "gfx942", "--report", str(report)]
    assert hcc.main(base) == 0 and json.loads(report.read_text())["summary"]["kernels"] == 1
    assert hcc.main(base + ["--include-slow"]) == 0 and json.loads(report.read_text())["summary"]["kernels"] == 2
