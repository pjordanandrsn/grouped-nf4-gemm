"""Tests for trace provenance.

A routing trace is reproducible only if the weights AND the library that runs
them are pinned. Neither was recorded, and the cost was concrete: the committed
OLMoE traces agree with a fresh capture of the same model, prompt and seed on
18% of layer-steps under transformers 5.15.1, and nothing in the repo said
which transformers they came from, so the drift was invisible
(bench/cold-engine/RESULTS-topk-frequency.md).

The helper must therefore be three things, each pinned below: deterministic,
sensitive to a config-only change, and incapable of crashing a capture that
has already paid for a model download.
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from capture_routing import env_fingerprint            # noqa: E402


def _model():
    m = types.SimpleNamespace()
    m.config = types.SimpleNamespace(architectures=["OlmoeForCausalLM"])
    return m


def _write(d, name, text):
    with open(os.path.join(d, name), "w") as f:
        f.write(text)


def test_records_the_library_versions():
    with tempfile.TemporaryDirectory() as d:
        e = env_fingerprint(d, _model())
    assert e["transformers"] and e["torch"] and e["python"]


def test_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        _write(d, "config.json", '{"a": 1}')
        assert env_fingerprint(d, _model()) == env_fingerprint(d, _model())


def test_detects_a_config_only_change():
    """The case that motivates hashing config.json separately: a revision that
    changes num_experts_per_tok leaves every weight untouched and changes the
    routing completely."""
    with tempfile.TemporaryDirectory() as d:
        _write(d, "config.json", '{"num_experts_per_tok": 8}')
        before = env_fingerprint(d, _model())["config"]
        _write(d, "config.json", '{"num_experts_per_tok": 4}')
        after = env_fingerprint(d, _model())["config"]
    assert before != after


def test_detects_a_weight_change_via_the_index():
    with tempfile.TemporaryDirectory() as d:
        _write(d, "model.safetensors.index.json", '{"weight_map": {"a": "1"}}')
        before = env_fingerprint(d, _model())["weight_index"]
        _write(d, "model.safetensors.index.json", '{"weight_map": {"a": "2"}}')
        after = env_fingerprint(d, _model())["weight_index"]
    assert before != after


def test_falls_back_to_a_single_shard():
    with tempfile.TemporaryDirectory() as d:
        _write(d, "model.safetensors", "xx")
        assert env_fingerprint(d, _model())["weight_index"].startswith(
            "model.safetensors:")


def test_names_the_file_it_hashed():
    """The hash is useless if you cannot tell which file produced it."""
    with tempfile.TemporaryDirectory() as d:
        _write(d, "config.json", "{}")
        assert env_fingerprint(d, _model())["config"].startswith("config.json:")


def test_missing_files_do_not_crash_a_paid_capture():
    """This runs AFTER the model download and the decode loop. Raising here
    would throw away the whole run over metadata."""
    e = env_fingerprint("/definitely/not/a/path", _model())
    assert e["config"] is None and e["weight_index"] is None
    assert e["transformers"]


def test_result_is_json_serialisable():
    """It goes straight into the trace's meta line."""
    with tempfile.TemporaryDirectory() as d:
        json.dumps(env_fingerprint(d, _model()))
