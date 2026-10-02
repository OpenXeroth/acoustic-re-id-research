"""Recovery cannot call a partial output complete or reuse it after its inputs change."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "v6_workflow", Path(__file__).parents[1] / "scripts" / "v6_workflow.py"
)
assert spec and spec.loader
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_existing_result_without_a_successful_contract_is_not_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("XENWARDEN_LEASE", "test-lease")
    monkeypatch.setattr(workflow, "archive", lambda path: None)
    output = tmp_path / "result.json"
    step = {
        "command": [sys.executable, "-c", f"open({str(output)!r}, 'w').write('{{}}')"],
        "outputs": {str(output): {"models": ["missing"]}},
    }
    with pytest.raises(ValueError, match="missing measured model"):
        workflow.run_step(tmp_path, "one", step)
    assert output.exists()
    assert not (tmp_path / "receipts" / "one.json").exists()


def test_completion_is_bound_to_input_bytes_and_resumes_archive_only(tmp_path, monkeypatch):
    monkeypatch.setenv("XENWARDEN_LEASE", "test-lease")
    source = tmp_path / "input.json"
    source.write_text("{}")
    output = tmp_path / "result.json"
    launches = []
    archive_attempts = []

    def run(command, **kwargs):
        launches.append(command)
        output.write_text('{"complete":true}')

    def archive(path):
        archive_attempts.append(path)
        if len(archive_attempts) == 1:
            raise RuntimeError("archive unavailable")

    monkeypatch.setattr(workflow.subprocess, "run", run)
    monkeypatch.setattr(workflow, "archive", archive)
    step = {
        "command": ["model"],
        "inputs": [str(source)],
        "outputs": {str(output): {"equals": {"complete": True}}},
    }
    with pytest.raises(RuntimeError, match="archive unavailable"):
        workflow.run_step(tmp_path, "one", step)
    workflow.run_step(tmp_path, "one", step)
    workflow.run_step(tmp_path, "one", step)
    assert len(launches) == 1
    source.write_text('{"changed":true}')
    workflow.run_step(tmp_path, "one", step)
    assert len(launches) == 2
    assert list((tmp_path / "superseded").glob("*.json"))


def test_missing_fixed_model_fails_even_when_another_model_reproduces(tmp_path):
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps({"models": ["a", "b"], "reproduces_fixed_sweep": {"a": {"identical": True}}})
    )
    with pytest.raises(ValueError, match="every requested model"):
        workflow.validate_result(output, {"fixed": True})


def test_adopting_existing_outputs_does_not_move_or_rerun_them(tmp_path, monkeypatch):
    monkeypatch.setenv("XENWARDEN_LEASE", "test-lease")
    monkeypatch.setattr(workflow, "archive", lambda path: None)
    output = tmp_path / "legacy.json"
    output.write_text('{"complete":true}')
    step = {
        "adopt_existing": True,
        "command": ["must-not-run"],
        "outputs": {str(output): {"equals": {"complete": True}}},
    }
    workflow.run_step(tmp_path, "legacy", step)
    assert output.exists()
    assert not (tmp_path / "superseded").exists()
