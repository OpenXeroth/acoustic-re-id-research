"""A reconstruction must recover historical answers, never improve them selectively."""

import json
from copy import deepcopy

import pytest

from xinyenyana.replay import apply_corrections


def sweep():
    row = {
        "accuracy": 0.5,
        "query_correct": "10",
        "enrollment_fisher_ratio": 1.0,
        "identity_block_bootstrap_accuracy_95": [0, 1],
        "permutation_p": 0.5,
        "numeric_precision": {"speech_matmul": "fp32"},
    }
    return {
        "endpoint": "test",
        "manifest_sha256": "same",
        "query_identities": ["a", "b"],
        "curve": {"m": {"one": row, "two": deepcopy(row)}},
        "chosen": {"m": {"representation": "one", **row}},
        "candidate_vectors": {"m": {"path": "candidates.npz", "sha256": "c"}},
        "stored_vectors": {"m": {"path": "chosen.npz", "sha256": "s", "representation": "one"}},
        "speech_precision": {"m": "fp32"},
    }


def test_reconstruction_replaces_complete_models_and_keeps_originals():
    reference, patch = sweep(), sweep()
    base = sweep()
    base["curve"]["m"]["one"]["query_correct"] = "01"
    base["candidate_vectors"]["m"]["path"] = "old.npz"
    result = apply_corrections(reference, base, [patch])
    assert result["replay_correction"]["reproduction"]["passed"]
    assert result["replay_correction"]["per_model"]["m"]["identical"] == 2
    assert result["candidate_vectors"]["m"]["path"] == "candidates.npz"
    assert base["candidate_vectors"]["m"]["path"] == "old.npz"
    result["curve"]["m"]["one"]["query_correct"] = "00"
    assert patch["curve"]["m"]["one"]["query_correct"] == "10"


@pytest.mark.parametrize(
    "failure",
    ["missing", "different_answers", "wrong_precision", "missing_vectors", "wrong_choice", "split"],
)
def test_reconstruction_rejects_incomplete_or_inconsistent_evidence(failure):
    reference, base, patch = sweep(), sweep(), sweep()
    if failure == "missing":
        del patch["curve"]["m"]["two"]
    elif failure == "different_answers":
        # The reported candidate, with the same aggregate accuracy.
        patch["curve"]["m"]["one"]["query_correct"] = "01"
        patch["chosen"]["m"]["query_correct"] = "01"
    elif failure == "wrong_precision":
        patch["curve"]["m"]["two"]["numeric_precision"]["speech_matmul"] = "tf32"
    elif failure == "missing_vectors":
        patch["stored_vectors"] = {}
    elif failure == "split":
        patch["splits"] = {"enrollment": "changed"}
    else:
        patch["stored_vectors"]["m"]["representation"] = "two"
    with pytest.raises(ValueError):
        apply_corrections(reference, base, [patch])


def test_reconstruction_refuses_alternatives_and_uncorrected_models():
    reference, base, patch = sweep(), sweep(), sweep()
    with pytest.raises(ValueError, match="multiple alternative"):
        apply_corrections(reference, base, [patch, patch])
    reference["curve"]["other"] = deepcopy(reference["curve"]["m"])
    with pytest.raises(ValueError, match="complete historical-candidate check"):
        apply_corrections(reference, base, [patch])
    base["controls"] = {"m": {"four_pairings": {}}}
    with pytest.raises(ValueError, match="recording controls"):
        apply_corrections(sweep(), base, [patch])


def test_reconstruction_checks_vector_bytes_before_publication(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from xinyenyana import cli_v6
    from xinyenyana.cli import app

    monkeypatch.setattr(cli_v6, "_require_lease_and_archive", lambda: None)
    vector = tmp_path / ("0" * 64 + ".npz")
    vector.write_bytes(b"corrupted vector store")
    patch = sweep()
    patch["candidate_vectors"]["m"] = {"path": str(vector), "sha256": "0" * 64}
    path = tmp_path / "patch.json"
    path.write_text(json.dumps(patch))
    output = tmp_path / "new.json"
    result = CliRunner().invoke(
        app,
        [
            "apply-v6-replay-corrections",
            "--reference",
            str(path),
            "--base",
            str(path),
            "--correction",
            str(path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0 and "invalid immutable vector object" in result.output
    assert not output.exists()
