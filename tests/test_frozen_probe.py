"""A scored query must not choose the representation it is scored through."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.cli import app
from xinyenyana.evaluation import KERNEL_RIDGE
from xinyenyana.frozen_probe import evaluate_frozen_layers


def case() -> tuple[Endpoint, dict[str, Any]]:
    rng = np.random.default_rng(17)
    records, useful, noise = [], [], []
    for split in ("enrollment", "query"):
        for identity in range(3):
            for index in range(6):
                name = f"{split}-{identity}-{index}"
                records.append(ClipRecord(name, Path(name), str(identity), split, {}))
                useful.append(np.eye(3)[identity] + rng.normal(0, 0.01, 3))
                noise.append(rng.normal(size=3))
    endpoint = Endpoint("synthetic", tuple(records), (), "fixture", "synthetic test")
    return endpoint, {"informative.mean": np.asarray(useful), "noise.mean": np.asarray(noise)}


def test_query_labels_and_vectors_cannot_change_selection() -> None:
    endpoint, vectors = case()
    result = evaluate_frozen_layers(endpoint, vectors, permutations=9, bootstrap_replicates=20)
    changed = replace(
        endpoint,
        records=tuple(
            replace(r, identity=str((int(r.identity) + 1) % 3)) if r.split == "query" else r
            for r in endpoint.records
        ),
    )
    altered_vectors = {name: values.copy() for name, values in vectors.items()}
    for name in altered_vectors:
        altered_vectors[name][18:] *= -1000
    other = evaluate_frozen_layers(
        changed, altered_vectors, permutations=9, bootstrap_replicates=20
    )
    assert result["selected"] == other["selected"] == "informative.mean"
    scored = result["candidates"]["informative.mean"]["heads"][KERNEL_RIDGE][
        "foreground_to_foreground"
    ]
    assert scored["classification"]["accuracy"] == 1.0
    assert len(scored["predictions"]) == 18


def test_cli_refuses_to_extract_without_a_lease(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delenv("XENWARDEN_LEASE", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "run-frozen-probe",
            "--endpoint",
            "rookid",
            "--sample-root",
            str(tmp_path),
            "--cache-root",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code != 0
    assert "XenWarden" in result.output
    assert not (tmp_path / "cache").exists()


def test_required_archive_failure_is_not_a_successful_command(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from xinyenyana import cli

    monkeypatch.setenv("XINYENYANA_RESULT_ARCHIVE", "gs://fixture/results")
    monkeypatch.setattr(
        cli, "archive_result", lambda *a, **k: {"archived": False, "error": "fixture"}
    )
    with pytest.raises(typer.Exit) as caught:
        cli._emit({"fixture": True}, tmp_path / "result.json", require_archive=True)
    assert caught.value.exit_code == 1
    assert (tmp_path / "result.json").exists()


def test_saved_predictions_are_not_duplicated_into_shared_job_logs(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    import json

    from xinyenyana import cli

    monkeypatch.setenv("XINYENYANA_RESULT_ARCHIVE", "gs://fixture/results")
    monkeypatch.setattr(cli, "archive_result", lambda *a, **k: {"archived": True})
    path = tmp_path / "result.json"
    cli._emit({"predictions": [{"actual_label": "private-fixture-label"}]}, path)
    assert "private-fixture-label" in path.read_text()
    assert json.loads(capsys.readouterr().out) == {"archived": True}


def test_input_fingerprint_detects_enrollment_audio_change(tmp_path: Path) -> None:
    from xinyenyana.a2 import ClipRecord, Endpoint
    from xinyenyana.frozen_probe import endpoint_audio_provenance

    enrollment, query = tmp_path / "enroll.wav", tmp_path / "query.wav"
    enrollment.write_bytes(b"first enrollment bytes")
    query.write_bytes(b"fixed query bytes")
    endpoint = Endpoint(
        name="fixture",
        records=(
            ClipRecord("enroll", enrollment, "a", "enrollment", {}),
            ClipRecord("query", query, "a", "query", {}),
        ),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="tests",
    )
    first = endpoint_audio_provenance(endpoint)
    enrollment.write_bytes(b"changed enrollment bytes")
    second = endpoint_audio_provenance(endpoint)
    assert first["audio_files_sha256"] != second["audio_files_sha256"]
    assert first["audio_files"][1] == second["audio_files"][1]


def test_a_duplicate_candidate_does_not_enlarge_the_significance_correction() -> None:
    from xinyenyana.frozen_probe import distinct_candidates

    rng = np.random.default_rng(7)
    base = rng.normal(size=(30, 6))
    other = rng.normal(size=(30, 4))
    vectors = {
        "embedding.mean": base,
        # what the network's own pool produces from the layer below it
        "post_convolution.mean": base.copy(),
        # the same vector with a spread over one time position, which is zero
        "embedding.mean_std": np.hstack([base, np.zeros((30, 6))]),
        "stage1.mean": other,
    }
    groups = distinct_candidates(vectors)
    assert len(groups) == 2
    assert sorted(groups["embedding.mean"]) == [
        "embedding.mean",
        "embedding.mean_std",
        "post_convolution.mean",
    ]
    assert groups["stage1.mean"] == ["stage1.mean"]


def test_candidates_that_differ_in_one_clip_are_counted_apart() -> None:
    from xinyenyana.frozen_probe import distinct_candidates

    rng = np.random.default_rng(11)
    base = rng.normal(size=(30, 6))
    nudged = base.copy()
    nudged[3, 2] += 1e-9
    groups = distinct_candidates({"a.mean": base, "b.mean": nudged})
    assert len(groups) == 2
