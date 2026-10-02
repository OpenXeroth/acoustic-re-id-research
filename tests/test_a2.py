"""Tests for the A2 layer decomposition and its context probes.

Each names an outcome the probe or the endpoint has to refuse or produce. The
probe tests build a target that is an exact linear function of the features and
one that is independent of them, so the floor and the ceiling are both checked
against something computable rather than against a previous run.
"""

from __future__ import annotations

import json
import math
import wave
from array import array
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.a2 import (
    LAYERS,
    ClipRecord,
    Endpoint,
    clip_diagnostics,
    clip_level_dbfs,
    estimability,
    extract_endpoint_layers,
    load_rookid,
    probe_targets,
    run_a2_endpoint,
)
from xinyenyana.evaluation import (
    PER_CLIP_L2,
    PER_DIMENSION,
    leave_one_identity_out,
    prepare_features,
)

SAMPLE_RATE = 22_050


# --- the probe can fail, and does ------------------------------------------


def test_a_target_unrelated_to_the_features_scores_at_the_floor() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(60, 8))
    identities = np.asarray([f"bird{index % 5}" for index in range(60)])
    target = rng.normal(size=60)
    result = leave_one_identity_out(
        features=features,
        identities=identities,
        targets={"noise": target},
        standardisation=PER_DIMENSION,
        ridge_lambda=1.0,
    )
    assert result["noise"]["r_squared"] < 0.2


def test_a_target_that_is_a_function_of_the_features_is_recovered() -> None:
    rng = np.random.default_rng(13)
    features = rng.normal(size=(60, 8))
    identities = np.asarray([f"bird{index % 5}" for index in range(60)])
    target = features @ np.arange(1.0, 9.0)
    result = leave_one_identity_out(
        features=features,
        identities=identities,
        targets={"linear": target},
        standardisation=PER_DIMENSION,
        ridge_lambda=1e-6,
    )
    assert result["linear"]["r_squared"] > 0.9


def test_a_target_no_two_identities_share_is_refused() -> None:
    features = np.random.default_rng(3).normal(size=(12, 4))
    identities = np.asarray([f"bird{index // 3}" for index in range(12)])
    nested = np.asarray([f"nestbox{index}" for index in range(12)], dtype=object)
    with pytest.raises(ValueError, match="no value shared by two identities"):
        leave_one_identity_out(
            features=features,
            identities=identities,
            targets={"nestbox": nested},
            categorical=frozenset({"nestbox"}),
            standardisation=PER_DIMENSION,
            ridge_lambda=1.0,
        )


def test_a_categorical_target_reports_accuracy_against_the_majority_rate() -> None:
    rng = np.random.default_rng(5)
    identities = np.asarray([f"bird{index % 4}" for index in range(48)])
    years = np.asarray(["2020"] * 24 + ["2021"] * 16 + ["2022"] * 8, dtype=object)
    directions = {
        "2020": np.array([3.0, 0.0, 0.0, 0.0, 0.0]),
        "2021": np.array([0.0, 3.0, 0.0, 0.0, 0.0]),
        "2022": np.array([0.0, 0.0, 3.0, 0.0, 0.0]),
    }
    features = rng.normal(size=(48, 5)) * 0.2
    features += np.vstack([directions[year] for year in years])
    result = leave_one_identity_out(
        features=features,
        identities=identities,
        targets={"year": years},
        categorical=frozenset({"year"}),
        standardisation=PER_DIMENSION,
        ridge_lambda=1e-3,
    )
    assert result["year"]["classes"] == 3.0
    assert result["year"]["majority_class_rate"] == pytest.approx(0.5)
    assert result["year"]["accuracy"] > 0.9


def test_a_categorical_target_the_features_do_not_carry_stays_near_chance() -> None:
    rng = np.random.default_rng(23)
    identities = np.asarray([f"bird{index % 4}" for index in range(48)])
    years = np.asarray(["2020", "2021", "2022"] * 16, dtype=object)
    result = leave_one_identity_out(
        features=rng.normal(size=(48, 5)),
        identities=identities,
        targets={"year": years},
        categorical=frozenset({"year"}),
        standardisation=PER_DIMENSION,
        ridge_lambda=1.0,
    )
    assert result["year"]["accuracy"] < 0.6


def test_a_target_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="has 3 values for 6 rows"):
        leave_one_identity_out(
            features=np.zeros((6, 2)),
            identities=np.asarray(["a", "a", "a", "b", "b", "b"]),
            targets={"short": np.zeros(3)},
            standardisation=PER_DIMENSION,
            ridge_lambda=1.0,
        )


def test_one_identity_cannot_be_left_out() -> None:
    with pytest.raises(ValueError, match="at least two identities"):
        leave_one_identity_out(
            features=np.zeros((4, 2)),
            identities=np.asarray(["a"] * 4),
            targets={"x": np.arange(4.0)},
            standardisation=PER_DIMENSION,
            ridge_lambda=1.0,
        )


# --- standardisation -------------------------------------------------------


def test_the_two_standardisation_rules_do_different_things() -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    query = np.array([[2.0, 3.0]])
    _, _, per_dimension = prepare_features(train, query, "source", standardisation=PER_DIMENSION)
    _, _, per_clip = prepare_features(train, query, "source", standardisation=PER_CLIP_L2)
    assert per_dimension["standardization"] == PER_DIMENSION
    assert per_clip["standardization"] == PER_CLIP_L2


def test_an_unknown_standardisation_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown standardisation"):
        prepare_features(np.zeros((2, 2)), np.zeros((1, 2)), "source", standardisation="whitening")


# --- endpoints -------------------------------------------------------------


def _clip(path: Path, *, frequency: float, seconds: float = 0.5, amplitude: int = 12_000) -> None:
    """A tone. ``amplitude`` exists so a caller can make the level actually vary.

    It was fixed, and every clip then had a level within 5e-7 of a decibel of
    every other. That is floating-point dust from sampling a sine, and it passed
    the constant-target guard because the guard's floor is absolute and the level
    of a full-scale tone is near 0 dBFS. Under a later numpy the dust cancelled,
    the target was correctly dropped, and a test that had been asserting the
    presence of a loudness target with no loudness in it failed. A caller that
    wants a loudness target now has to create one.
    """

    frames = int(SAMPLE_RATE * seconds)
    samples = array(
        "h",
        (
            int(amplitude * np.sin(2 * np.pi * frequency * index / SAMPLE_RATE))
            for index in range(frames)
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


def _endpoint(tmp_path: Path) -> Endpoint:
    records = []
    for index in range(8):
        identity = f"bird{index % 2}"
        split = "enrollment" if index < 4 else "query"
        path = tmp_path / f"{identity}-{index}.wav"
        _clip(path, frequency=400.0 + 60.0 * (index % 2), amplitude=3_000 + 1_200 * index)
        records.append(
            ClipRecord(
                filename=f"clip-{index}",
                path=path,
                identity=identity,
                split=split,
                context={"year": "2020" if index % 3 else "2021"},
            )
        )
    return Endpoint(
        name="test-endpoint",
        records=tuple(records),
        categorical_targets=("year",),
        manifest_sha256="0" * 64,
        source_document="docs/measured.md",
    )


def test_an_endpoint_missing_a_split_is_refused(tmp_path: Path) -> None:
    endpoint = _endpoint(tmp_path)
    with pytest.raises(ValueError, match="has splits"):
        Endpoint(
            name="broken",
            records=tuple(
                ClipRecord(
                    filename=r.filename,
                    path=r.path,
                    identity=r.identity,
                    split="enrollment",
                    context=r.context,
                )
                for r in endpoint.records
            ),
            categorical_targets=(),
            manifest_sha256="0" * 64,
            source_document="x",
        )


def test_two_clips_with_one_name_are_refused(tmp_path: Path) -> None:
    endpoint = _endpoint(tmp_path)
    duplicated = (*endpoint.records[:-1], endpoint.records[0])
    with pytest.raises(ValueError, match="two clips with one name"):
        Endpoint(
            name="broken",
            records=duplicated,
            categorical_targets=(),
            manifest_sha256="0" * 64,
            source_document="x",
        )


def test_split_digests_change_when_a_clip_moves_split(tmp_path: Path) -> None:
    endpoint = _endpoint(tmp_path)
    first = endpoint.records[0]
    moved = Endpoint(
        name=endpoint.name,
        records=(
            ClipRecord(
                filename=first.filename,
                path=first.path,
                identity=first.identity,
                split="query",
                context=first.context,
            ),
            *endpoint.records[1:],
        ),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="x",
    )
    assert endpoint.split_digests() != moved.split_digests()


def test_every_layer_comes_out_of_one_read_with_a_row_per_clip(tmp_path: Path) -> None:
    endpoint = _endpoint(tmp_path)
    matrices, diagnostics, names = extract_endpoint_layers(endpoint.records)
    assert set(matrices) == set(LAYERS)
    for layer in LAYERS:
        assert matrices[layer].shape[0] == len(endpoint.records)
        assert matrices[layer].shape[1] == len(names[layer])
    assert len(diagnostics) == len(endpoint.records)
    assert diagnostics[0]["duration_seconds"] == pytest.approx(0.5, abs=1e-3)


def test_estimability_reports_the_diagnostics_the_protocol_requires(tmp_path: Path) -> None:
    _, diagnostics, _ = extract_endpoint_layers(_endpoint(tmp_path).records)
    marks = estimability(diagnostics)
    assert set(marks) == {
        "clips",
        "f0_at_band_edge_fraction",
        "median_harmonic_to_noise_db",
        "formant_estimation_failed_fraction",
        "median_voiced_fraction",
        "median_notes",
        "median_energy_share_below_50_hz",
    }
    assert marks["clips"] == 8.0
    assert 0.0 <= marks["f0_at_band_edge_fraction"] <= 1.0


def test_a_constant_duration_target_is_dropped_with_its_reason(tmp_path: Path) -> None:
    """Every clip here is 0.5 s, as every E01d RookID clip is exactly 3 s."""

    endpoint = _endpoint(tmp_path)
    _, diagnostics, _ = extract_endpoint_layers(endpoint.records)
    targets, categorical, not_defined = probe_targets(endpoint=endpoint, diagnostics=diagnostics)
    assert set(targets) == {"rms_dbfs", "year"}
    assert not_defined == {"log_duration_seconds": "constant on this endpoint"}
    # the loudness target has to carry loudness, or its presence here proves
    # nothing about the guard that dropped the duration beside it
    assert float(np.ptp(targets["rms_dbfs"])) > 1.0
    assert categorical == frozenset({"year"})
    assert len(targets["rms_dbfs"]) == 4
    assert float(targets["rms_dbfs"][0]) < 0.0


def test_a_varying_duration_target_is_kept(tmp_path: Path) -> None:
    endpoint = _endpoint(tmp_path)
    records = tuple(
        ClipRecord(
            filename=r.filename,
            path=r.path,
            identity=r.identity,
            split=r.split,
            context=r.context,
        )
        for r in endpoint.records
    )
    for index, record in enumerate(records):
        _clip(record.path, frequency=440.0, seconds=0.4 + 0.05 * index)
    varied = Endpoint(
        name=endpoint.name,
        records=records,
        categorical_targets=endpoint.categorical_targets,
        manifest_sha256=endpoint.manifest_sha256,
        source_document=endpoint.source_document,
    )
    _, diagnostics, _ = extract_endpoint_layers(varied.records)
    targets, _, not_defined = probe_targets(endpoint=varied, diagnostics=diagnostics)
    assert "log_duration_seconds" in targets
    assert "log_duration_seconds" not in not_defined


def test_a_constant_continuous_target_is_refused_by_the_probe() -> None:
    with pytest.raises(ValueError, match="is constant"):
        leave_one_identity_out(
            features=np.random.default_rng(2).normal(size=(20, 4)),
            identities=np.asarray([f"bird{index % 4}" for index in range(20)]),
            targets={"duration": np.full(20, np.log(3.0))},
            standardisation=PER_DIMENSION,
            ridge_lambda=1.0,
        )


def test_the_rookid_loader_reads_the_manifest_it_is_given(tmp_path: Path) -> None:
    manifest = {
        "sample": {
            "files": [
                {
                    "filename": "Balbo-0001.wav",
                    "local_path": "clips/enrollment/Balbo/Balbo-0001.wav",
                    "identity": "Balbo",
                    "split": "enrollment",
                    "recording": "20200214_090050",
                    "event": "apb",
                },
                {
                    "filename": "Balbo-0002.wav",
                    "local_path": "clips/query/Balbo/Balbo-0002.wav",
                    "identity": "Balbo",
                    "split": "query",
                    "recording": "20200302_091000",
                    "event": "apb",
                },
            ]
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    endpoint = load_rookid(manifest_path=path, sample_root=tmp_path / "sample")
    assert endpoint.name == "rookid"
    assert endpoint.identities == ["Balbo"]
    assert endpoint.records[1].context["recording"] == "20200302_091000"
    assert endpoint.records[0].path == tmp_path / "sample" / "clips/enrollment/Balbo/Balbo-0001.wav"
    assert len(endpoint.manifest_sha256) == 64


def test_the_a2_runner_measures_every_layer_end_to_end(tmp_path: Path) -> None:
    """Drives the command's own entry point.

    The parts of A2 were covered and the assembly was not, which is the shape
    the adoption-rule defect had: every piece correct, the joining wrong, and
    the tests passing because they only ever saw the pieces.
    """

    summary = run_a2_endpoint(endpoint=_endpoint(tmp_path))
    assert summary["endpoint"] == "test-endpoint"
    assert summary["clips"] == 8
    assert summary["identities"] == 2
    assert summary["chance_accuracy"] == 0.5
    assert set(summary["layers"]) == set(LAYERS)
    assert summary["estimability"]["clips"] == 8.0
    for layer in LAYERS:
        block = summary["layers"][layer]
        assert 0.0 <= block["classification"]["accuracy"] <= 1.0
        assert len(block["identity_block_bootstrap_accuracy_95"]) == 2
        assert 0.0 <= block["permutation_control"]["p_value_plus_one"] <= 1.0
        assert block["feature_processing"]["standardization"] == PER_DIMENSION


def test_the_loudness_target_carries_the_clip_level_and_not_a_normalised_one(
    tmp_path: Path,
) -> None:
    """The target has to move when the audio gets louder.

    It did not. Every clip is divided by its own standard deviation when it is
    read, so a level computed after that read is 0.0 dBFS for every clip in
    every corpus. On forty little owl clips the target spanned 0.000002 dB where
    the stored audio spans 12.58 dB, and every ``rms_dbfs`` probe recorded before
    this test was a probe against a constant.
    """

    quiet = tmp_path / "quiet.wav"
    loud = tmp_path / "loud.wav"
    _clip(quiet, frequency=400.0, amplitude=1_000)
    _clip(loud, frequency=400.0, amplitude=16_000)

    levels = [clip_level_dbfs(quiet), clip_level_dbfs(loud)]

    # 16000/1000 is a factor of 16, which is 24.08 dB
    assert levels[1] - levels[0] == pytest.approx(20.0 * math.log10(16.0), abs=0.05)
    assert all(level < 0.0 for level in levels)


def test_the_diagnostics_report_the_level_the_clip_actually_has(tmp_path: Path) -> None:
    records = []
    for index in range(4):
        path = tmp_path / f"clip-{index}.wav"
        _clip(path, frequency=400.0, amplitude=2_000 + 4_000 * index)
        records.append(
            ClipRecord(
                filename=f"clip-{index}",
                path=path,
                identity=f"bird{index % 2}",
                split="enrollment" if index < 2 else "query",
                context={},
            )
        )
    levels = [row["rms_dbfs"] for row in clip_diagnostics(records)]
    assert levels == sorted(levels)
    assert float(np.ptp(levels)) > 5.0
