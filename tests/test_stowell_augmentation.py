"""The 2019 remedy: that it is their method, and that it changes the right thing."""

from __future__ import annotations

import wave
from array import array
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.stowell_augmentation import (
    augmented_endpoint,
    plan_augmentation,
    write_mixed_clip,
)


def _clip(path: Path, samples: np.ndarray, rate: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.round(samples * 32768.0).astype("<i2").tobytes())
    return path


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        raw = array("h")
        raw.frombytes(stream.readframes(stream.getnframes()))
    return np.asarray(raw, dtype=np.float64) / 32768.0


def _endpoint(tmp_path: Path, *, identities: int = 3, per_identity: int = 2) -> Any:
    rng = np.random.default_rng(11)
    records = []
    for index in range(identities):
        name = f"bird{index}"
        for clip in range(per_identity):
            for split in ("enrollment", "query"):
                for condition in ("foreground", "background"):
                    stem = f"{name}-{split}-{condition}-{clip}"
                    records.append(
                        ClipRecord(
                            filename=stem,
                            path=_clip(tmp_path / f"{stem}.wav", rng.normal(0, 0.2, 2048)),
                            identity=name,
                            split=split,
                            context={"condition": condition},
                        )
                    )
    return Endpoint(
        name="fixture",
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="docs/measured.md",
    )


def test_every_training_clip_gets_every_other_identity_and_never_its_own(tmp_path: Path) -> None:
    """Their method exists to make each individual's ambient sound appear under
    every individual's calls. Mixing a bird with its own background would leave
    the correlation it is meant to break exactly where it was."""

    endpoint = _endpoint(tmp_path, identities=3, per_identity=2)
    pairs = plan_augmentation(endpoint.records)
    training = [
        record
        for record in endpoint.records
        if record.split == "enrollment" and record.context["condition"] == "foreground"
    ]
    assert len(pairs) == len(training) * 2
    for foreground, donor in pairs:
        assert donor.identity != foreground.identity
        assert donor.split == "enrollment"
        assert donor.context["condition"] == "background"
    for record in training:
        donors = {donor.identity for source, donor in pairs if source.filename == record.filename}
        assert donors == {"bird0", "bird1", "bird2"} - {record.identity}


def test_no_query_clip_reaches_the_training_material(tmp_path: Path) -> None:
    """An augmentation that reached into the query split would be scoring on
    material the model was shown."""

    endpoint = _endpoint(tmp_path)
    for _, donor in plan_augmentation(endpoint.records):
        assert donor.split == "enrollment"
    enlarged, plan = augmented_endpoint(endpoint, scratch=tmp_path / "mixed")
    assert plan["query_clips_touched"] == 0
    query_before = [r.filename for r in endpoint.records if r.split == "query"]
    query_after = [r.filename for r in enlarged.records if r.split == "query"]
    assert query_before == query_after


def test_the_training_material_grows_by_the_identity_count(tmp_path: Path) -> None:
    """Their words: the dataset size increases by a factor of K."""

    endpoint = _endpoint(tmp_path, identities=4, per_identity=3)
    enlarged, plan = augmented_endpoint(endpoint, scratch=tmp_path / "mixed")
    before = plan["training_clips_before"]
    after = before + plan["mixed_clips_added"]
    assert after == before * plan["identities"]


def test_the_mix_is_a_plain_sum_and_not_a_level_matched_one(tmp_path: Path) -> None:
    """Their mixing is sox's plain addition. The project's existing challenge
    mixer rescales the ambient clip to a declared level ratio first, which is a
    different intervention and would make this a test of that instead."""

    loud = _clip(tmp_path / "loud.wav", np.full(1024, 0.4))
    quiet = _clip(tmp_path / "quiet.wav", np.full(1024, 0.02))
    foreground = ClipRecord(
        filename="loud", path=loud, identity="a", split="enrollment", context={}
    )
    donor = ClipRecord(filename="quiet", path=quiet, identity="b", split="enrollment", context={})
    mixed = _read(write_mixed_clip(foreground, donor, tmp_path / "out.wav"))
    # A plain sum keeps the 20:1 ratio the two clips had; a level-matched mix
    # would have raised the quiet one to a declared ratio first.
    assert mixed.max() == pytest.approx(0.95, abs=1e-3)
    assert np.allclose(mixed, mixed[0])
    ratio = 0.4 / 0.02
    reconstructed = 0.95 * (ratio / (ratio + 1) + 1 / (ratio + 1))
    assert mixed[0] == pytest.approx(reconstructed, abs=1e-3)


def test_an_endpoint_with_no_background_recordings_is_refused(tmp_path: Path) -> None:
    """Their method needs the background recordings they told people to publish."""

    endpoint = _endpoint(tmp_path)
    without = Endpoint(
        name="no-backgrounds",
        records=tuple(r for r in endpoint.records if r.context["condition"] == "foreground"),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="docs/measured.md",
    )
    with pytest.raises(ValueError, match="publishes no background recordings"):
        plan_augmentation(without.records)


def test_the_donor_choice_is_fixed_by_a_hash_and_not_by_chance(tmp_path: Path) -> None:
    """No score chooses a donor, and a rerun has to produce the same pairing."""

    endpoint = _endpoint(tmp_path, identities=3, per_identity=3)
    first = [(f.filename, d.filename) for f, d in plan_augmentation(endpoint.records)]
    second = [(f.filename, d.filename) for f, d in plan_augmentation(endpoint.records)]
    assert first == second
    assert len({d for _, d in first}) > 1
