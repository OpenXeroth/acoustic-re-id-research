"""The song-removal control through the closed-set path.

The property that matters is that the two arms differ in one thing: the samples
inside the published annotations. Same clips, same identities, same split, same
filenames, same head, same seed discipline.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.evaluation import HEADS
from xinyenyana.song_removed import (
    RIVAL_EXPLANATIONS,
    SONG_REMOVED,
    UNTOUCHED,
    compare_arms,
    masked_endpoint,
)


def write_clip(path: Path, samples: np.ndarray, rate: int = 22_050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(values.tobytes())


def great_tit(tmp_path: Path, *, annotated: bool = True, identities: int = 4) -> Endpoint:
    rng = np.random.default_rng(17)
    records = []
    for identity in range(identities):
        for split in ("enrollment", "query"):
            for index in range(3):
                name = f"b{identity}-{split}-{index}"
                path = tmp_path / "source" / f"{name}.wav"
                write_clip(path, rng.normal(0, 0.2, 22_050))
                context = {
                    "condition": "foreground",
                    "year": "2020",
                    "nest_x": float(identity),
                    "nest_y": float(identity * 2),
                }
                if annotated:
                    context["annotation"] = {"onsets": [0.2, 0.6], "offsets": [0.4, 0.8]}
                records.append(ClipRecord(name, path, f"bird-{identity}", split, context))
    return Endpoint("great-tit", tuple(records), ("year",), "fixture", "docs/measured.md")


def test_the_masked_arm_zeroes_the_notes_and_keeps_everything_else(tmp_path: Path) -> None:
    endpoint = great_tit(tmp_path)
    masked = masked_endpoint(endpoint, scratch=tmp_path / "masked")
    assert masked.name == "great-tit-song-removed"
    original = endpoint.records[0]
    written = masked.records[0]
    assert written.filename == original.filename
    assert written.identity == original.identity
    assert written.split == original.split
    with wave.open(str(original.path), "rb") as audio:
        before = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    with wave.open(str(written.path), "rb") as audio:
        after = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert before.size == after.size
    note = slice(int(0.25 * 22_050), int(0.35 * 22_050))
    outside = slice(int(0.90 * 22_050), 22_050)
    assert np.all(after[note] == 0)
    assert np.array_equal(after[outside], before[outside])


def test_an_endpoint_without_per_note_annotations_is_refused(tmp_path: Path) -> None:
    """Only the Great Tit publishes notes. Running this anywhere else would
    write clips identical to their sources and report a difference of zero as
    though the control had been applied."""

    endpoint = great_tit(tmp_path, annotated=False)
    with pytest.raises(ValueError, match="no per-note annotation"):
        masked_endpoint(endpoint, scratch=tmp_path / "masked")


def test_both_arms_are_scored_under_every_head_on_the_same_clips(tmp_path: Path) -> None:
    endpoint = great_tit(tmp_path)
    rng = np.random.default_rng(3)
    axes = rng.normal(size=(4, 16))
    untouched = np.asarray(
        [
            axes[int(record.identity.split("-")[1])] + rng.normal(0, 0.1, 16)
            for record in endpoint.records
        ]
    )
    masked = np.asarray([rng.normal(0, 1, 16) for _ in endpoint.records])
    result = compare_arms(
        endpoint,
        untouched_vectors=untouched,
        masked_vectors=masked,
        representation="embedding.mean",
    )
    assert set(result["arms"]) == {UNTOUCHED, SONG_REMOVED}
    for arm in result["arms"].values():
        assert set(arm) == set(HEADS)
        for head in arm.values():
            assert head["enrollment_calls"] == 12
            assert head["query_calls"] == 12
    assert result["rival_explanations"] == RIVAL_EXPLANATIONS


def test_the_majority_class_floor_is_reported_beside_uniform_chance(tmp_path: Path) -> None:
    """A four-bird result at 0.25 means one thing against a uniform chance of
    0.25 and another against a majority-class floor above it."""

    endpoint = great_tit(tmp_path, identities=4)
    vectors = np.random.default_rng(1).normal(size=(len(endpoint.records), 8))
    result = compare_arms(
        endpoint,
        untouched_vectors=vectors,
        masked_vectors=vectors,
        representation="embedding.mean",
    )
    assert abs(result["uniform_chance"] - 0.25) < 1e-9
    assert abs(result["majority_class_floor"] - 0.25) < 1e-9


def test_a_vector_count_that_does_not_match_the_clips_is_an_error(tmp_path: Path) -> None:
    endpoint = great_tit(tmp_path)
    good = np.zeros((len(endpoint.records), 8))
    with pytest.raises(ValueError, match="vectors for"):
        compare_arms(
            endpoint,
            untouched_vectors=good,
            masked_vectors=np.zeros((3, 8)),
            representation="embedding.mean",
        )


def test_a_separating_untouched_arm_and_a_noise_masked_arm_read_differently(
    tmp_path: Path,
) -> None:
    """The mutation this catches: scoring one arm twice. If both arms came from
    the same vectors their accuracies would be identical whatever was planted."""

    endpoint = great_tit(tmp_path)
    rng = np.random.default_rng(5)
    axes = rng.normal(size=(4, 24))
    untouched = np.asarray(
        [axes[int(r.identity.split("-")[1])] + rng.normal(0, 0.05, 24) for r in endpoint.records]
    )
    masked = rng.normal(0, 1, size=(len(endpoint.records), 24))
    result = compare_arms(
        endpoint,
        untouched_vectors=untouched,
        masked_vectors=masked,
        representation="embedding.mean",
    )
    clean = result["arms"][UNTOUCHED][HEADS[0]]["accuracy"]
    removed = result["arms"][SONG_REMOVED][HEADS[0]]["accuracy"]
    assert clean > 0.9
    assert removed < clean
