"""Resume must never mistake a changed input or interrupted write for evidence."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana.extraction_cache import ExtractionCache


def spec(**changes: Any) -> dict[str, Any]:
    return {
        "extractor_sha256": "extractor-one",
        "model_sha256": "model-one",
        "parameters": {"pooling": "mean"},
        "environment": {"numpy": np.__version__},
        **changes,
    }


def test_resume_reuses_complete_entries_but_a_changed_audio_file_does_not(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"first waveform fixture")
    cache = ExtractionCache(tmp_path / "cache", spec())
    calls = []

    def extract() -> dict[str, Any]:
        calls.append(True)
        return {"embedding": np.ones(4)}

    first = cache.get(audio, extract)
    second = ExtractionCache(tmp_path / "cache", spec()).get(audio, extract)
    assert first["embedding"] == pytest.approx(second["embedding"])
    assert len(calls) == 1
    audio.write_bytes(b"changed waveform fixture")
    cache.get(audio, extract)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"model_sha256": "different"},
        {"extractor_sha256": "different"},
        {"environment": {"numpy": "different"}},
        {"parameters": {"pooling": "mean-and-spread"}},
    ],
)
def test_every_extraction_dependency_changes_the_cache(tmp_path: Path, changes: Any) -> None:
    assert ExtractionCache(tmp_path, spec()).root != ExtractionCache(tmp_path, spec(**changes)).root


def test_failed_extraction_does_not_publish_an_entry(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"waveform fixture")
    cache = ExtractionCache(tmp_path / "cache", spec())

    def fail() -> dict[str, Any]:
        raise SystemExit(143)

    with pytest.raises(SystemExit):
        cache.get(audio, fail)
    assert not cache.entry(audio).exists()
    assert cache.get(audio, lambda: {"x": np.ones(2)})["x"].shape == (2,)


def test_nonfinite_features_are_not_cached(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"waveform fixture")
    cache = ExtractionCache(tmp_path / "cache", spec())
    with pytest.raises(ValueError, match="invalid numeric"):
        cache.get(audio, lambda: {"x": np.asarray([np.nan])})
    assert not cache.entry(audio).exists()


def test_corrupt_cache_is_reported_and_not_silently_replaced(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"waveform fixture")
    cache = ExtractionCache(tmp_path / "cache", spec())
    cache.entry(audio).write_bytes(b"interrupted or corrupt")
    with pytest.raises(ValueError):
        cache.get(audio, lambda: {"x": np.ones(2)})
