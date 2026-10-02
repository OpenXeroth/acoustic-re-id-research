"""Tests for a rate that does not fit on the machine being recorded, not fatal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana import a5
from xinyenyana.a2 import ClipRecord

MODEL = "microsoft/wavlm-base-plus"


def _records() -> list[ClipRecord]:
    return [
        ClipRecord(
            filename=f"{index}.wav",
            path=Path(f"{index}.wav"),
            identity="a" if index % 2 else "b",
            split="enrollment" if index < 4 else "query",
            context={"condition": "foreground"},
        )
        for index in range(6)
    ]


def _layers(width: int = 3) -> dict[int, Any]:
    return {0: np.ones((6, width)), 1: np.zeros((6, width))}


def test_a_rate_that_fits_is_yielded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a5, "SLOWDOWNS", (1, 2))
    monkeypatch.setattr(a5, "speech_layer_vectors", lambda *a, **k: _layers())
    skipped: list[dict[str, Any]] = []
    produced = list(a5.representations_of(MODEL, _records(), device="cpu", not_computed=skipped))
    assert len(produced) == 4
    assert skipped == []


def test_a_rate_that_does_not_fit_is_recorded_and_the_others_still_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a5, "SLOWDOWNS", (1, 2, 3))

    def loader(
        model: str, records: Any, *, slowdown: int, device: str, window_seconds: Any = None
    ) -> dict[int, Any]:
        if slowdown == 3:
            raise RuntimeError("a clip did not fit after 12 attempts")
        return _layers()

    monkeypatch.setattr(a5, "speech_layer_vectors", loader)
    skipped: list[dict[str, Any]] = []
    produced = list(a5.representations_of(MODEL, _records(), device="cpu", not_computed=skipped))
    assert {representation.slowdown for representation, _ in produced} == {1, 2}
    assert skipped == [
        {"model": MODEL, "slowdown": 3, "reason": "a clip did not fit after 12 attempts"}
    ]


def test_without_somewhere_to_record_it_the_failure_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that cannot report the gap must not be handed a quiet one."""

    monkeypatch.setattr(a5, "SLOWDOWNS", (1,))

    def loader(
        model: str, records: Any, *, slowdown: int, device: str, window_seconds: Any = None
    ) -> dict[int, Any]:
        raise RuntimeError("a clip did not fit")

    monkeypatch.setattr(a5, "speech_layer_vectors", loader)
    with pytest.raises(RuntimeError, match="did not fit"):
        list(a5.representations_of(MODEL, _records(), device="cpu"))


def test_a_model_with_no_loader_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    skipped: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="no loader for"):
        list(a5.representations_of("made/up", _records(), device="cpu", not_computed=skipped))
