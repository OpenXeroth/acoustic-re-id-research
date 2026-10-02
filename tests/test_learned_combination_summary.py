"""The paired difference must be paired, and a tie must be counted as a tie."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "learned_combination_summary",
    Path(__file__).resolve().parents[1] / "scripts" / "learned_combination_summary.py",
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def cell(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    return {
        "predictions": [
            {"actual_label": actual, "predicted_label": predicted}
            for actual, predicted in zip(labels, predictions, strict=True)
        ]
    }


LABELS = [name for name in ("a", "b", "c", "d") for _ in range(25)]


def test_an_arm_right_everywhere_against_one_wrong_everywhere_resolves() -> None:
    better = cell(LABELS, LABELS)
    worse = cell(LABELS, ["z"] * len(LABELS))
    observed, low, high = summary.paired_difference(better, worse, replicates=500, seed=3)
    assert observed == pytest.approx(1.0)
    assert low > 0.0
    assert high == pytest.approx(1.0)


def test_two_arms_with_the_same_answers_do_not_resolve() -> None:
    same = cell(LABELS, LABELS)
    observed, low, high = summary.paired_difference(same, same, replicates=500, seed=3)
    assert observed == pytest.approx(0.0)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_a_difference_carried_by_one_identity_does_not_resolve() -> None:
    """Identity blocks are the point: one bird's worth of gain must not clear."""
    better = cell(LABELS, LABELS)
    predictions = list(LABELS)
    for row, label in enumerate(LABELS):
        if label == "a":
            predictions[row] = "z"
    observed, low, high = summary.paired_difference(
        better, cell(LABELS, predictions), replicates=2000, seed=3
    )
    assert observed == pytest.approx(0.25)
    assert low <= 0.0 <= high


def test_arms_scored_on_different_clips_are_refused() -> None:
    with pytest.raises(ValueError, match="same probe clips"):
        summary.paired_difference(
            cell(["a", "b"], ["a", "b"]),
            cell(["b", "a"], ["b", "a"]),
            replicates=9,
            seed=3,
        )


def test_a_tie_at_the_top_names_every_arm_that_shares_it() -> None:
    tied = dict.fromkeys(summary.ARMS, 0.5)
    assert summary.best_arms(tied) == list(summary.ARMS)
    clear = dict(tied)
    clear["learned combination"] = 0.6
    assert summary.best_arms(clear) == ["learned combination"]
