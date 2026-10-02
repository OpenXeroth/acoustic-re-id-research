"""A tie at the top must be counted as a tie, not awarded to the first arm."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sequence_metric_summary",
    Path(__file__).resolve().parents[1] / "scripts" / "sequence_metric_summary.py",
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def result(endpoint: str, accuracies: dict[str, float]) -> dict[str, Any]:
    arm = {
        name: {
            "closed_set": {"accuracy": value},
            "few_shot": {shots: {"accuracy": value} for shots in summary.SHOTS},
            "open_set": {
                "threshold": value,
                "stranger_acceptance": value,
                "correct_known_acceptance": value,
            },
        }
        for name, value in accuracies.items()
    }
    return {"endpoint": endpoint, "arms": {"stage1.sequence": arm}}


def test_a_tie_at_the_top_is_not_awarded_to_the_arm_listed_first() -> None:
    tied = {"layer average": 0.5, "sequence average": 0.5, "learned sequence": 0.4}
    assert summary.best_arms(tied) == ["layer average", "sequence average"]


def test_an_outright_win_names_one_arm() -> None:
    clear = {"layer average": 0.4, "sequence average": 0.5, "learned sequence": 0.3}
    assert summary.best_arms(clear) == ["sequence average"]


def test_tied_cells_are_reported_separately_from_outright_wins(capsys: Any) -> None:
    summary.tally([result("e", {arm: 0.5 for arm in summary.ARMS})])
    printed = capsys.readouterr().out
    assert "closed set: 1 cells, 1 tied at the top" in printed
    assert (
        "outright totals: {'layer average': 0, 'sequence average': 0, 'learned sequence': 0}"
        in printed
    )


def test_the_median_of_an_even_count_is_the_midpoint() -> None:
    assert summary.median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)
    assert summary.median([1.0, 5.0, 3.0]) == pytest.approx(3.0)


def test_an_unavailable_open_set_is_named_rather_than_scored(capsys: Any) -> None:
    one = result("e", {arm: 0.5 for arm in summary.ARMS})
    for arm in one["arms"]["stage1.sequence"].values():
        arm["open_set"] = {"unavailable": "no stranger clips"}
    summary.open_set([one])
    assert capsys.readouterr().out.count("open set unavailable") == len(summary.ARMS)
