"""Publication tables must reject incomplete evidence and omit private observations."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from xinyenyana.open_set import STRANGER_BUDGETS, summarise_allocations


@pytest.fixture
def tables(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("paper_v6_controls"), importlib.import_module(
        "paper_v6_open_set"
    )


@pytest.mark.parametrize(
    "ready,validated,missing,invalid",
    [
        (False, 107, ["pending.json"], {}),
        (True, 224, [], {"changed.json": "wrong input bytes"}),
        (True, 192, [], {}),
    ],
)
def test_table_assembly_requires_a_complete_gate(
    tables: Any,
    tmp_path: Path,
    ready: bool,
    validated: int,
    missing: list[str],
    invalid: dict[str, str],
) -> None:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "ready_for_numerical_assembly": ready,
                "required_files": 224,
                "validated_files": validated,
                "missing": missing,
                "invalid": invalid,
            }
        )
    )
    assert tables[0].required_count() == 224
    with pytest.raises(ValueError, match="224 evidence contracts"):
        tables[0].complete_gate(path)


def test_as_norm_table_retains_intervals_but_no_private_query_annotations(tables: Any) -> None:
    arm = {
        "accuracy": 0.5,
        "identity_block_bootstrap_accuracy_95": [0.2, 0.8],
        "query_correct": "1010",
        "identity": "PRIVATE_SENTINEL",
    }
    condition = {
        "chance": 0.25,
        "raw_class_mean": arm,
        "as_norm": arm,
        "predictions": {"query_ids": ["PRIVATE_SENTINEL"]},
    }
    clean = tables[0].clean_as_norm(
        {
            "cohort_clips": 100,
            "cohort_top": 50,
            "calls_scored": condition,
            "backgrounds_scored": condition,
        }
    )
    assert clean["calls_scored"]["raw_class_mean"] == {
        "accuracy": 0.5,
        "identity_block_bootstrap_accuracy_95": [0.2, 0.8],
    }
    assert "PRIVATE_SENTINEL" not in json.dumps(clean)
    assert "query_correct" not in json.dumps(clean)


def allocations() -> list[dict[str, Any]]:
    return [
        {
            "roles": {"private": ["PRIVATE_SENTINEL"]},
            "observations": {"test": [{"identity": "PRIVATE_SENTINEL"}]},
            "closed_set_accuracy": 0.7,
            "known_vs_unknown_auroc": 0.8,
            "equal_error_rate": 0.2,
            "best_acceptance_at_budget": {f"{b:.2f}": 0.6 for b in STRANGER_BUDGETS},
            "true_accept_at_false_accept": {f"{b:.2f}": 0.65 for b in STRANGER_BUDGETS},
            "calibrated_on_other_birds": {
                f"{b:.2f}": {
                    "test": {
                        "known_correct_accept_rate": 0.4,
                        "unknown_false_accept_rate": i / 15,
                        "balanced_accuracy_known": 0.3,
                        "balanced_accuracy_unknown": 0.8,
                        "geometric_mean_known_unknown": (0.3 * 0.8) ** 0.5,
                    }
                }
                for b in STRANGER_BUDGETS
            },
        }
        for i in range(16)
    ]


def test_open_set_tables_report_achieved_stranger_rates_without_identity_records(
    tables: Any,
) -> None:
    runs = allocations()
    row = {
        "representation": "embedding",
        "allocations": runs,
        "allocation_sensitivity": summarise_allocations(runs),
    }
    public = tables[1].public_metrics(row)
    assert public["allocation_sensitivity"]["achieved_stranger_acceptance_calibrated_at_0.10"] == {
        "minimum": 0.0,
        "median": 0.5,
        "maximum": 1.0,
    }
    assert public["allocation_sensitivity"]["geometric_mean_known_unknown_calibrated_at_0.10"][
        "median"
    ] == pytest.approx((0.3 * 0.8) ** 0.5)
    assert "PRIVATE_SENTINEL" not in json.dumps(public)
    assert "allocations" not in public


def test_open_set_tables_reject_a_changed_allocation_summary(tables: Any) -> None:
    runs = allocations()
    summary = summarise_allocations(runs)
    summary["closed_set_accuracy"]["median"] = 0.1
    with pytest.raises(ValueError, match="do not reconstruct"):
        tables[1].public_metrics(
            {"representation": "embedding", "allocations": runs, "allocation_sensitivity": summary}
        )
    with pytest.raises(ValueError, match="sixteen allocations"):
        tables[1].public_metrics(
            {
                "representation": "embedding",
                "allocations": runs[:15],
                "allocation_sensitivity": summary,
            }
        )
