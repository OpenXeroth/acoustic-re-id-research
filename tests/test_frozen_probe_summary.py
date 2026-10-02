"""The selector's verdict must distinguish lower, equal and higher, not two of three."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location(
    "frozen_probe_summary",
    Path(__file__).resolve().parents[1] / "scripts" / "frozen_probe_summary.py",
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def result(endpoint: str, selected: str, chosen: float, baseline: float) -> dict[str, Any]:
    def candidate(value: float) -> dict[str, Any]:
        return {
            "heads": {
                head: {summary.PAIRING: {"classification": {"accuracy": value}}}
                for head in summary.HEADS
            }
        }

    return {
        "endpoint": endpoint,
        "selected": selected,
        "candidates": {selected: candidate(chosen), summary.BASELINE: candidate(baseline)},
        "distinct_candidates": 12,
        "candidate_keys": 14,
    }


def verdicts(printed: str) -> dict[str, int]:
    counts = {}
    for line in printed.splitlines():
        for name in ("selector lower", "equal", "selector higher"):
            if line.strip().startswith(name + ":"):
                counts[name] = int(line.split(":")[1])
    return counts


def test_a_selector_below_the_final_embedding_counts_as_lower(capsys: Any) -> None:
    printed = run(capsys, [result("e", "stage1.mean", 0.40, 0.97)])
    assert verdicts(printed) == {"selector lower": 3, "equal": 0, "selector higher": 0}


def test_a_selector_above_the_final_embedding_counts_as_higher(capsys: Any) -> None:
    printed = run(capsys, [result("e", "stage3.mean", 0.17, 0.16)])
    assert verdicts(printed) == {"selector lower": 0, "equal": 0, "selector higher": 3}


def test_the_selector_picking_the_final_embedding_counts_as_equal(capsys: Any) -> None:
    """Choosing the baseline itself is not a win for the selector."""
    printed = run(capsys, [result("e", summary.BASELINE, 0.50, 0.50)])
    assert verdicts(printed) == {"selector lower": 0, "equal": 3, "selector higher": 0}


def test_results_that_predate_the_divisor_are_named_rather_than_counted(capsys: Any) -> None:
    older = result("old", "stage1.mean", 0.4, 0.9)
    del older["distinct_candidates"]
    del older["candidate_keys"]
    printed = run(capsys, [older, result("new", "stage1.mean", 0.4, 0.9)])
    assert "not recorded" in printed
    assert "old" in printed.split("not recorded")[1]


def run(capsys: Any, results: list[dict[str, Any]]) -> str:
    import json
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        paths = []
        for index, item in enumerate(results):
            path = Path(work) / f"frozen-{index}.json"
            path.write_text(json.dumps(item))
            paths.append(str(path))
        sys.argv = ["frozen_probe_summary.py", *paths]
        summary.main()
    return capsys.readouterr().out
