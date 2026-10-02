"""The counts and medians `docs/measured.md` reports for architecture two.

Reads `run-sequence-metric` result files and prints, for the three arms:

* which arm is outright best in each endpoint-by-layer closed-set cell and each
  endpoint-by-layer-by-shot few-shot cell, with ties counted as ties and
  reported separately rather than awarded to whichever arm is listed first;
* the median over layers of the open-set threshold, the fraction of evaluation
  strangers it accepted, and the correct known acceptance at it.

    python scripts/sequence_metric_summary.py results/sequence-metric-*.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ARMS = ("layer average", "sequence average", "learned sequence")
SHOTS = ("1", "2", "5")
TOLERANCE = 1e-12


def best_arms(values: dict[str, float]) -> list[str]:
    """Every arm within tolerance of the top, so a tie reads as a tie."""
    top = max(values.values())
    return [arm for arm in ARMS if abs(values[arm] - top) <= TOLERANCE]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def tally(results: list[dict[str, Any]]) -> None:
    outright: dict[str, Counter[tuple[str, str]]] = {"closed": Counter(), "few": Counter()}
    shared: dict[str, Counter[tuple[str, str]]] = {"closed": Counter(), "few": Counter()}
    cells: Counter[str] = Counter()
    ties: Counter[str] = Counter()
    tied_on_rook: Counter[str] = Counter()
    for result in results:
        endpoint = result["endpoint"]
        for by_arm in result["arms"].values():
            pairs = [("closed", {arm: by_arm[arm]["closed_set"]["accuracy"] for arm in ARMS})]
            for shots in SHOTS:
                pairs.append(
                    ("few", {arm: by_arm[arm]["few_shot"][shots]["accuracy"] for arm in ARMS})
                )
            for kind, values in pairs:
                cells[kind] += 1
                winners = best_arms(values)
                if len(winners) == 1:
                    outright[kind][(endpoint, winners[0])] += 1
                    continue
                ties[kind] += 1
                if endpoint.startswith("rookid"):
                    tied_on_rook[kind] += 1
                for arm in winners:
                    shared[kind][(endpoint, arm)] += 1
    endpoints = sorted(result["endpoint"] for result in results)
    for kind, label in (("closed", "closed set"), ("few", "few shot")):
        print(
            f"{label}: {cells[kind]} cells, {ties[kind]} tied at the top, "
            f"{tied_on_rook[kind]} of those on the rook"
        )
        for endpoint in endpoints:
            outright_counts = " ".join(
                f"{arm.split()[0]}={outright[kind][(endpoint, arm)]}" for arm in ARMS
            )
            shared_counts = " ".join(
                f"{arm.split()[0]}={shared[kind][(endpoint, arm)]}" for arm in ARMS
            )
            print(f"  {endpoint:<22} outright {outright_counts} | shares a tie {shared_counts}")
        totals = {
            arm: sum(outright[kind][(endpoint, arm)] for endpoint in endpoints) for arm in ARMS
        }
        print(f"  outright totals: {totals}")
        print()


def open_set(results: list[dict[str, Any]]) -> None:
    print("open set, median over the layers. the threshold accepts at most a tenth")
    print("of the development strangers and is applied unchanged to the evaluation ones.")
    print()
    print(f"{'endpoint':<22} {'arm':<18} {'threshold':>9} {'stranger':>9} {'known':>9}")
    for result in results:
        endpoint = result["endpoint"]
        for arm in ARMS:
            blocks = [by_arm[arm]["open_set"] for by_arm in result["arms"].values()]
            blocks = [block for block in blocks if "unavailable" not in block]
            if not blocks:
                print(f"{endpoint:<22} {arm:<18}   open set unavailable")
                continue
            threshold = median([block["threshold"] for block in blocks])
            stranger = median([block["stranger_acceptance"] for block in blocks])
            known = median([block["correct_known_acceptance"] for block in blocks])
            print(f"{endpoint:<22} {arm:<18} {threshold:>9.4f} {stranger:>9.3f} {known:>9.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    arguments = parser.parse_args()
    results = [json.loads(path.read_text()) for path in arguments.results]
    results.sort(key=lambda result: result["endpoint"])
    tally(results)
    open_set(results)


if __name__ == "__main__":
    main()
