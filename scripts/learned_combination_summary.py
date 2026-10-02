"""The counts and paired differences `docs/measured.md` reports for architecture one.

Reads `run-learned-combination` result files and prints:

* which arm is outright best in each endpoint-by-head cell of the
  foreground-to-foreground pairing, with ties counted as ties rather than
  awarded to whichever arm is listed first;
* the identity-block bootstrap of the paired difference between two arms on the
  same probe clips, which the registration does not ask for and which is
  reported as an unregistered reading wherever it is quoted.

    python scripts/learned_combination_summary.py results/learned-combination-*.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ARMS = (
    "final layer alone",
    "layers concatenated",
    "final layer projected",
    "learned combination",
)
HEADS = (
    "kernel ridge to one-hot",
    "mean cosine similarity to each identity",
    "single nearest enrollment clip, cosine",
)
PAIRING = "foreground_to_foreground"
TOLERANCE = 1e-12
REPLICATES = 2000
SEED = 17


def best_arms(values: dict[str, float]) -> list[str]:
    """Every arm within tolerance of the top, so a tie reads as a tie."""
    top = max(values.values())
    return [arm for arm in ARMS if abs(values[arm] - top) <= TOLERANCE]


def correct_by_identity(cell: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Per-prediction correctness and the identity each prediction belongs to."""
    identities = [row["actual_label"] for row in cell["predictions"]]
    correct = np.array(
        [row["predicted_label"] == row["actual_label"] for row in cell["predictions"]],
        dtype=float,
    )
    return identities, correct


def paired_difference(
    first: dict[str, Any], second: dict[str, Any], *, replicates: int, seed: int
) -> tuple[float, float, float]:
    """Mean difference in accuracy and its identity-block bootstrap interval.

    The blocks are identities, matching every other interval this project
    reports, and the two arms are read on the same probe clips in the same
    order, so the difference is paired.
    """
    labels_first, correct_first = correct_by_identity(first)
    labels_second, correct_second = correct_by_identity(second)
    if labels_first != labels_second:
        raise ValueError("the two arms did not score the same probe clips in the same order")
    identities = sorted(set(labels_first))
    index = {
        name: np.array([row for row, label in enumerate(labels_first) if label == name])
        for name in identities
    }
    observed = float(correct_first.mean() - correct_second.mean())
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for replicate in range(replicates):
        picked = generator.choice(len(identities), size=len(identities), replace=True)
        rows = np.concatenate([index[identities[i]] for i in picked])
        draws[replicate] = float(correct_first[rows].mean() - correct_second[rows].mean())
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--against", default="final layer alone", choices=ARMS)
    parser.add_argument("--arm", default="learned combination", choices=ARMS)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    results = [json.loads(path.read_text()) for path in arguments.results]
    results.sort(key=lambda result: result["endpoint"])

    outright: Counter[str] = Counter()
    shared: Counter[str] = Counter()
    cells = ties = 0
    print(f"best arm on {PAIRING}, ties counted as ties")
    for result in results:
        for head in HEADS:
            values = {
                arm: result["arms"][arm][head][PAIRING]["classification"]["accuracy"]
                for arm in ARMS
            }
            winners = best_arms(values)
            cells += 1
            if len(winners) == 1:
                outright[winners[0]] += 1
            else:
                ties += 1
                for arm in winners:
                    shared[arm] += 1
            print(
                f"  {result['endpoint']:<22} {head:<42} {max(values.values()):.4f}  "
                + " = ".join(winners)
            )
    print(f"\n{cells} endpoint-by-head cells, {ties} tied at the top")
    print(f"  outright best: {dict(outright)}")
    print(f"  shares a top:  {dict(shared)}")

    print(
        f"\nunregistered: '{arguments.arm}' minus '{arguments.against}', paired on the same "
        f"probe clips,\nidentity-block bootstrap, {arguments.replicates} replicates"
    )
    for result in results:
        for head in HEADS:
            first = result["arms"][arguments.arm][head][PAIRING]
            second = result["arms"][arguments.against][head][PAIRING]
            observed, low, high = paired_difference(
                first, second, replicates=arguments.replicates, seed=arguments.seed
            )
            resolved = "excludes 0" if low > 0 or high < 0 else "includes 0"
            print(
                f"  {result['endpoint']:<22} {head:<42} "
                f"{observed:+.4f} [{low:+.3f}, {high:+.3f}] {resolved}"
            )


if __name__ == "__main__":
    main()
