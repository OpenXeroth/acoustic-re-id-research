"""How much of the split effect is real, and how much is six or ten animals.

The largest thing this project has measured is that dividing recordings between
enrolment and scoring matters more than the choice of encoder. Eighteen
comparisons, all falling, by between 0.187 and 0.508. Every one of those is a
single number with nothing attached, and a fall of 0.187 on ten birds is not
obviously distinguishable from a fall of zero.

This attaches an interval to each of them. The unit resampled is the animal and
not the clip, because many clips of one bird are not independent evidence, and
the same resample is applied to both splits so that the difference is paired:
the two figures move together when a bird is drawn twice, and what is left is
the split.

Nothing here reads audio or recomputes an embedding. It scores the authors'
own published vectors under their own two splits, which is what the original
comparison did.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

BOOTSTRAP_REPLICATES = 2000
SEED = 17


def correctness_by_identity(predictions: Sequence[dict[str, Any]]) -> dict[str, list[bool]]:
    """Per-animal lists of whether each of its query clips was named correctly."""

    scored: dict[str, list[bool]] = {}
    for row in predictions:
        actual = str(row["actual_label"])
        scored.setdefault(actual, []).append(str(row["predicted_label"]) == actual)
    if not scored:
        raise ValueError("no predictions to resample")
    return scored


def _accuracy(scored: dict[str, list[bool]], drawn: Sequence[str]) -> float:
    correct = 0
    total = 0
    for name in drawn:
        outcomes = scored[name]
        correct += sum(outcomes)
        total += len(outcomes)
    return correct / total if total else 0.0


def paired_split_interval(
    *,
    random_fold: Sequence[dict[str, Any]],
    recordists: Sequence[dict[str, Any]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    """An interval on the fall from a random division to the recordists' own.

    Both splits score the same animals, so one draw of animals is applied to
    both and the difference is taken inside the draw. An animal that appears in
    one split's predictions and not the other is refused rather than dropped:
    silently dropping it would change which animals the two figures describe.
    """

    import numpy as np

    left = correctness_by_identity(random_fold)
    right = correctness_by_identity(recordists)
    if set(left) != set(right):
        only_left = sorted(set(left) - set(right))
        only_right = sorted(set(right) - set(left))
        raise ValueError(
            f"the two splits score different animals: {only_left} only in the random "
            f"fold, {only_right} only in the recordists' split"
        )
    names = sorted(left)
    observed_left = _accuracy(left, names)
    observed_right = _accuracy(right, names)
    generator = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        drawn = [names[position] for position in generator.integers(0, len(names), len(names))]
        differences[index] = _accuracy(left, drawn) - _accuracy(right, drawn)
    low, high = (
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )
    return {
        "identities": len(names),
        "random_fold_accuracy": observed_left,
        "recordists_accuracy": observed_right,
        "fall": observed_left - observed_right,
        "fall_95": [low, high],
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "identity, the same draw applied to both splits",
        "query_clips": {
            "random_fold": sum(len(values) for values in left.values()),
            "recordists": sum(len(values) for values in right.values()),
        },
    }
