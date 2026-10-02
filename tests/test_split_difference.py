"""An interval on the split effect, and what it refuses to do."""

from __future__ import annotations

import numpy as np
import pytest

from xinyenyana.split_difference import correctness_by_identity, paired_split_interval


def predictions(correct_per_identity: dict[str, tuple[int, int]]) -> list[dict[str, str]]:
    """Query rows for each animal: (number named right, number named wrong)."""

    rows = []
    for name, (right, wrong) in correct_per_identity.items():
        rows += [{"actual_label": name, "predicted_label": name} for _ in range(right)]
        rows += [{"actual_label": name, "predicted_label": "other"} for _ in range(wrong)]
    return rows


def test_a_real_fall_is_found_and_its_interval_excludes_zero() -> None:
    high = predictions({f"b{i}": (19, 1) for i in range(12)})
    low = predictions({f"b{i}": (10, 10) for i in range(12)})
    result = paired_split_interval(random_fold=high, recordists=low, replicates=500)
    assert result["fall"] == pytest.approx(0.45, abs=0.01)
    assert result["fall_95"][0] > 0.0
    assert result["excludes_zero"] is True


def test_no_fall_gives_an_interval_spanning_zero() -> None:
    """The mutation this catches is an interval that excludes zero whatever the
    data, which would make every one of the eighteen comparisons look resolved."""

    same = predictions({f"b{i}": (14, 6) for i in range(12)})
    result = paired_split_interval(random_fold=same, recordists=list(same), replicates=500)
    assert result["fall"] == pytest.approx(0.0, abs=1e-9)
    assert result["excludes_zero"] is False


def test_a_fall_carried_by_one_animal_of_six_does_not_resolve() -> None:
    """Six animals is what this project has on its hardest endpoint. A fall that
    rests on one of them has to come back wide, because a resample that omits
    that animal sees no fall at all."""

    high = predictions({f"b{i}": (10, 10) for i in range(5)} | {"b5": (20, 0)})
    low = predictions({f"b{i}": (10, 10) for i in range(5)} | {"b5": (0, 20)})
    result = paired_split_interval(random_fold=high, recordists=low, replicates=1000)
    assert result["fall"] > 0.1
    assert result["fall_95"][0] <= 0.0


def test_the_same_draw_is_applied_to_both_splits() -> None:
    """Resampling the two splits independently would widen every interval by
    adding variation that the pairing removes."""

    rng = np.random.default_rng(3)
    shared = {f"b{i}": (int(rng.integers(5, 18)), 0) for i in range(10)}
    left = predictions({k: (v[0], 20 - v[0]) for k, v in shared.items()})
    right = predictions({k: (max(v[0] - 4, 0), 20 - max(v[0] - 4, 0)) for k, v in shared.items()})
    paired = paired_split_interval(random_fold=left, recordists=right, replicates=2000)
    width = paired["fall_95"][1] - paired["fall_95"][0]
    assert paired["fall"] == pytest.approx(0.2, abs=0.01)
    # Every animal falls by the same amount, so a paired interval is tight.
    assert width < 0.02


def test_two_splits_scoring_different_animals_are_refused() -> None:
    """Dropping the odd animal would silently change which animals the two
    figures describe, and the difference would no longer be between splits."""

    left = predictions({"a": (5, 5), "b": (5, 5)})
    right = predictions({"a": (5, 5), "c": (5, 5)})
    with pytest.raises(ValueError, match="score different animals"):
        paired_split_interval(random_fold=left, recordists=right, replicates=10)


def test_an_empty_prediction_set_is_an_error() -> None:
    with pytest.raises(ValueError, match="no predictions"):
        correctness_by_identity([])
