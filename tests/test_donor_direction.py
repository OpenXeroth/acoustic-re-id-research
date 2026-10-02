"""The donor-direction reading must separate a donor pull from no donor pull."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "donor_direction", Path(__file__).resolve().parents[1] / "scripts" / "donor_direction.py"
)
assert SPEC is not None and SPEC.loader is not None
donor_direction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(donor_direction)

IDENTITIES = frozenset({"a", "b", "c", "d"})


def rotated(labels: list[str], step: int) -> list[str]:
    order = sorted(IDENTITIES)
    return [order[(order.index(label) + step) % len(order)] for label in labels]


def test_every_wrong_answer_naming_the_donor_is_the_strongest_result() -> None:
    actual = np.array(sorted(IDENTITIES) * 8)
    donor = np.array(rotated(list(actual), 1))
    predicted = donor.copy()
    wrong, observed, shuffled, p_value = donor_direction.donor_share(
        actual, predicted, donor, replicates=199, seed=3
    )
    assert wrong == len(actual)
    assert observed == 1.0
    assert shuffled < 0.6
    assert p_value == pytest.approx(1 / 200)


def test_wrong_answers_avoiding_the_donor_cannot_clear() -> None:
    actual = np.array(sorted(IDENTITIES) * 8)
    donor = np.array(rotated(list(actual), 1))
    predicted = np.array(rotated(list(actual), 2))
    wrong, observed, shuffled, p_value = donor_direction.donor_share(
        actual, predicted, donor, replicates=199, seed=3
    )
    assert wrong == len(actual)
    assert observed == 0.0
    assert shuffled > 0.0
    assert p_value == 1.0


def test_a_donor_carrying_the_query_identity_is_refused() -> None:
    actual = np.array(["a", "b"])
    with pytest.raises(ValueError, match="own identity"):
        donor_direction.donor_share(
            actual, np.array(["b", "a"]), actual.copy(), replicates=9, seed=3
        )


def test_an_arm_with_no_wrong_answer_is_refused_rather_than_scored() -> None:
    actual = np.array(["a", "b"])
    with pytest.raises(ValueError, match="no wrong answers"):
        donor_direction.donor_share(
            actual, actual.copy(), np.array(["b", "a"]), replicates=9, seed=3
        )


def test_the_donor_identity_is_checked_against_the_endpoint() -> None:
    assert donor_direction.donor_identity("linhart2015_day2_a_0019.wav", IDENTITIES) == "a"
    with pytest.raises(ValueError, match="is not an identity"):
        donor_direction.donor_identity("linhart2015_day2_zz_0019.wav", IDENTITIES)


def test_the_shuffle_never_hands_a_query_its_own_identity() -> None:
    """The shuffled share pins down which pool the comparison draws from.

    Every answer here is wrong, so it is one of the three identities that are not
    the query's own. A shuffle over those three matches it one time in three. A
    shuffle that could also return the query's own identity spends a quarter of
    its draws on a value no wrong answer can equal, and settles at one in four.
    """
    actual = np.array(sorted(IDENTITIES) * 8)
    donor = np.array(rotated(list(actual), 1))
    predicted = np.array(rotated(list(actual), 2))
    _, _, shuffled, _ = donor_direction.donor_share(
        actual, predicted, donor, replicates=499, seed=11
    )
    assert shuffled == pytest.approx(1 / (len(IDENTITIES) - 1), abs=0.02)
