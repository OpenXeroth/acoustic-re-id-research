import numpy as np
import pytest

from xinyenyana.identity_information import beecher_hs, effective_group_size, f_ratios


def test_effective_group_size_is_the_group_size_when_balanced() -> None:
    assert effective_group_size([10, 10, 10]) == pytest.approx(10.0)
    assert effective_group_size([5, 15]) < 10.0


def test_f_ratio_matches_the_textbook_one_way_anova() -> None:
    from scipy.stats import f_oneway

    rng = np.random.default_rng(0)
    labels = [g for g in "abcd" for _ in range(7)]
    column = rng.normal(size=len(labels)) + np.repeat([0.0, 0.5, 1.0, 2.0], 7)
    expected = f_oneway(*[column[np.asarray(labels) == g] for g in "abcd"]).statistic
    assert f_ratios(column[:, None], labels)[0] == pytest.approx(expected)


def test_separated_individuals_carry_more_bits_than_shuffled_labels() -> None:
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(6, 32)) * 3.0
    labels = [str(i) for i in range(6) for _ in range(12)]
    vectors = np.vstack([centres[int(label)] + rng.normal(size=32) for label in labels])
    true = beecher_hs(vectors, labels)
    shuffled = beecher_hs(vectors, list(rng.permutation(labels)))
    assert true["components_used"] == 5
    assert true["hs_bits"] > 3.0 * max(shuffled["hs_bits"], 0.1)
    assert true["hs_bits_by_components"] == sorted(true["hs_bits_by_components"])


def test_components_never_exceed_individuals_less_one() -> None:
    rng = np.random.default_rng(2)
    labels = [g for g in "abc" for _ in range(20)]
    result = beecher_hs(rng.normal(size=(60, 50)), labels)
    assert result["components_used"] == 2
    assert all(bits >= 0 for bits in result["hs_bits_by_components"])
