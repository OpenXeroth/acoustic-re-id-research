"""Ties cannot expose labels through the ordering of scored observations."""

import numpy as np
import pytest

from xinyenyana.evaluation import verification_metrics
from xinyenyana.verification import equal_error_point, operating_points
from xinyenyana.verification_trials import _equal_error_rate


def test_identical_scores_have_chance_eer_and_no_low_far_acceptance():
    result = _equal_error_rate(np.ones(5), np.ones(5))
    assert result["equal_error_rate"] == 0.5
    assert result["threshold"] > 1.0
    metrics = verification_metrics(np.ones((5, 2)), np.array([0, 1, 0, 1, 0]))
    assert metrics == {"roc_auc": 0.5, "eer": 0.5, "tar_at_far_0_01": 0.0}


@pytest.mark.parametrize("seed", range(5))
def test_grouped_curve_matches_direct_threshold_evaluation(seed):
    rng = np.random.default_rng(seed)
    positive = rng.integers(-2, 3, size=37)
    negative = rng.integers(-2, 3, size=53)
    thresholds, far, frr = operating_points(positive, negative)
    assert len(thresholds) == len(np.unique(np.r_[positive, negative])) + 1
    np.testing.assert_allclose(far, [(negative >= t).mean() for t in thresholds])
    np.testing.assert_allclose(frr, [(positive < t).mean() for t in thresholds])
    error, threshold = equal_error_point(positive, negative)
    assert error == pytest.approx(
        ((negative >= threshold).mean() + (positive < threshold).mean()) / 2
    )
    assert equal_error_point(rng.permutation(positive), rng.permutation(negative)) == (
        error,
        threshold,
    )


def test_perfect_separation_and_reversal():
    assert equal_error_point([2, 2], [1, 1])[0] == 0.0
    assert equal_error_point([1, 1], [2, 2])[0] == 1.0


@pytest.mark.parametrize("positive,negative", [([], [1]), ([1], []), ([np.nan], [1])])
def test_invalid_scores_are_rejected(positive, negative):
    with pytest.raises(ValueError):
        operating_points(positive, negative)


@pytest.mark.parametrize("width", [1, 3])
def test_cosine_score_normalisation_rejects_raw_magnitude_controls(width):
    from xinyenyana.score_normalisation import class_mean_scores

    with pytest.raises(ValueError, match="raw duration/level controls"):
        class_mean_scores(
            enrolment=np.arange(1, 1 + 6 * width).reshape(6, width),
            enrolment_identities=["a", "b"] * 3,
            query=np.ones((2, width)),
        )
