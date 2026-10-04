"""Verification operating points at attainable score thresholds.

Equal scores must enter together: ordering observations within a tie uses the
answer labels to create a threshold that cannot be applied to a new score.
"""

from __future__ import annotations

from typing import Any


def operating_points(positive: Any, negative: Any) -> tuple[Any, Any, Any]:
    """Return thresholds, false-accept rates and false-reject rates (score >= threshold).

    Include rejection of every score, followed by every distinct score. Sorting
    and grouping costs O(n log n), including when millions of pairs are scored.
    """
    import numpy as np

    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.ndim != 1 or negative.ndim != 1 or not len(positive) or not len(negative):
        raise ValueError("verification requires two non-empty one-dimensional score arrays")
    scores = np.concatenate([positive, negative])
    if not np.isfinite(scores).all():
        raise ValueError("verification scores must be finite")
    labels = np.arange(len(scores)) < len(positive)
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    false_accept = np.r_[0.0, np.cumsum(~labels)[ends] / len(negative)]
    false_reject = np.r_[1.0, 1.0 - np.cumsum(labels)[ends] / len(positive)]
    thresholds = np.r_[np.nextafter(scores[0], np.inf), scores[ends]]
    return thresholds, false_accept, false_reject


def equal_error_point(positive: Any, negative: Any) -> tuple[float, float]:
    """Approximate EER at the attainable threshold closest to equal error rates.

    Retains the original nearest-operating-point convention, with ties grouped
    and the reject-all endpoint included. The estimate is the mean of FAR and
    FRR at that point; finite samples need not permit exact equality.
    """
    import numpy as np

    thresholds, false_accept, false_reject = operating_points(positive, negative)
    crossing = int(np.argmin(np.abs(false_accept - false_reject)))
    return (
        float((false_accept[crossing] + false_reject[crossing]) / 2),
        float(thresholds[crossing]),
    )
