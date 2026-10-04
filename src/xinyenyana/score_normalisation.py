"""Adaptive score normalisation, borrowed from speaker verification.

In speaker verification a raw similarity score is shifted and scaled by how
the same enrolled model, and the same test recording, score against a cohort
of other recordings. Matejka et al. (2017, Interspeech, doi
10.21437/Interspeech.2017-803) found the adaptive symmetric form (AS-norm),
which uses only the top-scoring cohort members, the best of several under
mismatched conditions, and recommend that the cohort include recordings from
the target domain.

Here the score is the class-mean head's: the cosine between a query clip and
the mean of an identity's unit-length enrolment vectors, after the head's own
standardisation. The cohort is the background recordings of the enrolment
split, which carry every enrolled territory's ambient sound and no calls. A
query whose place resembles a cohort background scores highly against it, so
the normalisation discounts exactly the similarity that place contributes.
Whether that removes place without removing the animal is what is measured.

Registered in ``docs/measurement-protocol.md`` (PA-V6): cohort size 50 (all
of them when fewer), the symmetric form
``0.5 * ((s - mu_e) / sd_e + (s - mu_q) / sd_q)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

COHORT_TOP = 50


def _unit(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def _top_stats(scores: Any, top: int) -> tuple[Any, Any]:
    """Mean and standard deviation of the ``top`` highest scores in each row."""

    import numpy as np

    keep = min(top, scores.shape[1])
    part = np.sort(scores, axis=1)[:, -keep:]
    spread = part.std(axis=1)
    return part.mean(axis=1), np.where(spread > 1e-12, spread, 1e-12)


def class_mean_scores(
    *, enrolment: Any, enrolment_identities: Sequence[str], query: Any
) -> tuple[Any, list[str], Any]:
    """Standardised raw scores of every query against every identity's mean.

    The standardisation is the head's (``evaluation.CLASS_MEAN``): every vector
    to unit length, centred on the enrolment mean, then to unit length again for
    the cosine. The mean cosine to an identity's clips is the dot product with
    the mean of those unit vectors, so the model is that mean, not rescaled.
    Returns the score matrix, the identity order and the models.
    """

    import numpy as np

    if np.asarray(enrolment).ndim == 2 and np.asarray(enrolment).shape[1] <= 3:
        raise ValueError(
            "cosine score normalisation is not defined for the raw duration/level controls"
        )
    train = _unit(enrolment)
    centre = train.mean(axis=0, keepdims=True)
    train = _unit(train - centre)
    labels = np.asarray(list(enrolment_identities))
    order = sorted(set(labels.tolist()))
    models = np.vstack([train[labels == name].mean(axis=0) for name in order])
    queries = _unit(_unit(query) - centre)
    return queries @ models.T, order, (models, centre)


def adaptive_s_norm(
    *,
    enrolment: Any,
    enrolment_identities: Sequence[str],
    query: Any,
    cohort: Any,
    top: int = COHORT_TOP,
) -> tuple[Any, Any, list[str]]:
    """Raw and AS-normalised score matrices, and the identity order of their columns."""

    raw, order, (models, centre) = class_mean_scores(
        enrolment=enrolment, enrolment_identities=enrolment_identities, query=query
    )
    cohort_units = _unit(_unit(cohort) - centre)
    queries = _unit(_unit(query) - centre)
    model_mean, model_sd = _top_stats(models @ cohort_units.T, top)
    query_mean, query_sd = _top_stats(queries @ cohort_units.T, top)
    normalised = 0.5 * (
        (raw - model_mean[None, :]) / model_sd[None, :]
        + (raw - query_mean[:, None]) / query_sd[:, None]
    )
    return raw, normalised, order


def accuracy_with_interval(
    *,
    scores: Any,
    order: Sequence[str],
    identities: Sequence[str],
    replicates: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Accuracy of the argmax, with a 95% interval over animals and the per-clip flags."""

    import numpy as np

    predicted = np.asarray(order)[np.argmax(scores, axis=1)]
    actual = np.asarray(list(identities))
    flags = predicted == actual
    animals = sorted(set(actual.tolist()))
    rows = [np.flatnonzero(actual == animal) for animal in animals]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        picked = rng.integers(0, len(animals), size=len(animals))
        draws.append(float(flags[np.concatenate([rows[i] for i in picked])].mean()))
    return {
        "accuracy": float(flags.mean()),
        "identity_block_bootstrap_accuracy_95": [
            float(v) for v in np.quantile(draws, [0.025, 0.975])
        ],
        "query_correct": "".join("1" if flag else "0" for flag in flags),
    }
