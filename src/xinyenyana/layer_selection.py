"""A second rule for choosing a model's layer: sessions held out within enrolment.

A5 chooses each model's layer and playback rate by the Fisher ratio of the
enrolment clips. Where every animal's enrolment clips come from its own
recordings, the layer that separates animals best in enrolment may be the one
that separates recordings best (Lighten, review of v5, Design Atlas Fig. 12).
This rule scores every candidate on enrolment recordings that the head did not
see: each animal's enrolment sessions are dealt round-robin into folds, the
fixed head is fitted on the other folds and scores the held-out one, and the
candidate with the highest held-out accuracy is chosen. Query clips are never
read, so the rule can be applied before scoring, as the Fisher ratio is.

An animal with a single enrolment session cannot be scored on a session the
head has not seen; its clips train the head in every fold and are never
scored. The number of animals and clips scored is reported with the figure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

FOLDS = 5
SALT = "xyy-v6-layer-holdout-20260925"
RIDGE_LAMBDA = 1.0


def session_folds(
    identities: Sequence[str], sessions: Sequence[str | None], *, folds: int = FOLDS
) -> list[int]:
    """The fold of every clip: each animal's sessions dealt round-robin by a salted hash."""

    order: dict[str, list[str | None]] = {}
    for identity, session in zip(identities, sessions, strict=True):
        known = order.setdefault(identity, [])
        if session not in known:
            known.append(session)
    fold_of: dict[tuple[str, str | None], int] = {}
    for identity, own in order.items():
        ranked = sorted(
            own, key=lambda s: hashlib.sha256(f"{SALT}|{identity}|{s}".encode()).hexdigest()
        )
        for index, session in enumerate(ranked):
            fold_of[(identity, session)] = index % folds
    return [fold_of[(i, s)] for i, s in zip(identities, sessions, strict=True)]


def held_out_accuracy(
    vectors: Any, identities: Sequence[str], sessions: Sequence[str | None], *, folds: int = FOLDS
) -> dict[str, Any]:
    """Accuracy of the fixed head on enrolment sessions it was not fitted on."""

    import numpy as np

    from xinyenyana.evaluation import prepare_features, ridge_projection, standardisation_for

    matrix = np.asarray(vectors, dtype=np.float64)
    labels = np.asarray(identities)
    fold = np.asarray(session_folds(identities, sessions, folds=folds))
    sessions_per_animal = {
        animal: len(
            {s for i, s in zip(identities, sessions, strict=True) if i == animal and s is not None}
        )
        for animal in set(identities)
    }
    incomplete = {i for i, s in zip(identities, sessions, strict=True) if s is None}
    scorable_animals = {i for i, n in sessions_per_animal.items() if n >= 2 and i not in incomplete}
    scorable = np.asarray([i in scorable_animals for i in identities])
    rule = standardisation_for(matrix.shape[1])
    correct = 0
    scored = 0
    for held in range(folds):
        test = (fold == held) & scorable
        # Single-session animals remain gallery competitors in every fold.
        # They cannot be scored, but dropping them would make fold zero easier.
        train = ~test
        if not test.any() or len(set(labels[train])) < 2:
            continue
        train_x, test_x, _ = prepare_features(
            matrix[train], matrix[test], "held-out", standardisation=rule
        )
        animals = sorted(set(labels[train]))
        one_hot = (labels[train][:, None] == np.asarray(animals)[None, :]).astype(np.float64)
        scores = ridge_projection(train_x, test_x, RIDGE_LAMBDA) @ one_hot
        predicted = np.asarray(animals)[np.argmax(scores, axis=1)]
        correct += int(np.sum(predicted == labels[test]))
        scored += int(test.sum())
    return {
        "accuracy": correct / scored if scored else None,
        "clips_scored": scored,
        "animals_scored": len(scorable_animals),
        "animals": len(sessions_per_animal),
        "animals_without_complete_session_labels": len(incomplete),
        "clips_with_a_named_session": sum(s is not None for s in sessions),
        "folds": folds,
    }


def choose(curve: dict[str, dict[str, Any]]) -> str | None:
    """The candidate with the highest held-out accuracy; ties to the Fisher ratio, then name."""

    if len(curve) == 1:
        # One candidate (an embedding with no layers to choose from) is the
        # choice under either rule, measured or not.
        return next(iter(curve))
    usable = {
        name: entry
        for name, entry in curve.items()
        if (entry.get("enrolment_held_out") or {}).get("accuracy") is not None
    }
    if not usable:
        return None
    return min(
        usable,
        key=lambda name: (
            -float(usable[name]["enrolment_held_out"]["accuracy"]),
            -float(usable[name]["enrollment_fisher_ratio"]),
            name,
        ),
    )
