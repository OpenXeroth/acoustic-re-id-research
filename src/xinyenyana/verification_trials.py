"""Same-or-different pairs, stratified by whether the session is shared.

An accuracy says how often the right name came out. It does not say what the
representation was reading. Recasting the evaluation as verification does, and
it needs no permutation.

Every pair of scoring clips falls into one of four types:

1. same individual, same recording
2. same individual, different recording
3. different individuals, same recording
4. different individuals, different recording

A representation that reads the animal scores type 2 high and type 3 low. A
representation that reads the recording does the opposite: it scores type 3,
two different animals heard through one microphone at one moment, above type 2,
one animal heard through two. **The comparison between types 2 and 3 is the
measurement**, and its sign alone answers the question the whole project is
about.

Types 1 and 3 exist only where a recording holds more than one individual. On
an endpoint where each animal was recorded alone, type 3 is empty and this
returns that fact rather than a number.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: The four types, named so a reader does not have to remember a numbering.
SAME_INDIVIDUAL_SAME_SESSION = "same individual, same recording"
SAME_INDIVIDUAL_OTHER_SESSION = "same individual, different recording"
OTHER_INDIVIDUAL_SAME_SESSION = "different individuals, same recording"
OTHER_INDIVIDUAL_OTHER_SESSION = "different individuals, different recording"

TRIAL_TYPES = (
    SAME_INDIVIDUAL_SAME_SESSION,
    SAME_INDIVIDUAL_OTHER_SESSION,
    OTHER_INDIVIDUAL_SAME_SESSION,
    OTHER_INDIVIDUAL_OTHER_SESSION,
)


def _equal_error_rate(positive: Any, negative: Any) -> dict[str, float] | None:
    """Where the false-accept and false-reject rates cross."""

    import numpy as np

    if not len(positive) or not len(negative):
        return None
    # One sort rather than a sweep over every distinct score. Sweeping is
    # quadratic, and the rook produces of the order of a million pairs, which
    # is the difference between a second and a day.
    scores = np.concatenate([positive, negative])
    is_positive = np.concatenate(
        [np.ones(len(positive), dtype=bool), np.zeros(len(negative), dtype=bool)]
    )
    order = np.argsort(-scores, kind="stable")
    scores, is_positive = scores[order], is_positive[order]
    accepted_positive = np.cumsum(is_positive)
    accepted_negative = np.cumsum(~is_positive)
    false_accept = accepted_negative / len(negative)
    false_reject = (len(positive) - accepted_positive) / len(positive)
    crossing = int(np.argmin(np.abs(false_accept - false_reject)))
    return {
        "equal_error_rate": round(float((false_accept[crossing] + false_reject[crossing]) / 2), 4),
        "threshold": round(float(scores[crossing]), 6),
    }


def _summary(scores: Any) -> dict[str, Any]:
    import numpy as np

    if not len(scores):
        return {"pairs": 0}
    return {
        "pairs": int(len(scores)),
        "mean": round(float(np.mean(scores)), 4),
        "median": round(float(np.median(scores)), 4),
        "percentile_5": round(float(np.quantile(scores, 0.05)), 4),
        "percentile_95": round(float(np.quantile(scores, 0.95)), 4),
    }


def stratified_trials(
    *,
    vectors: Any,
    identities: Sequence[str],
    sessions: Sequence[str],
    maximum_pairs_per_type: int = 200_000,
    seed: int = 19,
) -> dict[str, Any]:
    """Score every scoring-side pair, bucketed by the four types.

    The score is cosine similarity on the L2-normalised representation, which is
    the same quantity the nearest-centroid head compares.
    """

    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    array = array / np.maximum(norms, 1e-12)
    identity_array = np.asarray([str(value) for value in identities])
    session_array = np.asarray([str(value) for value in sessions])

    rng = np.random.default_rng(seed)
    total = len(array)
    # The same pairs, in the same order, as itertools.combinations, without a
    # Python tuple per pair; and scored in blocks, because gathering both sides
    # of 800,000 pairs of 1,024-number vectors at once needs 13 GB.
    rows, columns = np.triu_indices(total, k=1)
    pairs = np.stack([rows, columns], axis=1).astype(np.int64)
    if len(pairs) > maximum_pairs_per_type * 4:
        pairs = pairs[rng.choice(len(pairs), size=maximum_pairs_per_type * 4, replace=False)]
    left, right = pairs[:, 0], pairs[:, 1]
    scores = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), 20_000):
        stop = start + 20_000
        scores[start:stop] = np.einsum(
            "ij,ij->i", array[left[start:stop]], array[right[start:stop]]
        )
    same_individual = identity_array[left] == identity_array[right]
    same_session = session_array[left] == session_array[right]

    buckets = {
        SAME_INDIVIDUAL_SAME_SESSION: same_individual & same_session,
        SAME_INDIVIDUAL_OTHER_SESSION: same_individual & ~same_session,
        OTHER_INDIVIDUAL_SAME_SESSION: ~same_individual & same_session,
        OTHER_INDIVIDUAL_OTHER_SESSION: ~same_individual & ~same_session,
    }
    result: dict[str, Any] = {
        "clips": int(total),
        "pairs_scored": int(len(pairs)),
        "score": "cosine similarity of the L2-normalised representation",
        "types": {name: _summary(scores[mask]) for name, mask in buckets.items()},
    }

    # The decisive comparison. Both are "different recording or different
    # animal, but not both", so neither is helped by the trivial cue of being
    # the same moment.
    animal = scores[buckets[SAME_INDIVIDUAL_OTHER_SESSION]]
    recording = scores[buckets[OTHER_INDIVIDUAL_SAME_SESSION]]
    if len(animal) and len(recording):
        difference = float(np.mean(animal) - np.mean(recording))
        pooled = float(np.sqrt((np.var(animal) + np.var(recording)) / 2)) or 1e-12
        result["animal_against_recording"] = {
            "same_individual_different_recording_mean": round(float(np.mean(animal)), 4),
            "different_individuals_same_recording_mean": round(float(np.mean(recording)), 4),
            "difference": round(difference, 4),
            "standardised_difference": round(difference / pooled, 4),
            "reads": ("the animal" if difference > 0 else "the recording"),
            "equal_error_rate": _equal_error_rate(animal, recording),
        }
    else:
        result["animal_against_recording"] = {
            "constructible": False,
            "reason": (
                "no recording in the scoring split holds two or more individuals, "
                "so the different-individuals-same-recording type is empty"
            ),
        }

    # The conventional reading, for comparison with the speaker-verification
    # literature: same individual against different individuals, ignoring the
    # recording entirely. This is the number that looks good when the recording
    # is doing the work, which is why it is reported second.
    positive = scores[same_individual]
    negative = scores[~same_individual]
    result["ignoring_the_recording"] = {
        "same_individual": _summary(positive),
        "different_individuals": _summary(negative),
        "equal_error_rate": _equal_error_rate(positive, negative),
    }
    return result
