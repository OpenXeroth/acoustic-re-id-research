"""The stratified trials must give opposite answers on opposite data.

A measure that claims to tell "reading the animal" from "reading the
recording" is only worth running if it returns the first on data where identity
is the signal and the second on data where the recording is. Both cases are
constructed here, so the measure can fail.
"""

from __future__ import annotations

import numpy as np

from xinyenyana.verification_trials import (
    OTHER_INDIVIDUAL_SAME_SESSION,
    SAME_INDIVIDUAL_OTHER_SESSION,
    stratified_trials,
)

IDENTITIES = [f"bird{i}" for i in range(4)]
SESSIONS = [f"rec{i}" for i in range(4)]
CLIPS_PER_CELL = 8
WIDTH = 32


def _grid() -> tuple[list[str], list[str]]:
    identities: list[str] = []
    sessions: list[str] = []
    for identity in IDENTITIES:
        for session in SESSIONS:
            identities.extend([identity] * CLIPS_PER_CELL)
            sessions.extend([session] * CLIPS_PER_CELL)
    return identities, sessions


def _vectors(identities, sessions, *, carrier: str, seed: int) -> np.ndarray:
    """Vectors whose direction is set by the identity or by the session."""

    rng = np.random.default_rng(seed)
    centres = {
        value: rng.normal(size=WIDTH)
        for value in sorted(set(identities if carrier == "identity" else sessions))
    }
    keys = identities if carrier == "identity" else sessions
    return np.stack([centres[key] + 0.05 * rng.normal(size=WIDTH) for key in keys])


def test_identity_carrying_data_reads_the_animal() -> None:
    identities, sessions = _grid()
    result = stratified_trials(
        vectors=_vectors(identities, sessions, carrier="identity", seed=1),
        identities=identities,
        sessions=sessions,
    )
    verdict = result["animal_against_recording"]
    assert verdict["reads"] == "the animal"
    assert verdict["difference"] > 0.5
    assert verdict["equal_error_rate"]["equal_error_rate"] < 0.05


def test_session_carrying_data_reads_the_recording() -> None:
    identities, sessions = _grid()
    result = stratified_trials(
        vectors=_vectors(identities, sessions, carrier="session", seed=2),
        identities=identities,
        sessions=sessions,
    )
    verdict = result["animal_against_recording"]
    assert verdict["reads"] == "the recording"
    assert verdict["difference"] < -0.5


def test_one_individual_per_recording_is_reported_as_not_constructible() -> None:
    identities = [f"bird{i}" for i in range(4) for _ in range(CLIPS_PER_CELL)]
    sessions = [f"rec{i}" for i in range(4) for _ in range(CLIPS_PER_CELL)]
    rng = np.random.default_rng(3)
    result = stratified_trials(
        vectors=rng.normal(size=(len(identities), WIDTH)),
        identities=identities,
        sessions=sessions,
    )
    verdict = result["animal_against_recording"]
    assert verdict["constructible"] is False
    assert result["types"][OTHER_INDIVIDUAL_SAME_SESSION]["pairs"] == 0
    assert result["types"][SAME_INDIVIDUAL_OTHER_SESSION]["pairs"] == 0


def test_blocked_scoring_matches_scoring_every_pair_at_once() -> None:
    import itertools

    import numpy as np

    from xinyenyana.verification_trials import stratified_trials

    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(90, 16))
    identities = [f"bird{i % 5}" for i in range(90)]
    sessions = [f"rec{i % 7}" for i in range(90)]
    result = stratified_trials(
        vectors=vectors, identities=identities, sessions=sessions, maximum_pairs_per_type=250
    )

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = np.array(list(itertools.combinations(range(90), 2)), dtype=np.int64)
    pairs = pairs[np.random.default_rng(19).choice(len(pairs), size=1000, replace=False)]
    scores = np.einsum("ij,ij->i", unit[pairs[:, 0]], unit[pairs[:, 1]])
    ids = np.asarray(identities)
    same = ids[pairs[:, 0]] == ids[pairs[:, 1]]
    assert result["pairs_scored"] == 1000
    assert result["ignoring_the_recording"]["same_individual"]["mean"] == round(
        float(np.mean(scores[same])), 4
    )
