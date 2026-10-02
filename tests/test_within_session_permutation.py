"""The within-session null must separate what the ordinary null cannot.

Shuffling enrolment labels across the whole pile breaks the clip-to-label link
and the clip-to-recording link together, so clearing it does not establish that
the animal was identified. Shuffling within a recording holds the recording
constant. The difference is only worth reporting if it is real, so both cases
are constructed here: data where the recording carries the label, and data
where the animal does.
"""

from __future__ import annotations

import numpy as np

from xinyenyana.evaluation import PER_CLIP_L2, evaluate_endpoint

WIDTH = 24
PERMUTATIONS = 199
SEED = 5


def _records(identities, sessions, splits):
    return [
        {
            "filename": f"clip{index:04d}.wav",
            "identity": identity,
            "split": split,
            "condition": "foreground",
            "session": session,
        }
        for index, (identity, session, split) in enumerate(
            zip(identities, sessions, splits, strict=True)
        )
    ]


def _design(*, carrier: str, seed: int):
    """Four birds in each of four recordings, per split, so every type exists."""

    identities, sessions, splits = [], [], []
    for split in ("enrollment", "query"):
        for session in range(4):
            for bird in range(4):
                for _ in range(6):
                    identities.append(f"bird{bird}")
                    sessions.append(f"{split}-rec{session}")
                    splits.append(split)
    rng = np.random.default_rng(seed)
    if carrier == "identity":
        centres = {value: rng.normal(size=WIDTH) for value in sorted(set(identities))}
        keys = identities
    else:
        centres = {value: rng.normal(size=WIDTH) for value in sorted(set(sessions))}
        keys = sessions
    vectors = np.stack([centres[key] + 0.05 * rng.normal(size=WIDTH) for key in keys])
    return _records(identities, sessions, splits), vectors


def _run(records, vectors):
    return evaluate_endpoint(
        records=records,
        vectors=vectors,
        representation="test-representation",
        enrollment_condition="foreground",
        query_condition="foreground",
        ridge_lambda=1.0,
        seed=SEED,
        permutations=PERMUTATIONS,
        bootstrap_replicates=200,
        standardisation=PER_CLIP_L2,
    )


def test_identity_carrying_data_clears_both_nulls() -> None:
    result = _run(*_design(carrier="identity", seed=11))
    assert result["classification"]["accuracy"] > 0.9
    assert result["permutation_control"]["p_value_plus_one"] <= 0.01
    within = result["within_session_permutation"]
    assert within["sessions_holding_two_or_more_identities"] == 4
    assert within["p_value_plus_one"] <= 0.01


def test_the_within_session_null_is_a_different_object_from_the_ordinary_one() -> None:
    """On recording-carried data the two nulls must not agree by construction."""

    result = _run(*_design(carrier="session", seed=12))
    within = result["within_session_permutation"]
    assert within["sessions_holding_two_or_more_identities"] == 4
    assert within["enrollment_clips_in_those_sessions"] == within["enrollment_clips"]
    # The within-session null is drawn from a different distribution: shuffling
    # inside a recording cannot manufacture the between-recording structure the
    # ordinary shuffle destroys.
    assert within["mean_accuracy"] != result["permutation_control"]["mean_accuracy"]


def test_one_individual_per_recording_reports_that_it_is_not_constructible() -> None:
    identities, sessions, splits = [], [], []
    for split in ("enrollment", "query"):
        for bird in range(4):
            for _ in range(6):
                identities.append(f"bird{bird}")
                sessions.append(f"{split}-bird{bird}")
                splits.append(split)
    rng = np.random.default_rng(13)
    vectors = rng.normal(size=(len(identities), WIDTH))
    result = _run(_records(identities, sessions, splits), vectors)
    within = result["within_session_permutation"]
    assert within["constructible"] is False
    assert "two or more individuals" in within["reason"]


def test_an_endpoint_with_no_session_field_says_so_rather_than_guessing() -> None:
    identities = [f"bird{b}" for b in range(4) for _ in range(6)] * 2
    splits = ["enrollment"] * 24 + ["query"] * 24
    records = [
        {
            "filename": f"clip{i:04d}.wav",
            "identity": identity,
            "split": split,
            "condition": "foreground",
        }
        for i, (identity, split) in enumerate(zip(identities, splits, strict=True))
    ]
    rng = np.random.default_rng(14)
    result = _run(records, rng.normal(size=(len(records), WIDTH)))
    assert result["within_session_permutation"]["constructible"] is False
    assert "no recording or session field" in result["within_session_permutation"]["reason"]


def test_the_permutation_upper_tail_is_above_uniform_chance() -> None:
    result = _run(*_design(carrier="identity", seed=15))
    tail = result["permutation_upper_tail"]
    assert tail["uniform_chance"] == 0.25
    assert tail["percentile_95"] >= tail["uniform_chance"]


def test_the_hierarchical_bootstrap_is_not_narrower_than_the_identity_one() -> None:
    result = _run(*_design(carrier="identity", seed=16))
    identity_interval = result["identity_block_bootstrap_accuracy_95"]
    hierarchical = result["hierarchical_bootstrap"]["accuracy_95"]
    assert hierarchical[1] - hierarchical[0] >= (identity_interval[1] - identity_interval[0]) - 1e-9
