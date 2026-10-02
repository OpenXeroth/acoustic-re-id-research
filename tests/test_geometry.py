"""Whether identity and the recording occupy the same directions.

Each test states the mutation it catches, and each fails when the behaviour it
names is removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from xinyenyana.geometry import (
    between_class_directions,
    crossing,
    run_geometry,
    subspace_overlap,
)


def planted(
    *, identities: int, sessions: int, per_cell: int, shared_axis: float, width: int = 32
) -> dict[str, object]:
    """Clips whose vectors carry an identity part and a session part in known
    amounts, with ``shared_axis`` deciding how far the session is built on the
    same directions as identity. At 0 the two are orthogonal by construction; at
    1 the session moves along the identity directions and nothing else."""

    rng = np.random.default_rng(17)
    identity_axes = rng.normal(size=(identities, width))
    # A session that IS an identity axis would make the pair (identity, session)
    # interchangeable, and no probe could recover one from the other however the
    # code behaved. A shared session instead lies somewhere in the span of the
    # identity directions without being any one of them, which is the situation
    # a real recording effect would produce.
    inside = rng.normal(size=(sessions, identities)) @ identity_axes
    basis = np.linalg.qr(identity_axes.T)[0]
    outside = rng.normal(size=(sessions, width))
    outside = outside - (outside @ basis) @ basis.T
    session_axes = shared_axis * inside + (1 - shared_axis) * outside
    rows, who, when = [], [], []
    for identity in range(identities):
        for session in range(sessions):
            for _ in range(per_cell):
                rows.append(
                    identity_axes[identity] + session_axes[session] + rng.normal(0, 0.05, width)
                )
                who.append(f"i{identity}")
                when.append(f"s{session}")
    return {"features": np.asarray(rows), "identities": who, "sessions": when}


def test_a_dataset_giving_each_animal_its_own_recordings_is_refused() -> None:
    """The mutation this catches: returning an overlap on a split where the two
    are nested. The Great Tit has 102 recordings for 16 birds and no recording
    holds two birds, so any number computed there would describe the nesting."""

    structure = crossing(["a", "a", "b", "b"], ["r1", "r2", "r3", "r4"])
    assert structure["admissible"] is False
    assert "cannot be separated" in structure["reason"]

    report = run_geometry(
        features=np.random.default_rng(1).normal(size=(4, 8)),
        identities=["a", "a", "b", "b"],
        sessions=["r1", "r2", "r3", "r4"],
        representation="fixture",
        shuffles=5,
    )
    assert report["measured"] is False
    assert "overlap" not in report


def test_a_crossed_split_is_admitted() -> None:
    structure = crossing(["a", "b", "a", "b"], ["r1", "r1", "r2", "r2"])
    assert structure["admissible"] is True
    assert structure["sessions_holding_more_than_one_identity"] == 2
    assert structure["identities_appearing_in_more_than_one_session"] == 2


def test_orthogonal_signals_read_as_orthogonal_and_shared_ones_do_not() -> None:
    """The measurement has to move with the thing it claims to measure. Two
    corpora differing only in whether the session was built on the identity
    directions must not produce the same overlap."""

    apart = planted(identities=5, sessions=5, per_cell=4, shared_axis=0.0)
    together = planted(identities=5, sessions=5, per_cell=4, shared_axis=1.0)
    low = run_geometry(**apart, representation="fixture", shuffles=99)
    high = run_geometry(**together, representation="fixture", shuffles=99)
    assert high["overlap"]["mean_cosine"] > low["overlap"]["mean_cosine"] + 0.2
    assert high["overlap"]["mean_cosine"] > 0.7


def test_the_unaligned_reference_brackets_the_overlap_in_both_directions() -> None:
    """A mean cosine of 0.6 means nothing on its own. A planted shared session
    has to sit above what no alignment looks like and a planted orthogonal one
    below it, or the reference is decoration."""

    together = run_geometry(
        **planted(identities=5, sessions=5, per_cell=4, shared_axis=1.0),
        representation="fixture",
        shuffles=199,
    )["overlap"]
    apart = run_geometry(
        **planted(identities=5, sessions=5, per_cell=4, shared_axis=0.0),
        representation="fixture",
        shuffles=199,
    )["overlap"]
    assert together["mean_cosine"] > together["unaligned_reference_95"][1]
    assert apart["mean_cosine"] < apart["unaligned_reference_95"][0]


def test_the_label_shuffle_that_was_removed_would_have_been_higher_than_the_truth() -> None:
    """The reference this module rejected, kept as a test so the reason survives
    the next person who thinks of it. Permuting the session labels and rebuilding
    the subspace gives a higher overlap than a session planted orthogonal to
    identity, because the means of arbitrary groups of clips vary along whichever
    directions carry the most variance, and those are the identity directions."""

    case = planted(identities=5, sessions=5, per_cell=4, shared_axis=0.0)
    unit = np.asarray(case["features"])
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    identity_space = between_class_directions(unit, case["identities"], 4)
    truth = subspace_overlap(identity_space, between_class_directions(unit, case["sessions"], 4))[
        "mean_cosine"
    ]
    rng = np.random.default_rng(3)
    shuffled = [
        subspace_overlap(
            identity_space,
            between_class_directions(unit, rng.permutation(case["sessions"]).tolist(), 4),
        )["mean_cosine"]
        for _ in range(25)
    ]
    assert float(np.median(shuffled)) > truth + 0.3


def test_the_session_is_readable_from_the_identity_directions_when_it_is_planted_there() -> None:
    """The sharpest claim this module makes. It has to fail when the session is
    not in those directions, or it is not evidence of anything."""

    together = run_geometry(
        **planted(identities=5, sessions=5, per_cell=4, shared_axis=1.0),
        representation="fixture",
        shuffles=19,
    )
    apart = run_geometry(
        **planted(identities=5, sessions=5, per_cell=4, shared_axis=0.0),
        representation="fixture",
        shuffles=19,
    )
    shared = together["session_read_from_identity_directions"]
    separate = apart["session_read_from_identity_directions"]
    assert shared["accuracy"] > shared["majority_class_rate"]
    assert shared["accuracy"] > separate["accuracy"]


def test_the_probe_follows_the_rule_the_classifier_uses() -> None:
    """Nothing in this module may come from a preparation the rest of the
    project does not apply. An earlier version of it standardised each
    coordinate across clips instead, justified by a fixture that turned out to
    have made identity and session mathematically interchangeable; with that
    fixture corrected the two rules agree, and the departure was removed."""

    from xinyenyana.evaluation import PER_CLIP_L2, PER_DIMENSION, leave_one_identity_out

    case = planted(identities=5, sessions=5, per_cell=4, shared_axis=1.0)
    unit = np.asarray(case["features"])
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    coordinates = unit @ between_class_directions(unit, case["identities"], 4)
    scored = {
        rule: leave_one_identity_out(
            features=coordinates,
            identities=np.asarray(case["identities"], dtype=object),
            targets={"session": np.asarray(case["sessions"], dtype=object)},
            categorical=frozenset({"session"}),
            standardisation=rule,
            ridge_lambda=1.0,
        )["session"]["accuracy"]
        for rule in (PER_DIMENSION, PER_CLIP_L2)
    }
    assert scored[PER_CLIP_L2] > 0.5
    assert abs(scored[PER_CLIP_L2] - scored[PER_DIMENSION]) < 0.2

    report = run_geometry(**case, representation="fixture", shuffles=9)
    assert report["standardisation"] == PER_CLIP_L2


def test_the_rank_cap_is_reported_rather_than_read_as_a_property_of_the_encoder() -> None:
    """Six animals cap the identity subspace at five directions whatever the
    representation holds. The mutation this catches is reporting that five as a
    measurement."""

    report = run_geometry(
        **planted(identities=6, sessions=20, per_cell=2, shared_axis=0.5),
        representation="fixture",
        shuffles=9,
    )
    assert report["identity_rank_cap"] == 5
    assert report["session_rank_cap"] == 19
    assert report["subspace_dimensions"] == 5
    assert "rank at most C - 1" in report["rank_note"]


def test_the_two_subspaces_are_built_by_the_same_function() -> None:
    """If identity were whitened against one nuisance and session against
    another, the angle between them would describe the whitening."""

    case = planted(identities=4, sessions=4, per_cell=3, shared_axis=0.3)
    first = between_class_directions(case["features"], case["identities"], 3)
    second = between_class_directions(case["features"], case["sessions"], 3)
    assert first.shape == second.shape
    assert subspace_overlap(first, first)["mean_cosine"] == pytest.approx(1.0, abs=1e-9)
