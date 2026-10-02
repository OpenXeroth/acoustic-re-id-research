"""Tests for E-REID-01's session compensations.

Each treatment is checked against a property that would fail if it were not doing
what its name says, on synthetic data where the session effect is planted and
therefore known.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xinyenyana.compensation import (
    MINIMUM_SESSIONS,
    NONE,
    NUISANCE_PROJECTION,
    PER_SESSION_BATCH,
    PER_SESSION_STREAMING,
    PROJECTION_DIMENSIONS,
    TREATMENTS,
    WCCN,
    WCCN_SHRINKAGE,
    applicable,
    compensate,
    majority_class_rate,
    parameters_for,
)


def _planted(sessions: int = 6, identities: int = 4, per: int = 12) -> dict[str, Any]:
    """Identity in one direction, session in another, orthogonal by construction.

    A compensation that removes the session direction should leave the identity
    direction alone. Nothing here is fitted, so the answer is known in advance.
    """

    rng = np.random.default_rng(3)
    width = 16
    identity_axis = np.eye(width)[0]
    session_axis = np.eye(width)[1]
    rows, names, days, clock = [], [], [], []
    tick = 0.0
    for day in range(sessions):
        for index in range(identities):
            for _ in range(per):
                vector = (
                    3.0 * (index + 1) * identity_axis
                    + 9.0 * (day + 1) * session_axis
                    + rng.normal(0.0, 0.1, size=width)
                )
                rows.append(vector)
                names.append(f"bird{index}")
                days.append(f"2013-01-{day + 1:02d}")
                clock.append(tick)
                tick += 1.0
    return {
        "vectors": np.asarray(rows),
        "identities": names,
        "sessions": days,
        "order": clock,
        "enrollment": list(range(len(rows))),
    }


def _session_energy(vectors: Any, sessions: list[str]) -> float:
    """How much of the spread is between sessions rather than within them."""

    overall = vectors.mean(axis=0)
    means = np.vstack(
        [
            vectors[[i for i, s in enumerate(sessions) if s == name]].mean(axis=0)
            for name in sorted(set(sessions))
        ]
    )
    return float(np.sum((means - overall) ** 2))


def test_every_treatment_is_named_and_has_a_registered_grid() -> None:
    assert len(TREATMENTS) == 5
    assert parameters_for(WCCN) == WCCN_SHRINKAGE
    assert parameters_for(NUISANCE_PROJECTION) == PROJECTION_DIMENSIONS
    assert parameters_for(NONE) == (None,)


def test_doing_nothing_does_nothing() -> None:
    case = _planted()
    out = compensate(
        NONE,
        vectors=case["vectors"],
        enrollment_rows=case["enrollment"],
        identities=case["identities"],
        sessions=case["sessions"],
        order=case["order"],
        parameter=None,
    )
    assert np.allclose(out, case["vectors"])


def test_per_session_centring_removes_the_session_and_keeps_the_identity() -> None:
    case = _planted()
    before = _session_energy(case["vectors"], case["sessions"])
    out = compensate(
        PER_SESSION_BATCH,
        vectors=case["vectors"],
        enrollment_rows=case["enrollment"],
        identities=case["identities"],
        sessions=case["sessions"],
        order=case["order"],
        parameter=None,
    )
    after = _session_energy(out, case["sessions"])
    assert after < before * 1e-6, (before, after)
    # the identity direction survives: birds are still ordered along it
    means = [
        out[[i for i, n in enumerate(case["identities"]) if n == name]].mean(axis=0)[0]
        for name in sorted(set(case["identities"]))
    ]
    assert means == sorted(means)
    assert means[-1] - means[0] > 5.0


def test_the_streaming_variant_is_not_the_batch_one() -> None:
    """If they agreed, the deployable question the pair exists to ask is empty."""

    case = _planted()
    shared = {
        "vectors": case["vectors"],
        "enrollment_rows": case["enrollment"],
        "identities": case["identities"],
        "sessions": case["sessions"],
        "order": case["order"],
        "parameter": None,
    }
    batch = compensate(PER_SESSION_BATCH, **shared)
    streaming = compensate(PER_SESSION_STREAMING, **shared)
    assert not np.allclose(batch, streaming)
    # and the streaming one still removes most of the session
    assert _session_energy(streaming, case["sessions"]) < _session_energy(
        case["vectors"], case["sessions"]
    )


def test_the_streaming_variant_never_uses_a_later_query_clip() -> None:
    """The property that makes it deployable, checked rather than asserted.

    A live system answers a clip knowing the enrolment set and the clips of that
    session it has already seen, and nothing else. So changing a later query clip
    must leave every earlier query clip untouched.

    The first version of this test disturbed the last row while treating every
    row as enrolment, which failed for a reason that is not a leak: the warm-up
    for a session's first few clips subtracts the enrolment global mean, and
    moving an enrolment clip moves that mean. That is a transform fitted on
    enrolment, which is allowed. The property below is the one that matters.
    """

    case = _planted()
    half = len(case["vectors"]) // 2
    shared = {
        "enrollment_rows": list(range(half)),
        "identities": case["identities"],
        "sessions": case["sessions"],
        "order": case["order"],
        "parameter": None,
    }
    first = compensate(PER_SESSION_STREAMING, vectors=case["vectors"], **shared)
    disturbed = case["vectors"].copy()
    disturbed[-1] += 50.0
    second = compensate(PER_SESSION_STREAMING, vectors=disturbed, **shared)
    assert np.allclose(first[:-1], second[:-1])


def test_the_subspace_removal_removes_the_session_direction() -> None:
    case = _planted()
    before = _session_energy(case["vectors"], case["sessions"])
    out = compensate(
        NUISANCE_PROJECTION,
        vectors=case["vectors"],
        enrollment_rows=case["enrollment"],
        identities=case["identities"],
        sessions=case["sessions"],
        order=case["order"],
        parameter=1,
    )
    assert _session_energy(out, case["sessions"]) < before * 0.01
    # zero directions removed is the baseline, built into the sweep
    none_removed = compensate(
        NUISANCE_PROJECTION,
        vectors=case["vectors"],
        enrollment_rows=case["enrollment"],
        identities=case["identities"],
        sessions=case["sessions"],
        order=case["order"],
        parameter=0,
    )
    assert np.allclose(none_removed, case["vectors"])


def test_full_shrinkage_leaves_the_geometry_alone() -> None:
    """At shrinkage 1 the whitening is a scalar, so the sweep contains its own control."""

    case = _planted()
    out = compensate(
        WCCN,
        vectors=case["vectors"],
        enrollment_rows=case["enrollment"],
        identities=case["identities"],
        sessions=case["sessions"],
        order=case["order"],
        parameter=1.0,
    )
    scale = out[0, 0] / case["vectors"][0, 0]
    assert np.allclose(out, case["vectors"] * scale)


def test_within_class_whitening_shrinks_the_within_identity_spread() -> None:
    case = _planted()

    def within(vectors: Any) -> float:
        blocks = [
            vectors[[i for i, n in enumerate(case["identities"]) if n == name]]
            for name in sorted(set(case["identities"]))
        ]
        return float(np.mean([np.var(b - b.mean(axis=0), axis=0).sum() for b in blocks]))

    out = compensate(
        WCCN,
        vectors=case["vectors"],
        enrollment_rows=case["enrollment"],
        identities=case["identities"],
        sessions=case["sessions"],
        order=case["order"],
        parameter=0.0,
    )
    assert within(out) < within(case["vectors"])


def test_a_treatment_that_needs_sessions_says_so_instead_of_failing_quietly() -> None:
    case = _planted(sessions=1)
    for treatment in (PER_SESSION_BATCH, PER_SESSION_STREAMING, NUISANCE_PROJECTION):
        reason = applicable(
            treatment, sessions=case["sessions"], enrollment_rows=case["enrollment"]
        )
        assert reason is not None and str(MINIMUM_SESSIONS) in reason
    assert applicable(WCCN, sessions=None, enrollment_rows=case["enrollment"]) is None
    assert applicable(NONE, sessions=None, enrollment_rows=case["enrollment"]) is None
    assert (
        applicable(PER_SESSION_BATCH, sessions=None, enrollment_rows=case["enrollment"])
        == "the endpoint carries no session label"
    )


def test_an_unknown_treatment_is_refused() -> None:
    case = _planted()
    with pytest.raises(ValueError, match="unknown treatment"):
        compensate(
            "magic",
            vectors=case["vectors"],
            enrollment_rows=case["enrollment"],
            identities=case["identities"],
            sessions=case["sessions"],
            order=case["order"],
            parameter=None,
        )


def test_the_trivial_floor_is_the_commonest_identity_not_one_over_n() -> None:
    identities = ["a"] * 7 + ["b"] * 2 + ["c"]
    assert majority_class_rate(identities, list(range(10))) == pytest.approx(0.7)
    assert majority_class_rate(identities, [7, 8, 9]) == pytest.approx(2 / 3)


def test_a_compensation_is_fitted_on_enrolment_alone() -> None:
    """Query clips must not reach a fitted transform.

    Moving every query clip a long way must leave the enrolment rows of a fitted
    treatment exactly where they were.
    """

    case = _planted()
    enrolment = list(range(0, len(case["vectors"]) // 2))
    shared = {
        "enrollment_rows": enrolment,
        "identities": case["identities"],
        "sessions": case["sessions"],
        "order": case["order"],
    }
    disturbed = case["vectors"].copy()
    disturbed[len(enrolment) :] += 100.0
    for treatment, parameter in ((WCCN, 0.1), (NUISANCE_PROJECTION, 2)):
        before = compensate(treatment, vectors=case["vectors"], parameter=parameter, **shared)
        after = compensate(treatment, vectors=disturbed, parameter=parameter, **shared)
        assert np.allclose(before[enrolment], after[enrolment]), treatment


# --- what the first run got wrong ------------------------------------------


def _two_condition_endpoint(tmp_path: Any) -> Any:
    """An endpoint carrying a background recording beside every call.

    The background clips carry the bird's identity and are not the bird. A
    treatment fitted over both estimates the difference between a call and its
    silence as if it were within-identity variation.
    """

    from xinyenyana.a2 import ClipRecord, Endpoint

    records = []
    for index in range(16):
        identity = f"bird{index % 4}"
        split = "enrollment" if index < 8 else "query"
        for condition in ("foreground", "background"):
            records.append(
                ClipRecord(
                    filename=f"{identity}-{index}-{condition}",
                    path=tmp_path / f"{identity}-{index}-{condition}.wav",
                    identity=identity,
                    split=split,
                    context={"condition": condition, "recorded": f"2013-01-{index % 5 + 1:02d}"},
                )
            )
    return Endpoint(
        name="two-condition",
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="docs/measured.md",
    )


def test_only_the_clips_the_head_scores_are_fitted_on(tmp_path: Any) -> None:
    """The defect the first run of E-REID-01 carried.

    Every treatment was fitted over foreground and background together, because
    an endpoint that publishes a background recording beside every call holds
    both in one record set. Half the fitting data was ambient sound.
    """

    from xinyenyana.compensation import run_compensation

    endpoint = _two_condition_endpoint(tmp_path)
    rng = np.random.default_rng(2)
    vectors = rng.normal(0.0, 1.0, size=(len(endpoint.records), 8))
    summary = run_compensation(
        endpoint=endpoint,
        vectors=vectors,
        representation="synthetic",
        standardisation="per-clip L2",
    )
    # 8 enrolment indices and 8 query indices, each with a foreground and a
    # background clip: 32 records, of which the head scores 8 and 8.
    assert summary["clips"] == 32
    assert summary["enrollment_clips"] == 8
    assert summary["query_clips"] == 8


def test_the_recording_day_probe_is_reported(tmp_path: Any) -> None:
    """Prediction 3 needs a number, and the first run produced none."""

    from xinyenyana.compensation import run_compensation

    endpoint = _two_condition_endpoint(tmp_path)
    rng = np.random.default_rng(4)
    vectors = rng.normal(0.0, 1.0, size=(len(endpoint.records), 8))
    summary = run_compensation(
        endpoint=endpoint,
        vectors=vectors,
        representation="synthetic",
        standardisation="per-clip L2",
    )
    for rows in summary["treatments"].values():
        for row in rows:
            assert row["recording_day_probe"] is not None
            assert "accuracy" in row["recording_day_probe"]
            assert "majority_class_rate" in row["recording_day_probe"]


def _stowell_shaped_endpoint(*, identity_in_background: bool) -> tuple[Any, Any]:
    """An endpoint with a background recording beside every call, and its vectors.

    The identity signature is planted in the foreground clips always and in the
    background clips only when asked for, so the background arm has a known
    answer both ways.
    """

    from pathlib import Path

    from xinyenyana.a2 import ClipRecord, Endpoint

    identities = ("alpha", "beta", "gamma")
    width = 8
    records: list[ClipRecord] = []
    for identity in identities:
        for condition in ("foreground", "background"):
            for split, count in (("enrollment", 20), ("query", 20)):
                for index in range(count):
                    name = f"{identity}-{condition}-{split}-{index}.wav"
                    records.append(
                        ClipRecord(
                            filename=name,
                            path=Path(name),
                            identity=identity,
                            split=split,
                            context={"condition": condition},
                        )
                    )
    endpoint = Endpoint(
        name="synthetic-with-backgrounds",
        records=tuple(records),
        categorical_targets=("identity",),
        manifest_sha256="0" * 64,
        source_document="a synthetic endpoint written by this test",
    )
    generator = np.random.default_rng(11)
    signatures = {identity: generator.normal(size=width) * 4.0 for identity in identities}
    rows = []
    for record in records:
        noise = generator.normal(size=width) * 0.1
        carries = record.context["condition"] == "foreground" or identity_in_background
        rows.append(signatures[record.identity] + noise if carries else noise)
    return endpoint, np.asarray(rows, dtype=np.float64)


def _compensation(*, identity_in_background: bool) -> dict[str, Any]:
    from xinyenyana.compensation import run_compensation
    from xinyenyana.evaluation import PER_CLIP_L2

    endpoint, vectors = _stowell_shaped_endpoint(identity_in_background=identity_in_background)
    return run_compensation(
        endpoint=endpoint,
        vectors=vectors,
        representation="synthetic",
        standardisation=PER_CLIP_L2,
    )


def test_an_endpoint_with_backgrounds_carries_a_background_only_arm() -> None:
    summary = _compensation(identity_in_background=False)
    assert summary["background_only_arm"] is True
    for rows in summary["treatments"].values():
        for row in rows:
            assert row["background_only"] is not None
            assert row["background_only"]["enrollment_calls"] == 60
            assert row["background_only"]["query_calls"] == 60


def test_the_background_arm_finds_the_place_only_when_the_place_carries_identity() -> None:
    absent = _compensation(identity_in_background=False)["treatments"][NONE]
    present = _compensation(identity_in_background=True)["treatments"][NONE]
    chance = 1.0 / 3
    assert all(row["background_only"]["accuracy"] < chance + 0.15 for row in absent)
    assert all(row["background_only"]["accuracy"] > 0.9 for row in present)


def test_an_endpoint_without_backgrounds_has_no_background_arm() -> None:
    from pathlib import Path

    from xinyenyana.a2 import ClipRecord, Endpoint
    from xinyenyana.compensation import run_compensation
    from xinyenyana.evaluation import PER_CLIP_L2

    records = tuple(
        ClipRecord(
            filename=f"{identity}-{split}-{index}.wav",
            path=Path(f"{identity}-{split}-{index}.wav"),
            identity=identity,
            split=split,
            context={},
        )
        for identity in ("alpha", "beta", "gamma")
        for split in ("enrollment", "query")
        for index in range(20)
    )
    endpoint = Endpoint(
        name="synthetic-no-backgrounds",
        records=records,
        categorical_targets=("identity",),
        manifest_sha256="0" * 64,
        source_document="a synthetic endpoint written by this test",
    )
    generator = np.random.default_rng(5)
    signatures = {name: generator.normal(size=8) * 4.0 for name in ("alpha", "beta", "gamma")}
    vectors = np.asarray(
        [signatures[record.identity] + generator.normal(size=8) * 0.1 for record in records]
    )
    summary = run_compensation(
        endpoint=endpoint,
        vectors=vectors,
        representation="synthetic",
        standardisation=PER_CLIP_L2,
    )
    assert summary["background_only_arm"] is False
    for rows in summary["treatments"].values():
        for row in rows:
            assert row["background_only"] is None


def _query_dated_endpoint(tmp_path: Any) -> Any:
    """The little penguin's across-night shape: dates on the query side only.

    The release names every extra-night clip with the date it was recorded and
    names no main-folder clip with one, so the enrolment side of this split
    carries no date at all.
    """

    from xinyenyana.a2 import ClipRecord, Endpoint

    records = []
    for index in range(24):
        identity = f"nest{index % 4}"
        enrolled = index < 12
        context: dict[str, Any] = {"condition": "foreground", "night": "first"}
        if not enrolled:
            context = {
                "condition": "foreground",
                "night": "later",
                "recorded": f"2025-02-{18 + index % 3:02d}",
            }
        records.append(
            ClipRecord(
                filename=f"{identity}-{index}",
                path=tmp_path / f"{identity}-{index}.wav",
                identity=identity,
                split="enrollment" if enrolled else "query",
                context=context,
            )
        )
    return Endpoint(
        name="penguin-acrossnight",
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="docs/measured.md",
    )


def test_the_day_probe_runs_when_only_the_query_side_carries_a_date(tmp_path: Any) -> None:
    """The mutation this catches: requiring every clip to carry a date before
    the probe fires. The probe reads the query clips alone, so that requirement
    skipped the little penguin across nights, which is one of the two endpoints
    the probe is registered to run on, and left no reading and no reason.
    """

    from xinyenyana.compensation import NONE, run_compensation

    endpoint = _query_dated_endpoint(tmp_path)
    rng = np.random.default_rng(11)
    vectors = rng.normal(0.0, 1.0, size=(len(endpoint.records), 8))
    summary = run_compensation(
        endpoint=endpoint,
        vectors=vectors,
        representation="synthetic",
        standardisation="per-clip L2",
    )
    assert summary["sessions"] is None
    rows = summary["treatments"][NONE]
    assert rows
    for row in rows:
        assert row["recording_day_probe"] is not None
        assert "accuracy" in row["recording_day_probe"]
        assert "majority_class_rate" in row["recording_day_probe"]
