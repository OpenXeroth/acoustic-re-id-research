"""The two session readings: what they measure and what they refuse to measure.

Each test here fails when the behaviour it names is removed, and the mutation
that would make each one pass for the wrong reason is stated in its docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from xinyenyana.evaluation import PER_CLIP_L2, PER_DIMENSION
from xinyenyana.nuisance import (
    MINIMUM_SAME_SESSION_PAIRS,
    PAIR_CAP,
    context_representation,
    session_similarity,
)


def corpus(
    *, identities: int, sessions: int, per_cell: int, session_shift: float, identity_shift: float
) -> dict[str, object]:
    """Clips whose vectors carry an identity part and a session part, in
    known amounts, so the statistic can be checked against what was planted."""

    rng = np.random.default_rng(17)
    rows, names, who, when = [], [], [], []
    identity_axes = rng.normal(size=(identities, 8))
    session_axes = rng.normal(size=(sessions, 8))
    for identity in range(identities):
        for session in range(sessions):
            for index in range(per_cell):
                rows.append(
                    identity_shift * identity_axes[identity]
                    + session_shift * session_axes[session]
                    + rng.normal(0, 0.1, 8)
                )
                names.append(f"i{identity}-s{session}-{index}")
                who.append(f"identity-{identity}")
                when.append(f"session-{session}")
    return {
        "features": np.asarray(rows),
        "identities": who,
        "sessions": when,
        "filenames": names,
    }


def test_a_planted_session_effect_is_found_and_its_interval_excludes_zero() -> None:
    result = session_similarity(
        **corpus(identities=6, sessions=6, per_cell=4, session_shift=1.0, identity_shift=0.2),
        standardisation=PER_CLIP_L2,
        replicates=200,
    )
    assert result["sufficient"] is True
    assert result["difference"] > 0.1
    assert result["difference_95"][0] > 0.0


def test_no_session_effect_leaves_an_interval_spanning_zero() -> None:
    """The mutation this catches: reporting the point difference without its
    interval. On clips whose only structure is identity, the difference is
    near zero and its interval has to admit that."""

    result = session_similarity(
        **corpus(identities=6, sessions=6, per_cell=4, session_shift=0.0, identity_shift=1.0),
        standardisation=PER_CLIP_L2,
        replicates=200,
    )
    assert result["sufficient"] is True
    assert abs(result["difference"]) < 0.1
    assert result["difference_95"][0] < 0.0 < result["difference_95"][1]


def test_every_pair_is_two_different_animals() -> None:
    """Identity is excluded by construction, so a corpus with a large identity
    effect and no session effect must not produce one. Counting same-identity
    pairs into the same-session arm would break this."""

    result = session_similarity(
        **corpus(identities=4, sessions=5, per_cell=5, session_shift=0.0, identity_shift=3.0),
        standardisation=PER_CLIP_L2,
        replicates=200,
    )
    assert abs(result["difference"]) < 0.1


def test_too_few_sessions_is_reported_with_its_counts_not_as_a_number() -> None:
    result = session_similarity(
        **corpus(identities=4, sessions=2, per_cell=4, session_shift=1.0, identity_shift=0.2),
        standardisation=PER_CLIP_L2,
        replicates=50,
    )
    assert result["sufficient"] is False
    assert "sessions" in str(result["insufficient_because"])
    assert "same_session_median" not in result


def test_too_few_same_session_pairs_is_reported_the_same_way() -> None:
    small = corpus(identities=2, sessions=5, per_cell=2, session_shift=1.0, identity_shift=0.2)
    result = session_similarity(**small, standardisation=PER_CLIP_L2, replicates=50)
    assert result["sufficient"] is False
    assert result["same_session_pairs"] < MINIMUM_SAME_SESSION_PAIRS


def test_the_interval_resamples_sessions_and_not_pairs() -> None:
    """One clip appears in many pairs, so an interval built by resampling pairs
    is too narrow. Cluster resampling has to give a wider interval than the
    naive one on the same data; if the two agree, the clustering is not firing."""

    case = corpus(identities=6, sessions=5, per_cell=5, session_shift=0.6, identity_shift=0.6)
    result = session_similarity(**case, standardisation=PER_CLIP_L2, replicates=400)
    clustered_width = result["difference_95"][1] - result["difference_95"][0]
    assert result["same_session_clusters"] == 5
    assert result["different_session_clusters"] == 10
    assert clustered_width > 0.0
    assert result["same_session_clusters"] < result["same_session_pairs"]


def test_the_pair_cap_is_the_registered_one_and_is_reported() -> None:
    result = session_similarity(
        **corpus(identities=5, sessions=5, per_cell=4, session_shift=0.5, identity_shift=0.5),
        standardisation=PER_CLIP_L2,
        replicates=50,
    )
    assert result["pair_cap"] == PAIR_CAP
    assert result["same_session_pairs_used"] == min(PAIR_CAP, result["same_session_pairs"])


def test_an_unknown_standardisation_is_refused() -> None:
    with pytest.raises(ValueError, match="standardisation"):
        session_similarity(
            **corpus(identities=4, sessions=4, per_cell=4, session_shift=0.5, identity_shift=0.5),
            standardisation="whatever",
            replicates=10,
        )


def great_tit_contexts(shared_year: bool) -> tuple[list[dict[str, object]], list[str]]:
    contexts, splits = [], []
    for split, year in (("enrollment", "2020"), ("query", "2020" if shared_year else "2021")):
        for nest in range(4):
            for _ in range(3):
                contexts.append({"year": year, "nest_x": float(nest), "nest_y": float(nest * 2)})
                splits.append(split)
    return contexts, splits


def test_the_context_baseline_fires_where_its_values_recur_across_the_split() -> None:
    contexts, splits = great_tit_contexts(shared_year=True)
    built = context_representation(endpoint_name="great-tit", contexts=contexts, splits=splits)
    assert built["applicable"] is True
    assert built["vectors"].shape[0] == len(contexts)
    assert "nest_x" in built["names"]


def test_a_categorical_field_with_no_shared_value_blocks_the_baseline() -> None:
    """The mutation this catches: one-hot encoding over every value rather than
    over the enrolment values, which would hide that a query year never
    enrolled and let the baseline report a floor score as though it had run."""

    contexts, splits = great_tit_contexts(shared_year=False)
    built = context_representation(endpoint_name="great-tit", contexts=contexts, splits=splits)
    assert built["applicable"] is False
    assert "year" in built["reason"]


def test_a_session_index_that_is_the_split_cannot_fire() -> None:
    contexts = [{"recording": "morning"} for _ in range(8)] + [
        {"recording": "afternoon"} for _ in range(8)
    ]
    splits = ["enrollment"] * 8 + ["query"] * 8
    built = context_representation(endpoint_name="rookid", contexts=contexts, splits=splits)
    assert built["applicable"] is False
    assert "no value in both splits" in built["reason"]


def test_a_constant_query_encoding_cannot_fire() -> None:
    contexts = [{"calendar_date_ordinal": float(index % 5)} for index in range(10)] + [
        {"calendar_date_ordinal": 2.0} for _ in range(10)
    ]
    splits = ["enrollment"] * 10 + ["query"] * 10
    built = context_representation(endpoint_name="zebra-finch", contexts=contexts, splits=splits)
    assert built["applicable"] is False
    assert "identically" in built["reason"]


def test_an_endpoint_publishing_nothing_says_so_rather_than_scoring_zero() -> None:
    built = context_representation(
        endpoint_name="chiffchaff-withinyear",
        contexts=[{} for _ in range(4)],
        splits=["enrollment", "enrollment", "query", "query"],
    )
    assert built["applicable"] is False
    assert "publishes no" in built["reason"]
    assert built["fields"] == []


def test_a_day_field_reads_a_timestamp_as_a_calendar_ordinal() -> None:
    contexts = [
        {"treatment": "16", "recorded": f"2013-01-{day:02d}T08:00:00"} for day in range(1, 9)
    ] + [{"treatment": "16", "recorded": f"2013-01-{day:02d}T09:00:00"} for day in range(1, 9)]
    splits = ["enrollment"] * 8 + ["query"] * 8
    built = context_representation(
        endpoint_name="bat-acrosstreatment", contexts=contexts, splits=splits
    )
    assert built["applicable"] is True
    day_field = next(entry for entry in built["fields"] if entry["field"] == "recorded")
    assert day_field["enrolment_range"][1] - day_field["enrolment_range"][0] == 7.0


def test_a_missing_field_is_an_error_rather_than_a_filled_in_value() -> None:
    with pytest.raises(ValueError, match="carry no"):
        context_representation(
            endpoint_name="zebra-finch",
            contexts=[{"calendar_date_ordinal": 1.0}, {}],
            splits=["enrollment", "query"],
        )


def test_per_dimension_standardisation_is_accepted_for_a_narrow_control() -> None:
    case = corpus(identities=5, sessions=5, per_cell=4, session_shift=0.8, identity_shift=0.2)
    result = session_similarity(**case, standardisation=PER_DIMENSION, replicates=100)
    assert result["sufficient"] is True


class FakeRecord:
    def __init__(self, filename: str, identity: str, split: str, context: dict) -> None:
        self.filename, self.identity, self.split, self.context = filename, identity, split, context


class FakeEndpoint:
    def __init__(self, name: str, records: list) -> None:
        self.name, self.records, self.manifest_sha256 = name, records, "fixture"

    @property
    def identities(self) -> list[str]:
        return sorted({record.identity for record in self.records})


def test_an_endpoint_with_no_session_field_says_so_rather_than_reporting_nothing() -> None:
    """The mutation this catches: omitting the similarity block when no session
    exists, which reads in a results table as a null result rather than as a
    measurement that could not be made."""

    from xinyenyana.nuisance import run_nuisance

    records = [
        FakeRecord(f"c{index}", f"i{index % 3}", "enrollment" if index % 2 else "query", {})
        for index in range(12)
    ]
    result = run_nuisance(
        FakeEndpoint("chiffchaff-withinyear", records),
        np.random.default_rng(1).normal(size=(12, 4)),
        representation="embedding.mean",
        session_key=None,
    )
    assert result["cross_identity_session_similarity"]["available"] is False
    assert "publishes no session field" in result["cross_identity_session_similarity"]["reason"]
    assert result["context_baseline"]["applicable"] is False


def test_the_run_reports_both_splits_separately() -> None:
    from xinyenyana.nuisance import run_nuisance

    records, rows = [], []
    rng = np.random.default_rng(17)
    axes = rng.normal(size=(6, 8))
    for split in ("enrollment", "query"):
        for identity in range(5):
            for session in range(6):
                for index in range(3):
                    records.append(
                        FakeRecord(
                            f"{split}-{identity}-{session}-{index}",
                            f"i{identity}",
                            split,
                            {"date": f"d{session}", "calendar_date_ordinal": float(session)},
                        )
                    )
                    rows.append(axes[session] + rng.normal(0, 0.2, 8))
    result = run_nuisance(
        FakeEndpoint("zebra-finch", records),
        np.asarray(rows),
        representation="embedding.mean",
        session_key="date",
        replicates=100,
    )
    similarity = result["cross_identity_session_similarity"]
    assert similarity["available"] is True
    for split in ("enrollment", "query"):
        assert similarity[split]["sufficient"] is True
        assert similarity[split]["difference"] > 0.1


def test_a_vector_count_that_does_not_match_the_clips_is_an_error() -> None:
    from xinyenyana.nuisance import run_nuisance

    records = [FakeRecord("a", "i0", "enrollment", {}), FakeRecord("b", "i1", "query", {})]
    with pytest.raises(ValueError, match="vectors for"):
        run_nuisance(
            FakeEndpoint("zebra-finch", records),
            np.zeros((3, 4)),
            representation="embedding.mean",
            session_key=None,
        )


def test_a_timestamp_session_is_grouped_by_its_calendar_date_on_the_bat() -> None:
    """The mutation this catches: reading the session field whole on an endpoint
    whose context table calls it a day. On the bat that turns almost every clip
    into its own session and leaves 14 same-session cross-identity pairs where
    the registered grouping has 301,313, so the statistic reports itself
    insufficient instead of measuring anything."""

    from xinyenyana.nuisance import run_nuisance

    records, rows = [], []
    rng = np.random.default_rng(23)
    axes = rng.normal(size=(6, 8))
    for split in ("enrollment", "query"):
        for identity in range(5):
            for day in range(6):
                for index in range(3):
                    records.append(
                        FakeRecord(
                            f"{split}-{identity}-{day}-{index}",
                            f"i{identity}",
                            split,
                            {
                                # Same day, a different second for every clip.
                                "recorded": f"2013-02-{day + 10:02d} 00:{index:02d}:{identity:02d}",
                                "treatment": "t1",
                            },
                        )
                    )
                    rows.append(axes[day] + rng.normal(0, 0.2, 8))
    result = run_nuisance(
        FakeEndpoint("bat-acrosstreatment", records),
        np.asarray(rows),
        representation="embedding.mean",
        session_key="recorded",
        replicates=100,
    )
    assert result["session_granularity"] == "day"
    similarity = result["cross_identity_session_similarity"]
    for split in ("enrollment", "query"):
        assert similarity[split]["sessions"] == 6
        assert similarity[split]["sufficient"] is True
        assert similarity[split]["difference"] > 0.1


def test_a_session_field_the_context_table_does_not_call_a_day_is_read_whole() -> None:
    """The mutation this catches: truncating every session label to ten
    characters, which would silently merge two BirdPark recording files whose
    names share a prefix."""

    from xinyenyana.nuisance import run_nuisance

    records, rows = [], []
    rng = np.random.default_rng(29)
    for index in range(12):
        records.append(
            FakeRecord(
                f"c{index}",
                f"i{index % 3}",
                "enrollment",
                {"session": f"BP_2022-09-17_08-12-5{index % 4}"},
            )
        )
        rows.append(rng.normal(size=8))
    result = run_nuisance(
        FakeEndpoint("birdpark-juv03", records),
        np.asarray(rows),
        representation="embedding.mean",
        session_key="session",
        replicates=50,
    )
    assert result["session_granularity"] == "value"
    assert result["cross_identity_session_similarity"]["enrollment"]["sessions"] == 4


def test_every_endpoint_the_loader_accepts_is_declared_one_way_or_the_other() -> None:
    """The mutation this catches: adding an endpoint and leaving the context
    table alone, which for RookID meant its declared field was never read and
    every result recorded "the release publishes no recording-circumstance
    field" as the reason. A name in neither collection now raises."""

    from xinyenyana.a2 import STOWELL_ENDPOINTS
    from xinyenyana.bats import BAT_ENDPOINTS
    from xinyenyana.birdpark import BIRDPARK_ENDPOINTS
    from xinyenyana.huang import HUANG_ENDPOINTS
    from xinyenyana.nuisance import CONTEXT_FIELDS, endpoint_fields
    from xinyenyana.right_whale import ENDPOINT as RIGHT_WHALE

    names = {
        *STOWELL_ENDPOINTS,
        *HUANG_ENDPOINTS,
        *BAT_ENDPOINTS,
        *BIRDPARK_ENDPOINTS,
        RIGHT_WHALE,
        # The three endpoints whose loaders name them from their manifests.
        "rookid",
        "rookid-ch2",
        "zebra-finch",
        "great-tit",
    }
    for name in sorted(names):
        endpoint_fields(name)
    assert endpoint_fields("rookid-ch2") == CONTEXT_FIELDS["rookid"]
    assert endpoint_fields("right-whale-aiid") == ()


def test_an_endpoint_name_in_neither_collection_is_an_error() -> None:
    from xinyenyana.nuisance import endpoint_fields

    with pytest.raises(ValueError, match="neither CONTEXT_FIELDS nor NO_CONTEXT_FIELDS"):
        endpoint_fields("rookid-ch7-typo")
