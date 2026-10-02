"""Two readings of whether a representation encodes the recording session.

The first is the context baseline. Knowing the year and the nestbox, three
numbers and no audio, separates enrolled Great Tits from strangers about as well
as BirdNET's 1,024 numbers do. That control exists on one endpoint and this
builds it wherever one can be built.

The second is the cross-identity same-session similarity. The recording-day
probe already registered under session compensation is a classifier fitted on
every identity but one, so it can only choose among sessions seen in training,
and only the bat's query split has sessions shared across identities. This asks
the same question with a statistic that needs no classifier: for pairs of clips
from two **different** animals, is the similarity higher when the two share a
session. Identity cannot explain a difference, because every pair is two
different animals.

Neither separates a recording effect from a social or spatial one. Two animals
sharing a session may also share a cage or a territory, and that is reported
with every figure rather than tested here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from xinyenyana.evaluation import PER_CLIP_L2, PER_DIMENSION

#: Below this many sessions, or this many same-session pairs, the statistic is
#: recorded as insufficient with its counts rather than reported. Registered in
#: docs/measurement-protocol.md before any figure was read.
MINIMUM_SESSIONS = 4
MINIMUM_SAME_SESSION_PAIRS = 100

#: Each arm is capped at this many pairs, chosen by a hash of the pair rather
#: than by any score, so the bootstrap is tractable on an endpoint whose pair
#: count runs to millions. It is above every same-session count this project
#: has, so it binds only on the larger arm of the larger endpoints.
PAIR_CAP = 200_000
PAIR_SALT = "xyy-session-similarity-20260915"

BOOTSTRAP_REPLICATES = 2000
SEED = 17


def _pair_rank(left: str, right: str) -> int:
    """A stable order over pairs. No score moves it."""

    first, second = sorted((left, right))
    return int.from_bytes(
        hashlib.sha256(f"{PAIR_SALT}:{first}:{second}".encode()).digest()[:8], "big"
    )


def _standardised(features: Any, standardisation: str) -> Any:
    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    if standardisation == PER_DIMENSION:
        mean = values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        values = (values - mean) / scale
    elif standardisation == PER_CLIP_L2:
        pass
    else:
        raise ValueError(f"unknown standardisation: {standardisation}")
    lengths = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values / lengths


def session_similarity(
    *,
    features: Any,
    identities: Sequence[str],
    sessions: Sequence[str],
    filenames: Sequence[str],
    standardisation: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Same-session against different-session similarity, over different animals.

    Every pair is two different identities, so identity is excluded as an
    explanation by construction. The interval resamples the clusters a pair
    belongs to rather than the pairs themselves, because one clip appears in
    many pairs and a bootstrap over pairs would understate the width: a
    same-session pair clusters on its session, a different-session pair on its
    unordered pair of sessions.
    """

    import numpy as np

    counts = {len(identities), len(sessions), len(filenames), len(np.asarray(features))}
    if len(counts) != 1:
        raise ValueError("features, identities, sessions and filenames must line up")
    unit = _standardised(features, standardisation)
    identity = np.asarray(identities, dtype=object)
    session = np.asarray(sessions, dtype=object)
    distinct_sessions = sorted(set(session.tolist()))

    same: list[tuple[int, str, float]] = []
    other: list[tuple[int, str, float]] = []
    similarity = unit @ unit.T
    for left in range(len(identity)):
        for right in range(left + 1, len(identity)):
            if identity[left] == identity[right]:
                continue
            rank = _pair_rank(str(filenames[left]), str(filenames[right]))
            value = float(similarity[left, right])
            if session[left] == session[right]:
                same.append((rank, str(session[left]), value))
            else:
                first, second = sorted((str(session[left]), str(session[right])))
                other.append((rank, f"{first}|{second}", value))

    report: dict[str, Any] = {
        "sessions": len(distinct_sessions),
        "same_session_pairs": len(same),
        "different_session_pairs": len(other),
        "pair_cap": PAIR_CAP,
        "minimum_sessions": MINIMUM_SESSIONS,
        "minimum_same_session_pairs": MINIMUM_SAME_SESSION_PAIRS,
        "bootstrap": "sessions for the same-session arm, session pairs for the other",
        "replicates": replicates,
        "seed": seed,
    }
    if len(distinct_sessions) < MINIMUM_SESSIONS or len(same) < MINIMUM_SAME_SESSION_PAIRS:
        report["sufficient"] = False
        report["insufficient_because"] = (
            f"{len(distinct_sessions)} sessions and {len(same)} same-session cross-identity "
            f"pairs, against a registered floor of {MINIMUM_SESSIONS} and "
            f"{MINIMUM_SAME_SESSION_PAIRS}"
        )
        return report

    def cap(rows: list[tuple[int, str, float]]) -> list[tuple[str, float]]:
        kept = sorted(rows)[:PAIR_CAP]
        return [(cluster, value) for _, cluster, value in kept]

    same_capped, other_capped = cap(same), cap(other)
    report["same_session_pairs_used"] = len(same_capped)
    report["different_session_pairs_used"] = len(other_capped)

    def clustered(rows: list[tuple[str, float]]) -> tuple[Any, list[Any]]:
        values = np.asarray([value for _, value in rows], dtype=np.float64)
        order: dict[str, list[int]] = {}
        for index, (cluster, _) in enumerate(rows):
            order.setdefault(cluster, []).append(index)
        return values, [np.asarray(members) for members in order.values()]

    same_values, same_clusters = clustered(same_capped)
    other_values, other_clusters = clustered(other_capped)
    same_median = float(np.median(same_values))
    other_median = float(np.median(other_values))

    generator = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        drawn_same = np.concatenate(
            [
                same_clusters[i]
                for i in generator.integers(0, len(same_clusters), len(same_clusters))
            ]
        )
        drawn_other = np.concatenate(
            [
                other_clusters[i]
                for i in generator.integers(0, len(other_clusters), len(other_clusters))
            ]
        )
        differences[index] = np.median(same_values[drawn_same]) - np.median(
            other_values[drawn_other]
        )

    report.update(
        {
            "sufficient": True,
            "same_session_median": same_median,
            "different_session_median": other_median,
            "difference": same_median - other_median,
            "difference_95": [
                float(np.percentile(differences, 2.5)),
                float(np.percentile(differences, 97.5)),
            ],
            "same_session_clusters": len(same_clusters),
            "different_session_clusters": len(other_clusters),
            # Neither of these is tested here and both are live explanations.
            "rival_explanations": (
                "two animals recorded in one session may share a cage, a territory "
                "or a social group, so a same-session similarity need not be a "
                "property of the recording"
            ),
        }
    )
    return report


#: What each endpoint publishes about a recording's circumstances, and nothing
#: about its sound. A field named here is read from the clip's own context;
#: an endpoint absent from this table publishes none. Registered in
#: docs/measurement-protocol.md.
CONTEXT_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "great-tit": (("year", "categorical"), ("nest_x", "continuous"), ("nest_y", "continuous")),
    "bat-acrosstreatment": (("treatment", "categorical"), ("recorded", "day")),
    "zebra-finch": (("calendar_date_ordinal", "continuous"),),
    "rookid": (("recording", "categorical"),),
}

#: Endpoints whose release publishes nothing about a recording's circumstances.
#: Listed rather than inferred from a name being absent above, because an
#: endpoint carries the name its loader gives it and those names are not all
#: literals: a RookID sample built on a channel other than the registered one is
#: named ``rookid-ch<N>``, and for two weeks that name matched no key here, so
#: the declared ``recording`` field was never read and every RookID result
#: recorded "the release publishes no recording-circumstance field" as its
#: reason. That reason is the evidence a reader weighs, and it was a typo
#: wearing a finding's clothes. A name in neither collection now raises.
NO_CONTEXT_FIELDS: frozenset[str] = frozenset(
    {
        "chiffchaff-withinyear",
        "chiffchaff-acrossyear",
        "littleowl-acrossyear",
        "pipit-withinyear",
        "pipit-acrossyear",
        "cockatoo-fold1",
        "cockatoo-fold2",
        "cockatoo-fold3",
        "cockatoo-fold4",
        "cockatoo-fold5",
        "penguin-fold1",
        "penguin-fold2",
        "penguin-fold3",
        "penguin-fold4",
        "penguin-fold5",
        "penguin-acrossnight",
        "right-whale-aiid",
        "birdpark-juv01",
        "birdpark-juv03",
        "birdpark-cop08",
        "birdpark-cop09",
    }
)


def endpoint_fields(endpoint_name: str) -> tuple[tuple[str, str], ...]:
    """The recording-circumstance fields one endpoint publishes, or none.

    A RookID sample cut from a channel other than the registered one is named
    ``rookid-ch<N>`` by its loader, and that variant reads the same manifest and
    publishes the same field, so it resolves to the same row. Every other name
    matches exactly. A name in neither collection raises, because the
    alternative is recording a misspelling as a property of a release.
    """

    if endpoint_name in CONTEXT_FIELDS:
        return CONTEXT_FIELDS[endpoint_name]
    channel = endpoint_name.removeprefix("rookid-ch")
    if channel != endpoint_name and channel.isdigit():
        return CONTEXT_FIELDS["rookid"]
    if endpoint_name in NO_CONTEXT_FIELDS:
        return ()
    raise ValueError(
        f"{endpoint_name!r} appears in neither CONTEXT_FIELDS nor NO_CONTEXT_FIELDS; "
        "an endpoint that publishes no recording-circumstance field is listed there "
        "rather than inferred from its absence here"
    )


def context_representation(
    *,
    endpoint_name: str,
    contexts: Sequence[dict[str, Any]],
    splits: Sequence[str],
) -> dict[str, Any]:
    """The recording's circumstances as a representation, and whether it can fire.

    A categorical field is one-hot over the values seen in the **enrolment**
    split alone, so a query value that never enrolled encodes as zeros, which is
    what a classifier fitted on enrolment actually knows about it. A continuous
    field passes through. A ``day`` field is a timestamp read as its calendar
    date and then as an ordinal, which is continuous.

    Every split in this project separates sessions, so a field that is only a
    session index takes query values that never enrolled and the head can
    predict nothing from it. That is checked here rather than assumed: the
    representation is applicable when its encoded query rows are not all
    identical, and when every categorical field it uses has at least one value
    in both splits. An endpoint the check rules out is reported with its counts
    and the reason, not as a score of zero.
    """

    import numpy as np

    declared = endpoint_fields(endpoint_name)
    if not declared:
        return {
            "endpoint": endpoint_name,
            "applicable": False,
            "reason": "the release publishes no recording-circumstance field",
            "fields": [],
        }
    split = np.asarray(splits, dtype=object)
    enrolment = split == "enrollment"
    query = split == "query"
    columns: list[Any] = []
    names: list[str] = []
    detail: list[dict[str, Any]] = []
    blocked: list[str] = []

    for field, kind in declared:
        missing = [index for index, row in enumerate(contexts) if field not in row]
        if missing:
            raise ValueError(f"{endpoint_name}: {len(missing)} clips carry no {field!r}")
        raw = [row[field] for row in contexts]
        if kind == "categorical":
            values = np.asarray([str(value) for value in raw], dtype=object)
            enrolled = sorted(set(values[enrolment].tolist()))
            queried = sorted(set(values[query].tolist()))
            shared = sorted(set(enrolled) & set(queried))
            detail.append(
                {
                    "field": field,
                    "kind": kind,
                    "enrolment_values": len(enrolled),
                    "query_values": len(queried),
                    "shared_values": len(shared),
                }
            )
            if not shared:
                blocked.append(
                    f"{field} takes no value in both splits "
                    f"({len(enrolled)} enrolled, {len(queried)} queried)"
                )
            for value in enrolled:
                columns.append((values == value).astype(np.float64))
                names.append(f"{field}={value}")
        else:
            if kind == "day":
                import datetime

                numbers = np.asarray(
                    [datetime.date.fromisoformat(str(value)[:10]).toordinal() for value in raw],
                    dtype=np.float64,
                )
            else:
                numbers = np.asarray([float(value) for value in raw], dtype=np.float64)
            detail.append(
                {
                    "field": field,
                    "kind": kind,
                    "enrolment_range": [
                        float(numbers[enrolment].min()),
                        float(numbers[enrolment].max()),
                    ],
                    "query_range": [float(numbers[query].min()), float(numbers[query].max())],
                }
            )
            columns.append(numbers)
            names.append(field)

    matrix = np.column_stack(columns) if columns else np.zeros((len(contexts), 0))
    distinct_query_rows = len({tuple(row) for row in matrix[query].tolist()})
    if distinct_query_rows < 2:
        blocked.append(
            f"every query clip encodes identically, so the head can name one class "
            f"({distinct_query_rows} distinct query row)"
        )
    return {
        "endpoint": endpoint_name,
        "applicable": not blocked,
        "reason": "; ".join(blocked) if blocked else "",
        "fields": detail,
        "names": names,
        "vectors": matrix,
        "distinct_query_rows": distinct_query_rows,
    }


def _session_granularity(endpoint_name: str, session_key: str) -> str:
    """How the session field is read: as its calendar date, or as it stands.

    The field named as the session is often the same field the context baseline
    already declares in :data:`CONTEXT_FIELDS`, and that table says whether it
    is a timestamp. A timestamp read whole makes almost every clip its own
    session: on the bat, ``recorded`` holds a time to the second, so its 6,801
    enrolment clips fall into 6,178 sessions and 14 cross-identity pairs share
    one. Read as its calendar date the same clips fall into 62 sessions and
    301,313 such pairs, which are the counts registered in
    ``docs/measurement-protocol.md`` before any figure was read. One table
    therefore decides the grouping for both readings of a field, so they cannot
    disagree about what a session is.
    """

    for field, kind in endpoint_fields(endpoint_name):
        if field == session_key:
            return "day" if kind == "day" else "value"
    return "value"


def _session_value(raw: Any, granularity: str) -> str:
    """One clip's session label under that granularity."""

    if granularity != "day":
        return str(raw)
    import datetime

    return datetime.date.fromisoformat(str(raw)[:10]).isoformat()


def run_nuisance(
    endpoint: Any,
    vectors: Any,
    *,
    representation: str,
    session_key: str | None,
    standardisation: str = PER_CLIP_L2,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Both session readings for one endpoint and one representation.

    ``session_key`` names the context field that holds the session. An endpoint
    that publishes none passes ``None`` and the similarity is recorded as
    unavailable with that reason, which is not the same as a null result.
    """

    import numpy as np

    contexts = [record.context for record in endpoint.records]
    splits = [record.split for record in endpoint.records]
    filenames = [record.filename for record in endpoint.records]
    identities = [record.identity for record in endpoint.records]
    matrix = np.asarray(vectors, dtype=np.float64)
    if len(matrix) != len(endpoint.records):
        raise ValueError(f"{len(matrix)} vectors for {len(endpoint.records)} clips")

    granularity = None if session_key is None else _session_granularity(endpoint.name, session_key)
    similarity: dict[str, Any] = {}
    if session_key is None:
        similarity = {
            "available": False,
            "reason": "this endpoint publishes no session field",
        }
    else:
        missing = [
            name for name, row in zip(filenames, contexts, strict=True) if session_key not in row
        ]
        if missing:
            raise ValueError(f"{len(missing)} clips carry no {session_key!r}, first {missing[0]}")
        sessions = [_session_value(row[session_key], str(granularity)) for row in contexts]
        for split in ("enrollment", "query"):
            rows = [index for index, value in enumerate(splits) if value == split]
            similarity[split] = session_similarity(
                features=matrix[rows],
                identities=[identities[index] for index in rows],
                sessions=[sessions[index] for index in rows],
                filenames=[filenames[index] for index in rows],
                standardisation=standardisation,
                replicates=replicates,
                seed=seed,
            )
        similarity["available"] = True

    built = context_representation(endpoint_name=endpoint.name, contexts=contexts, splits=splits)
    baseline = {key: value for key, value in built.items() if key != "vectors"}
    if built["applicable"]:
        baseline["width"] = int(built["vectors"].shape[1])

    return {
        "endpoint": endpoint.name,
        "representation": representation,
        "identities": len(endpoint.identities),
        "clips": len(endpoint.records),
        "manifest_sha256": endpoint.manifest_sha256,
        "session_key": session_key,
        "session_granularity": granularity,
        "standardisation": standardisation,
        "cross_identity_session_similarity": similarity,
        "context_baseline": baseline,
    }
