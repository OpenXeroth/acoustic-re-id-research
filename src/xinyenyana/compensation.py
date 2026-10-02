"""Removing the recording from a representation, and measuring whether it helped.

E-REID-01. This project has measured that dividing recordings between enrolment
and scoring changes identity accuracy by more than the encoder does. This module
asks the next question: whether the recording occupies directions that identity
does not need, so that subtracting them recovers what the division takes away.

Every treatment here is fitted on enrolment clips alone and applied unchanged to
both sides, so nothing is tuned on the clips it is scored on. The parameters are
registered in `docs/measurement-protocol.md` and swept over a fixed grid, and
the whole curve is reported rather than its best point.

The speaker-verification literature built this family for exactly this problem.
The sweep in `docs/research/where-this-work-sits.md` found no application of any
of it to animal individual identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xinyenyana.evaluation import HEADS, evaluate_endpoint, leave_one_identity_out

NONE = "none"
PER_SESSION_BATCH = "per-session mean, batch"
PER_SESSION_STREAMING = "per-session mean, streaming"
WCCN = "within-class covariance normalisation"
NUISANCE_PROJECTION = "between-session subspace removed"

TREATMENTS: tuple[str, ...] = (
    NONE,
    PER_SESSION_BATCH,
    PER_SESSION_STREAMING,
    WCCN,
    NUISANCE_PROJECTION,
)

#: Shrinkage toward a scaled identity matrix. At 1.0 the transform is a scalar,
#: so that end of the sweep reproduces the untreated baseline and the curve has
#: its own control built into it.
WCCN_SHRINKAGE: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 0.9, 1.0)

#: Directions of between-session scatter removed. At 0 nothing is removed.
PROJECTION_DIMENSIONS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)

#: Clips of a session that must be seen before the streaming mean is used. A
#: deployment receives clips one at a time and cannot average a whole night
#: before answering; this is what makes the streaming variant deployable and the
#: batch one not.
STREAMING_MINIMUM_HISTORY = 5

#: Sessions required on the enrolment side before a session-aware treatment can
#: be fitted. Two sessions give a between-session scatter of rank one, which is
#: not a subspace.
MINIMUM_SESSIONS = 3

EIGENVALUE_FLOOR = 1e-10


def parameters_for(treatment: str) -> tuple[Any, ...]:
    """The registered grid for a treatment, or a single null value."""

    if treatment == WCCN:
        return WCCN_SHRINKAGE
    if treatment == NUISANCE_PROJECTION:
        return PROJECTION_DIMENSIONS
    return (None,)


def applicable(
    treatment: str, *, sessions: Sequence[str] | None, enrollment_rows: Any
) -> str | None:
    """The reason a treatment cannot run here, or None if it can.

    Returned rather than raised so that a run records what it could not do and
    why, instead of a gap nobody sees.
    """

    if treatment in {NONE, WCCN}:
        return None
    if sessions is None:
        return "the endpoint carries no session label"
    enrolled = {sessions[row] for row in enrollment_rows}
    if len(enrolled) < MINIMUM_SESSIONS:
        return (
            f"the enrolment side spans {len(enrolled)} session(s) and {MINIMUM_SESSIONS} are needed"
        )
    return None


def _inverse_square_root(matrix: Any) -> Any:
    import numpy as np

    values, vectors = np.linalg.eigh(matrix)
    floor = EIGENVALUE_FLOOR * max(float(values.max()), 1e-30)
    values = np.maximum(values, floor)
    return (vectors / np.sqrt(values)) @ vectors.T


def compensate(
    treatment: str,
    *,
    vectors: Any,
    enrollment_rows: Sequence[int],
    identities: Sequence[str],
    sessions: Sequence[str] | None,
    order: Sequence[float] | None,
    parameter: Any,
) -> Any:
    """Apply one treatment, fitted on the enrolment rows alone.

    ``order`` is what a streaming treatment consumes: a number per clip that
    puts the clips of a session in the order they were recorded.
    """

    import numpy as np

    features = np.asarray(vectors, dtype=np.float64).copy()
    enrollment = list(enrollment_rows)
    if treatment == NONE:
        return features

    if treatment == WCCN:
        shrinkage = float(parameter)
        train = features[enrollment]
        names = [identities[row] for row in enrollment]
        centred = []
        for name in sorted(set(names)):
            rows = [index for index, other in enumerate(names) if other == name]
            block = train[rows]
            centred.append(block - block.mean(axis=0, keepdims=True))
        stacked = np.vstack(centred)
        within = (stacked.T @ stacked) / max(len(stacked) - len(set(names)), 1)
        scale = float(np.trace(within)) / within.shape[0]
        shrunk = (1.0 - shrinkage) * within + shrinkage * scale * np.eye(within.shape[0])
        return features @ _inverse_square_root(shrunk)

    if sessions is None:
        raise ValueError(f"{treatment} needs a session label")

    if treatment == PER_SESSION_BATCH:
        # The mean of a session is computed over that session's clips on the side
        # of the split they fall on, so an enrolment mean is never informed by a
        # query clip. No identity label is used anywhere here.
        enrolled = set(enrollment)
        for side in (enrolled, set(range(len(features))) - enrolled):
            by_session: dict[str, list[int]] = {}
            for row in sorted(side):
                by_session.setdefault(sessions[row], []).append(row)
            for rows in by_session.values():
                features[rows] -= features[rows].mean(axis=0, keepdims=True)
        return features

    if treatment == PER_SESSION_STREAMING:
        if order is None:
            raise ValueError("the streaming treatment needs a recorded order")
        fallback = features[enrollment].mean(axis=0)
        by_session_all: dict[str, list[int]] = {}
        for row in range(len(features)):
            by_session_all.setdefault(sessions[row], []).append(row)
        adjusted = features.copy()
        for rows in by_session_all.values():
            ordered = sorted(rows, key=lambda row: (order[row], row))
            running = np.zeros(features.shape[1], dtype=np.float64)
            seen = 0
            for row in ordered:
                subtract = running / seen if seen >= STREAMING_MINIMUM_HISTORY else fallback
                adjusted[row] = features[row] - subtract
                running += features[row]
                seen += 1
        return adjusted

    if treatment == NUISANCE_PROJECTION:
        dimensions = int(parameter)
        if dimensions == 0:
            return features
        train = features[enrollment]
        names = [sessions[row] for row in enrollment]
        means = np.vstack(
            [
                train[[i for i, s in enumerate(names) if s == name]].mean(axis=0)
                for name in sorted(set(names))
            ]
        )
        centred = means - train.mean(axis=0, keepdims=True)
        between = (centred.T @ centred) / max(len(means) - 1, 1)
        values, directions = np.linalg.eigh(between)
        leading = directions[:, ::-1][:, : min(dimensions, directions.shape[1])]
        return features - (features @ leading) @ leading.T

    raise ValueError(f"unknown treatment: {treatment}")


RIDGE_LAMBDA = 1.0
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
SEED = 29


def majority_class_rate(identities: Sequence[str], query_rows: Sequence[int]) -> float:
    """The trivial floor: always answer the commonest identity in the query set.

    Reported beside uniform chance because with an unbalanced query set it is the
    higher of the two, and it is the one a method has to clear.
    """

    from collections import Counter

    counts = Counter(identities[row] for row in query_rows)
    total = sum(counts.values())
    return float(max(counts.values()) / total) if total else 0.0


def run_compensation(
    *,
    endpoint: Any,
    vectors: Any,
    representation: str,
    standardisation: str,
    enrollment_filter: dict[str, Any] | None = None,
    permutations: int = PERMUTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Every treatment, every registered parameter value, every head.

    ``enrollment_filter`` restricts which enrolment clips are used, by a context
    key and a set of values, with an optional cap that subsamples deterministically
    so that a restricted arm can be compared against a size-matched one.
    """

    import numpy as np

    records = endpoint.records
    identities = [record.identity for record in records]
    sessions: list[str] | None = None
    order: list[float] | None = None
    if all("recorded" in record.context for record in records):
        sessions = [str(record.context["recorded"])[:10] for record in records]
        order = [
            float(index)
            for index in np.argsort(np.argsort([str(r.context["recorded"]) for r in records]))
        ]

    # Only the clips the head will score. An endpoint that carries a background
    # recording beside every call holds both conditions in one record set, and
    # the first run of this fitted every treatment over foreground and background
    # together: 546 of the little owl's 1,091 enrolment rows and 5,011 of the
    # chiffchaff's 10,118 were ambient sound with no bird in it, carrying the
    # bird's identity label. The largest within-identity direction that produces
    # is the difference between a call and its silence, which is not the quantity
    # any of these treatments is meant to estimate.
    def _foreground(row: int) -> bool:
        return str(records[row].context.get("condition", "foreground")) == "foreground"

    enrollment_rows = [
        i for i, record in enumerate(records) if record.split == "enrollment" and _foreground(i)
    ]
    query_rows = [
        i for i, record in enumerate(records) if record.split == "query" and _foreground(i)
    ]
    filter_note: dict[str, Any] = {}
    if enrollment_filter:
        key = str(enrollment_filter["key"])
        values = {str(value) for value in enrollment_filter["values"]}
        kept = [row for row in enrollment_rows if str(records[row].context.get(key)) in values]
        cap = enrollment_filter.get("cap")
        if cap is not None and len(kept) > int(cap):
            rng = np.random.default_rng(SEED)
            kept = sorted(rng.choice(kept, size=int(cap), replace=False).tolist())
        filter_note = {
            "key": key,
            "values": sorted(values),
            "cap": cap,
            "enrollment_clips_kept": len(kept),
            "enrollment_clips_dropped": len(enrollment_rows) - len(kept),
        }
        enrollment_rows = kept

    evaluation_records = [record.as_evaluation_record() for record in records]
    dropped = set(range(len(records))) - set(enrollment_rows) - set(query_rows)
    for row in sorted(dropped):
        evaluation_records[row] = {**evaluation_records[row], "split": "unused"}

    # E-REID-01 registers the background-only accuracy under each treatment on
    # the endpoints that publish a background recording beside every call. The
    # treatment is still fitted on the foreground enrolment clips alone; this
    # arm applies it unchanged to the ambient recordings and asks the same
    # question of them, so a compensation that raises identity accuracy by
    # sharpening the place shows up here rather than being inferred from the
    # foreground number moving.
    background_records = (
        [record.as_evaluation_record() for record in records]
        if any(str(r.context.get("condition", "foreground")) == "background" for r in records)
        else None
    )

    # The recording-day probe asks what the **query** clips say about the day
    # they were recorded, so it needs a date on the query side and nothing more.
    # Gating it on every clip carrying one silently skipped the little penguin
    # across nights, whose query side publishes a date in each filename and
    # whose enrolment side publishes none; that endpoint is one of the two the
    # probe is registered to run on.
    day_probe_target = None
    if query_rows and all("recorded" in records[row].context for row in query_rows):
        days = [str(records[row].context["recorded"])[:10] for row in query_rows]
        if len(set(days)) > 1:
            day_probe_target = days

    summary: dict[str, Any] = {
        "endpoint": endpoint.name,
        "representation": representation,
        "standardisation": standardisation,
        "clips": len(records),
        "identities": len(endpoint.identities),
        "enrollment_clips": len(enrollment_rows),
        "query_clips": len(query_rows),
        "chance_accuracy": 1.0 / len(endpoint.identities),
        "majority_class_rate": majority_class_rate(identities, query_rows),
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "seed": SEED,
        "permutations": permutations,
        "bootstrap_replicates": bootstrap_replicates,
        "sessions": len({sessions[row] for row in enrollment_rows}) if sessions else None,
        "enrollment_filter": filter_note or None,
        "background_only_arm": background_records is not None,
        "treatments": {},
        "not_applicable": {},
    }

    seed = SEED
    for treatment in TREATMENTS:
        reason = applicable(treatment, sessions=sessions, enrollment_rows=enrollment_rows)
        if reason is not None:
            summary["not_applicable"][treatment] = reason
            continue
        rows: list[dict[str, Any]] = []
        for parameter in parameters_for(treatment):
            adjusted = compensate(
                treatment,
                vectors=vectors,
                enrollment_rows=enrollment_rows,
                identities=identities,
                sessions=sessions,
                order=order,
                parameter=parameter,
            )
            probe = None
            if day_probe_target is not None:
                # Prediction 3: a session-aware treatment should make the
                # recording day harder to read out of the query clips. Computed
                # once per parameter value, because it does not depend on the
                # head.
                probe = leave_one_identity_out(
                    features=np.asarray(adjusted)[query_rows],
                    identities=np.asarray([identities[row] for row in query_rows]),
                    targets={"recording_day": np.asarray(day_probe_target, dtype=object)},
                    categorical=frozenset({"recording_day"}),
                    standardisation=standardisation,
                    ridge_lambda=RIDGE_LAMBDA,
                )["recording_day"]
            for head in HEADS:
                seed += 1
                evaluated = evaluate_endpoint(
                    records=evaluation_records,
                    vectors=adjusted,
                    representation=representation,
                    enrollment_condition="foreground",
                    query_condition="foreground",
                    ridge_lambda=RIDGE_LAMBDA,
                    seed=seed,
                    permutations=permutations,
                    bootstrap_replicates=bootstrap_replicates,
                    standardisation=standardisation,
                    head=head,
                )
                background = None
                if background_records is not None:
                    scored = evaluate_endpoint(
                        records=background_records,
                        vectors=adjusted,
                        representation=representation,
                        enrollment_condition="background",
                        query_condition="background",
                        ridge_lambda=RIDGE_LAMBDA,
                        seed=seed,
                        permutations=permutations,
                        bootstrap_replicates=bootstrap_replicates,
                        standardisation=standardisation,
                        head=head,
                    )
                    background = {
                        "accuracy": scored["classification"]["accuracy"],
                        "macro_recall": scored["classification"]["macro_recall"],
                        "identity_block_bootstrap_accuracy_95": scored[
                            "identity_block_bootstrap_accuracy_95"
                        ],
                        "permutation_p": scored["permutation_control"]["p_value_plus_one"],
                        "enrollment_calls": scored["enrollment_calls"],
                        "query_calls": scored["query_calls"],
                    }
                rows.append(
                    {
                        "parameter": parameter,
                        "head": head,
                        "recording_day_probe": probe,
                        "background_only": background,
                        "accuracy": evaluated["classification"]["accuracy"],
                        "macro_recall": evaluated["classification"]["macro_recall"],
                        "identity_block_bootstrap_accuracy_95": evaluated[
                            "identity_block_bootstrap_accuracy_95"
                        ],
                        "permutation_p": evaluated["permutation_control"]["p_value_plus_one"],
                        "roc_auc": evaluated["verification"]["roc_auc"],
                    }
                )
        summary["treatments"][treatment] = rows
    return summary
