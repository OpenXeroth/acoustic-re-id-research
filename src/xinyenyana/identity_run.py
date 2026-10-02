"""One endpoint, BirdNET's final embedding, and every null the paper reports.

This is the measurement behind the paper's headline table. It replaces three
scripts written on 2026-09-21 (`run_rook_identity.py`, `run_confound_index.py`
and `great_tit_place_control.py`) whose outputs carried no source digest and
were never archived, so none of their numbers could be cited under the paper's
statement that every figure was read back from the archive.

Per endpoint it reports, from one seed:

- accuracy through the fixed kernel-ridge head;
- the interval over animals and the three-stage interval over animals, then
  recordings, then clips, where the endpoint names its recordings;
- the ordinary label permutation and the 95th percentile of its null, which is
  what a result has to beat, beside the uniform-chance figure it replaces;
- the within-recording permutation where a recording holds two or more animals;
- the stratified verification trials on the scoring clips;
- the confounding index of the clips actually scored.

Two options exist for the paper's two targeted questions. ``random_fold``
reassigns the same clips at random, per animal, keeping each animal's number of
scoring clips, so a comparison with the session split changes the division and
nothing else. ``equivalence_margin`` adds a one-sided test that accuracy is
below chance plus the margin, over the same animal bootstrap.
"""

from __future__ import annotations

import collections
import math
import random
from collections.abc import Sequence
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint

SEED = 17
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
RIDGE_LAMBDA = 1.0
FINAL_EMBEDDING = "embedding.mean"


def call_key(record: ClipRecord) -> tuple[str, str]:
    """Identity and the call, so the same call can be found on another channel."""

    stem = str(record.filename).rsplit("/", 1)[-1]
    return record.identity, stem.rsplit("_ch", 1)[0]


def balance_enrolment(records: Sequence[ClipRecord], cap: int, seed: int = 101) -> list[ClipRecord]:
    """Cap every animal's enrolment at the same number of clips."""

    rng = random.Random(seed)
    by_identity: dict[str, list[ClipRecord]] = {}
    for record in records:
        if record.split == "enrollment":
            by_identity.setdefault(record.identity, []).append(record)
    kept: list[ClipRecord] = []
    for _identity, rows in sorted(by_identity.items()):
        ordered = sorted(rows, key=lambda r: r.filename)
        rng.shuffle(ordered)
        kept.extend(ordered[:cap])
    return kept + [record for record in records if record.split == "query"]


def random_fold(
    records: Sequence[ClipRecord], seed: int
) -> tuple[list[ClipRecord], dict[str, Any]]:
    """The same clips, divided at random per animal instead of by session.

    Each animal keeps exactly as many scoring clips as the session split gave
    it, so the two runs differ in which clips score and in nothing else. This is
    the design the protocol prohibits for reporting identity; here it is built
    only to measure what that prohibition is worth.
    """

    rng = random.Random(seed)
    by_identity: dict[str, list[ClipRecord]] = collections.defaultdict(list)
    for record in records:
        by_identity[record.identity].append(record)
    moved = 0
    rebuilt: list[ClipRecord] = []
    for _identity, rows in sorted(by_identity.items()):
        ordered = sorted(rows, key=lambda r: r.filename)
        query_count = sum(1 for r in ordered if r.split == "query")
        rng.shuffle(ordered)
        for position, record in enumerate(ordered):
            split = "query" if position < query_count else "enrollment"
            moved += split != record.split
            rebuilt.append(
                ClipRecord(
                    filename=record.filename,
                    path=record.path,
                    identity=record.identity,
                    split=split,
                    context=record.context,
                )
            )
    report = {
        "method": "clips shuffled within each animal; each animal keeps its scoring-clip count",
        "seed": seed,
        "clips": len(rebuilt),
        "clips_moved_across_the_session_boundary": moved // 2,
        "share_of_scoring_clips_moved": (moved // 2)
        / max(1, sum(1 for r in records if r.split == "query")),
    }
    return rebuilt, report


def one_sided_equivalence(
    *,
    identities: Sequence[str],
    correct: Sequence[bool],
    bound: float,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Is accuracy below ``bound``, with the animals as the unit of resampling?

    The null is that accuracy is at or above the bound. It is rejected at 0.05
    when the 95th percentile of the animal bootstrap lies below the bound, which
    is the upper half of the two one-sided tests; the lower half, accuracy far
    below chance, is not a claim anyone makes about a background recording.
    """

    import numpy as np

    labels = np.asarray(identities)
    flags = np.asarray(correct, dtype=float)
    animals = sorted(set(identities))
    rows = [np.flatnonzero(labels == animal) for animal in animals]
    rng = np.random.default_rng(seed + 11)
    draws = np.asarray(
        [
            flags[
                np.concatenate([rows[i] for i in rng.integers(0, len(animals), len(animals))])
            ].mean()
            for _ in range(replicates)
        ]
    )
    upper = float(np.quantile(draws, 0.95))
    return {
        "bound": bound,
        "accuracy": float(flags.mean()),
        "upper_95_one_sided": upper,
        "p_value": float(np.mean(draws >= bound)),
        "equivalent_at_0.05": upper < bound,
        "replicates": replicates,
    }


def measure(
    *,
    endpoint: Endpoint,
    records: Sequence[ClipRecord],
    vectors: Any,
    condition: str = "foreground",
    seed: int = SEED,
    equivalence_margin: float | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """Every figure the headline table reports for one endpoint."""

    import numpy as np

    from xinyenyana.confound_index import confound_index, verdicts
    from xinyenyana.evaluation import PER_CLIP_L2, evaluate_endpoint
    from xinyenyana.verification_trials import stratified_trials

    evaluation_records = [record.as_evaluation_record() for record in records]
    result = evaluate_endpoint(
        records=evaluation_records,
        vectors=vectors,
        representation="birdnet-v2.4",
        enrollment_condition=condition,
        query_condition=condition,
        ridge_lambda=RIDGE_LAMBDA,
        seed=seed,
        permutations=permutations,
        bootstrap_replicates=replicates,
        standardisation=PER_CLIP_L2,
    )
    predictions = result.pop("predictions")
    query = [
        index
        for index, record in enumerate(records)
        if record.split == "query"
        and str(record.context.get("condition", "foreground")) == condition
    ]
    sessions = [records[i].session for i in query]
    trials: dict[str, Any]
    if all(session is not None for session in sessions):
        trials = stratified_trials(
            vectors=np.asarray(vectors)[query],
            identities=[records[i].identity for i in query],
            sessions=[str(session) for session in sessions],
        )
    else:
        trials = {"constructible": False, "reason": "the clips carry no recording or session field"}
    scored = [r for r in records if str(r.context.get("condition", "foreground")) == condition]
    index = confound_index(name=endpoint.name, records=scored)
    index["verdicts"] = verdicts(index)
    identities = sorted({r.identity for r in scored})
    summary: dict[str, Any] = {
        "endpoint": endpoint.name,
        "condition": condition,
        "identities": len(identities),
        "uniform_chance": 1.0 / len(identities),
        "majority_class_rate": max(
            collections.Counter(r.identity for r in scored if r.split == "query").values()
        )
        / max(1, sum(1 for r in scored if r.split == "query")),
        "seed": seed,
        "evaluation": result,
        "stratified_trials": trials,
        "confound_index": index,
        "query_correct": "".join(
            "1" if p["actual_label"] == p["predicted_label"] else "0" for p in predictions
        ),
        "query_identities": [p["actual_label"] for p in predictions],
    }
    if equivalence_margin is not None:
        summary["equivalence"] = {
            "margin": equivalence_margin,
            **one_sided_equivalence(
                identities=summary["query_identities"],
                correct=[flag == "1" for flag in summary["query_correct"]],
                bound=summary["uniform_chance"] + equivalence_margin,
                seed=seed,
                replicates=replicates,
            ),
        }
    return summary


def place_control(records: Sequence[ClipRecord]) -> dict[str, Any]:
    """Does any nestbox or recording appear on both sides of the great tit split?

    Counted from the manifest; nothing reads audio or fits anything. Within a
    split each great tit is one nestbox and a few recordings, so identity and
    place are one variable there. What decides whether a result could have come
    from learning the place is whether the place an animal is scored at is one
    it was enrolled at.
    """

    side: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    years: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    recordings: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    where: dict[str, dict[str, tuple[float, float]]] = collections.defaultdict(dict)
    for record in records:
        context = record.context
        side[record.identity][record.split].add(str(context["nestbox"]))
        years[record.identity][record.split].add(str(context["year"]))
        recordings[record.identity][record.split].add(str(context["source_recording"]))
        where[record.identity][record.split] = (float(context["nest_x"]), float(context["nest_y"]))
    per_identity: list[dict[str, Any]] = []
    for identity in sorted(side):
        (ex, ey), (qx, qy) = where[identity]["enrollment"], where[identity]["query"]
        per_identity.append(
            {
                "identity": identity,
                "shares_a_nestbox": bool(side[identity]["enrollment"] & side[identity]["query"]),
                "shares_a_source_recording": bool(
                    recordings[identity]["enrollment"] & recordings[identity]["query"]
                ),
                "years_differ": not (years[identity]["enrollment"] & years[identity]["query"]),
                "metres_between_nestboxes": round(math.hypot(ex - qx, ey - qy), 1),
            }
        )
    distances = sorted(float(row["metres_between_nestboxes"]) for row in per_identity)
    middle = len(distances) // 2
    median = (
        distances[middle] if len(distances) % 2 else (distances[middle - 1] + distances[middle]) / 2
    )
    return {
        "identities": len(per_identity),
        "identities_sharing_a_nestbox_across_the_split": sum(
            row["shares_a_nestbox"] for row in per_identity
        ),
        "identities_sharing_a_source_recording_across_the_split": sum(
            row["shares_a_source_recording"] for row in per_identity
        ),
        "identities_whose_years_differ_across_the_split": sum(
            row["years_differ"] for row in per_identity
        ),
        "metres_between_nestboxes": {
            "minimum": distances[0],
            "median": median,
            "maximum": distances[-1],
        },
        "per_identity": per_identity,
    }
