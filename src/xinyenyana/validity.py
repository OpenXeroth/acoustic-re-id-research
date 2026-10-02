"""The validity gate: how much of an identity result is the place, not the animal.

Stowell and colleagues published a background recording beside every call, the
ambient sound from the same location with no call in it, so that a classifier
can be run on the place alone. A method that identifies individuals from the
backgrounds has been caught reading the location, and that disqualifies it
whatever its foreground score.

The background is never a feature. It is only ever the control a method has to
fail.
"""

from __future__ import annotations

from typing import Any

from xinyenyana.a2 import Endpoint
from xinyenyana.evaluation import HEADS, KERNEL_RIDGE, PER_CLIP_L2, evaluate_endpoint

RIDGE_LAMBDA = 1.0
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
SEED = 17

#: Every pairing of what a gallery is built from with what is scored against it.
#: The first is the ordinary measurement, the second is the probe, and the two
#: crossings say whether the place alone carries the match.
PAIRINGS: tuple[tuple[str, str], ...] = (
    ("foreground", "foreground"),
    ("background", "background"),
    ("foreground", "background"),
    ("background", "foreground"),
)


def foreground_enrolment_fisher(endpoint: Endpoint, vectors: Any) -> float:
    """Between-identity over within-identity variance, on one side of one condition.

    The enrolment-only criterion this project already uses to choose a layer and
    a playback rate. It reads no query clip and no score, so where one endpoint
    is built in several arms, as the right whale is at eleven frequency shifts,
    it can name the reported arm without the choice being made on the answer.
    """

    import numpy as np

    from xinyenyana.a5 import fisher_ratio

    rows = [
        index
        for index, record in enumerate(endpoint.records)
        if record.split == "enrollment"
        and str(record.context.get("condition", "foreground")) == "foreground"
    ]
    if len(rows) < 2:
        raise ValueError(f"{endpoint.name} has {len(rows)} foreground enrolment clips")
    matrix = np.asarray(vectors, dtype=np.float64)[rows]
    return float(fisher_ratio(matrix, [endpoint.records[index].identity for index in rows]))


def run_validity_gate(
    *,
    endpoint: Endpoint,
    vectors: Any,
    representation: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """One representation scored on every pairing of conditions."""

    conditions = {str(record.context.get("condition", "foreground")) for record in endpoint.records}
    missing = {name for pairing in PAIRINGS for name in pairing} - conditions
    if missing:
        raise ValueError(
            f"{endpoint.name} carries no {', '.join(sorted(missing))} clips, "
            "so the place cannot be separated from the animal on it"
        )
    records = [record.as_evaluation_record() for record in endpoint.records]
    results: dict[str, Any] = {}
    per_head: dict[str, dict[str, Any]] = {head: {} for head in HEADS}
    seed = SEED
    for enrollment, query in PAIRINGS:
        seed += 1
        for head in HEADS:
            evaluated = evaluate_endpoint(
                records=records,
                vectors=vectors,
                representation=representation,
                enrollment_condition=enrollment,
                query_condition=query,
                ridge_lambda=RIDGE_LAMBDA,
                # The seed belongs to the pairing rather than the head, so the
                # three heads see the same permutations and the same bootstrap
                # draws and are comparable on one pairing.
                seed=seed,
                permutations=permutations,
                bootstrap_replicates=bootstrap_replicates,
                standardisation=PER_CLIP_L2,
                head=head,
            )
            reading = {
                "accuracy": evaluated["classification"]["accuracy"],
                "macro_recall": evaluated["classification"]["macro_recall"],
                "identity_block_bootstrap_accuracy_95": evaluated[
                    "identity_block_bootstrap_accuracy_95"
                ],
                "permutation_p": evaluated["permutation_control"]["p_value_plus_one"],
                "roc_auc": evaluated["verification"]["roc_auc"],
                "enrollment_calls": evaluated["enrollment_calls"],
                "query_calls": evaluated["query_calls"],
            }
            per_head[head][f"{enrollment}_to_{query}"] = reading
            if head == KERNEL_RIDGE:
                # The keys every gate result already on record carries. Adding
                # two heads must not move a number already published, so the
                # fitted head keeps its own place in the result unchanged.
                results[f"{enrollment}_to_{query}"] = {
                    key: value for key, value in reading.items() if key != "macro_recall"
                }
    fisher = foreground_enrolment_fisher(endpoint, vectors)
    chance = 1.0 / len(endpoint.identities)
    background = results["background_to_background"]["accuracy"]
    return {
        "endpoint": endpoint.name,
        "representation": representation,
        "heads": list(HEADS),
        "pairings_by_head": per_head,
        "foreground_enrolment_fisher_ratio": fisher,
        "identities": len(endpoint.identities),
        "clips": len(endpoint.records),
        "chance_accuracy": chance,
        "manifest_sha256": endpoint.manifest_sha256,
        "seed": SEED,
        "bootstrap_replicates": bootstrap_replicates,
        "permutations": permutations,
        "pairings": results,
        # Stated as a ratio rather than a verdict. A method that reads the place
        # scores far above chance on backgrounds alone; nothing here declares a
        # threshold at which it is disqualified.
        "background_only_over_chance": background / chance,
    }
