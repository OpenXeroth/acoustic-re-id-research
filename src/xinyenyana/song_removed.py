"""The song-removal control, through the closed-set path.

Deleting every annotated note from a Great Tit recording, and leaving every
other sample exactly as published, still separates enrolled birds from strangers
at 0.548 where the untouched songs reach 0.617. That figure is one median over
sixteen allocations of the open-set experiment under one head, and nothing has
asked what the same audio does when the task is naming a bird rather than
deciding whether to answer.

The question it answers. What survives note removal is either the bird's
recording, meaning the nestbox, the recorder and the ambient sound of that
territory, or something in the bird's own sound outside the annotated notes,
such as calls the annotator did not mark. The first is another instance of the
confound. The second is a completeness problem in the published annotations,
which is a different finding. Nothing registered here separates the two, and
that is said with the result rather than left to the reader.

Only the Great Tit publishes per-note annotations, so this stays on the Great
Tit. The masking itself is ``open_set.write_masked_clip``, already registered
and already used for the open-set figure, so both arms mask identically.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from xinyenyana.a2 import Endpoint
from xinyenyana.evaluation import HEADS, PER_CLIP_L2, evaluate_endpoint
from xinyenyana.open_set import write_masked_clip

RIDGE_LAMBDA = 1.0
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
SEED = 17

#: The arm names, so a result cannot be read as the wrong one.
UNTOUCHED = "untouched"
SONG_REMOVED = "song-removed"

#: Read with every figure. Neither is tested by anything registered here.
RIVAL_EXPLANATIONS = (
    "what survives note removal is either the recording, meaning the nestbox, "
    "the recorder and the ambient sound of that territory, or sound the bird "
    "made outside the annotated notes, such as calls the annotator did not "
    "mark; this measurement does not separate them"
)


def masked_endpoint(endpoint: Endpoint, *, scratch: Path) -> Endpoint:
    """The same endpoint with every annotated note zeroed.

    Same clips, same identities, same split, same filenames. Only the samples
    inside the published annotations change, so the two arms differ in one thing.
    """

    if not all("annotation" in record.context for record in endpoint.records):
        missing = sum(1 for record in endpoint.records if "annotation" not in record.context)
        raise ValueError(
            f"{endpoint.name}: {missing} clips carry no per-note annotation, "
            "so the song cannot be removed from them"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    return replace(
        endpoint,
        name=f"{endpoint.name}-song-removed",
        records=tuple(
            replace(
                record,
                path=write_masked_clip(record, scratch / f"{record.filename}.wav"),
            )
            for record in endpoint.records
        ),
    )


def compare_arms(
    endpoint: Endpoint,
    *,
    untouched_vectors: Any,
    masked_vectors: Any,
    representation: str,
    permutations: int = PERMUTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Both arms on the same clips and the same split, under all three heads."""

    import numpy as np

    records = [record.as_evaluation_record() for record in endpoint.records]
    arms: dict[str, Any] = {}
    seed = SEED
    for arm, vectors in ((UNTOUCHED, untouched_vectors), (SONG_REMOVED, masked_vectors)):
        matrix = np.asarray(vectors, dtype=np.float64)
        if len(matrix) != len(records):
            raise ValueError(f"{arm}: {len(matrix)} vectors for {len(records)} clips")
        per_head: dict[str, Any] = {}
        for head in HEADS:
            seed += 1
            evaluated = evaluate_endpoint(
                records=records,
                vectors=matrix,
                representation=representation,
                enrollment_condition="foreground",
                query_condition="foreground",
                ridge_lambda=RIDGE_LAMBDA,
                seed=seed,
                permutations=permutations,
                bootstrap_replicates=bootstrap_replicates,
                standardisation=PER_CLIP_L2,
                head=head,
            )
            per_head[head] = {
                "accuracy": evaluated["classification"]["accuracy"],
                "macro_recall": evaluated["classification"]["macro_recall"],
                "identity_block_bootstrap_accuracy_95": evaluated[
                    "identity_block_bootstrap_accuracy_95"
                ],
                "permutation_p": evaluated["permutation_control"]["p_value_plus_one"],
                "roc_auc": evaluated["verification"]["roc_auc"],
                "predictions": evaluated["predictions"],
                "enrollment_calls": evaluated["enrollment_calls"],
                "query_calls": evaluated["query_calls"],
            }
        arms[arm] = per_head

    query = [record for record in endpoint.records if record.split == "query"]
    counts = [record.identity for record in query]
    majority = max(counts.count(identity) for identity in set(counts)) / len(counts)
    return {
        "endpoint": endpoint.name,
        "representation": representation,
        "identities": len(endpoint.identities),
        "uniform_chance": 1.0 / len(endpoint.identities),
        "majority_class_floor": majority,
        "manifest_sha256": endpoint.manifest_sha256,
        "heads": list(HEADS),
        "ridge_lambda": RIDGE_LAMBDA,
        "permutations": permutations,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": SEED,
        "arms": arms,
        "rival_explanations": RIVAL_EXPLANATIONS,
    }
