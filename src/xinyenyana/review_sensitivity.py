"""Count-matched review diagnostics; retain original published-fold results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xinyenyana.a2 import ClipRecord
from xinyenyana.archive import canonical_sha256
from xinyenyana.evaluation import (
    CLASS_MEAN,
    HEADS,
    PER_CLIP_L2,
    classification_metrics,
    head_operator,
    prepare_features,
)
from xinyenyana.identity_run import random_fold

SPLIT_SEEDS = tuple(range(101, 111))


def matched_rows(records: Sequence[ClipRecord], seed: int) -> tuple[list[int], list[int]]:
    """Reuse the rook allocation rule without reordering the embedding matrix."""
    if len({r.filename for r in records}) != len(records):
        raise ValueError("clip names must be unique")
    shuffled, _ = random_fold(records, seed)
    splits = {r.filename: r.split for r in shuffled}
    train = [i for i, r in enumerate(records) if splits[r.filename] == "enrollment"]
    query = [i for i, r in enumerate(records) if splits[r.filename] == "query"]
    for identity in {r.identity for r in records}:
        for label, rows in (("enrollment", train), ("query", query)):
            count = sum(r.identity == identity and r.split == label for r in records)
            if not count or count != sum(records[i].identity == identity for i in rows):
                raise ValueError("each individual must retain nonempty, equal split counts")
    return train, query


def score_split(
    records: Sequence[ClipRecord], vectors: Any, train: list[int], query: list[int]
) -> dict[str, Any]:
    """The original three heads, with preprocessing fitted on enrolment only."""
    import numpy as np

    identities = sorted({r.identity for r in records})
    labels = np.array([identities.index(r.identity) for r in records])
    x, y, processing = prepare_features(
        vectors[train], vectors[query], "published-vectors", standardisation=PER_CLIP_L2
    )
    actual = labels[query]
    heads = {}
    for head in HEADS:
        onehot = np.eye(len(identities))
        if head == CLASS_MEAN:
            counts = np.bincount(labels[train], minlength=len(identities))
            onehot = onehot / np.maximum(counts, 1)[:, None]
        scores = head_operator(head, train=x, query=y, ridge_lambda=1.0) @ onehot[labels[train]]
        predicted = np.argmax(scores, axis=1)
        heads[head] = {
            **classification_metrics(actual, predicted),
            "predictions": predicted.tolist(),
            "correct_per_individual": [
                int(np.sum((predicted == actual) & (actual == i))) for i in range(len(identities))
            ],
        }
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarities = []
    shares_block = []
    for qi in query:
        same = [i for i in train if labels[i] == labels[qi]]
        similarities.append(float(np.max(unit[same] @ unit[qi])))
        shares_block.append(any(records[i].split == records[qi].split for i in same))
    return {
        "heads": heads,
        "processing": processing,
        "enrollment_rows": train,
        "query_rows": query,
        "split_sha256": canonical_sha256({"enrollment": train, "query": query}),
        "query_labels": actual.tolist(),
        "query_counts_per_individual": np.bincount(actual, minlength=len(identities)).tolist(),
        "nearest_same_individual_cosine": similarities,
        "nearest_same_individual_cosine_median": float(np.median(similarities)),
        "same_individual_original_block_represented": shares_block,
        "block_warning": "Original train/test block, not verified source-recording membership",
    }


def split_summary(baseline: dict[str, Any], random_splits: Sequence[dict[str, Any]]) -> Any:
    """Paired individual bootstrap on the mean across predeclared random seeds."""
    import numpy as np

    counts = np.asarray(baseline["query_counts_per_individual"])
    if any(r["query_counts_per_individual"] != counts.tolist() for r in random_splits):
        raise ValueError("query counts differ")
    draws = np.random.default_rng(17).integers(0, len(counts), (10000, len(counts)))
    result = {}
    for head in HEADS:
        base = baseline["heads"][head]
        runs = [r["heads"][head] for r in random_splits]
        differences = np.mean([r["correct_per_individual"] for r in runs], axis=0) - np.asarray(
            base["correct_per_individual"]
        )
        bootstrap = differences[draws].sum(axis=1) / counts[draws].sum(axis=1)
        result[head] = {
            "session_accuracy": base["accuracy"],
            "session_balanced_accuracy": base["macro_recall"],
            "random_accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
            "random_accuracy_range": [float(f([r["accuracy"] for r in runs])) for f in (min, max)],
            "random_balanced_accuracy_mean": float(np.mean([r["macro_recall"] for r in runs])),
            "difference_mean": float(differences.sum() / counts.sum()),
            "difference_95_individual_bootstrap": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "interval_conditions": "10 fixed random splits; individuals resampled jointly",
        }
    return result
