"""Architecture one: one representation built from every BirdNET layer at once.

The frozen comparison asks which single layer and pooling rule is best and
answers with a rule that reads only enrolment clips. On all four endpoints it
has run on, that rule chose a layer below the final embedding, and on the little
owl it chose the one layer that reads the recording. This module asks the
question the frozen comparison cannot: whether the layers are worth more
together than any one of them is alone.

Everything fitted here is fitted on enrolment clips of the enrolment condition
and applied unchanged to the scored clips. Nothing reads a query label, and no
hyperparameter is chosen by looking at a score: the common width, the ridge on
the within-identity scatter, the seed and the replicate counts are all fixed in
`docs/measurement-protocol.md` before this ran.

The permutation control refits the projection under each shuffle, so the p-value
is about the whole fitted procedure and not about the head alone. That is why
this module carries its own scoring loop instead of calling `evaluate_endpoint`,
whose control holds the projection fixed. On planted noise at four sizes the two
controls agree, which `tests/test_learned_combination.py` records; the refitting
one is used anyway, because its validity does not rest on that agreement holding
for data this project has not seen.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from xinyenyana.evaluation import (
    CLASS_MEAN,
    HEADS,
    classification_metrics,
    head_operator,
    prepare_features,
    verification_metrics,
)
from xinyenyana.representations import BIRDNET_LAYERS

#: The seven layers, read as the average over time, shallowest first. The
#: mean-and-spread variants are the same layers read differently and the frozen
#: comparison already reports them; combining fourteen readings of seven layers
#: would put three copies of one vector into the concatenation.
LAYER_KEYS: tuple[str, ...] = tuple(f"{short}.mean" for short, _ in BIRDNET_LAYERS)

#: How many directions each layer contributes. Fixed in advance. A sweep over
#: this number would need a selection partition, and this benchmark has none
#: that does not leak a recording: the Stowell endpoints carry no recording or
#: session field, so an inner split of the enrolment clips cannot be made to
#: hold recordings apart.
COMMON_WIDTH = 16

#: Added to the within-identity scatter before it is inverted, as a multiple of
#: that scatter's own mean diagonal. RookID enrols 45 clips against a 1,024-wide
#: layer, so the scatter is singular there and the inverse is not defined
#: without it.
WITHIN_IDENTITY_RIDGE = 1e-3

RIDGE_LAMBDA = 1.0
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
SEED = 17

#: The arms, and what each one isolates. `learned combination` is the
#: architecture. `final layer projected` runs the same fit on the final
#: embedding alone, so a gain that is only the projection shows up as no gain
#: here. `layers concatenated` stacks the seven layers with nothing fitted, so a
#: gain that is only having more numbers shows up there. `final layer alone` is
#: the figure every other BirdNET result in this project is built on.
LEARNED_COMBINATION = "learned combination"
FINAL_LAYER_PROJECTED = "final layer projected"
LAYERS_CONCATENATED = "layers concatenated"
FINAL_LAYER_ALONE = "final layer alone"
ARMS: tuple[str, ...] = (
    LEARNED_COMBINATION,
    FINAL_LAYER_PROJECTED,
    LAYERS_CONCATENATED,
    FINAL_LAYER_ALONE,
)

#: The arms that fit something on the enrolment labels, and therefore need the
#: projection refitted under every permutation.
FITTED_ARMS: frozenset[str] = frozenset({LEARNED_COMBINATION, FINAL_LAYER_PROJECTED})


def _normalised(matrix: Any) -> Any:
    """Per-clip L2, the rule every BirdNET figure in this project uses."""

    import numpy as np

    values = np.asarray(matrix, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def total_scatter(features: Any) -> Any:
    """The scatter of these clips about their own mean, which no label changes.

    Hoisted out of `discriminant_projection` because the permutation control
    refits a thousand times per layer and this term is the same every time.
    """

    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    centred = values - values.mean(axis=0)
    return centred.T @ centred


def discriminant_projection(
    features: Any,
    labels: Sequence[str],
    *,
    width: int = COMMON_WIDTH,
    ridge: float = WITHIN_IDENTITY_RIDGE,
    total: Any = None,
) -> Any:
    """Directions that separate identities relative to how much one identity varies.

    The within-identity scatter is ridged by a multiple of its own mean diagonal
    and inverted; the between-identity scatter is taken through it; the leading
    eigenvectors come back. This is the ordinary discriminant construction and
    it has no free number beyond the two named above.

    Returned as a matrix to multiply features by, so the same object applies to
    the scored clips without refitting.
    """

    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    names = np.asarray(labels)
    unique = np.unique(names)
    if len(unique) < 2:
        raise ValueError("a discriminant needs at least two identities")
    overall = values.mean(axis=0)
    if total is None:
        centred = values - overall
        total = centred.T @ centred
    # The between-identity scatter as a product, because its rank is one less
    # than the number of identities and everything below uses that. Building it
    # as a square matrix and then taking a general eigendecomposition of a
    # 1,024-wide problem costs seconds per call, and the permutation control
    # makes this call a thousand times per layer.
    offsets = np.vstack(
        [
            np.sqrt(np.sum(names == identity)) * (values[names == identity].mean(axis=0) - overall)
            for identity in unique
        ]
    ).T
    within = total - offsets @ offsets.T
    floor = ridge * max(float(np.mean(np.diag(within))), 1e-30)
    within = within + floor * np.eye(within.shape[0])
    keep = min(width, len(unique) - 1, values.shape[1])
    # Whiten by the within-identity scatter, then take the leading directions of
    # the whitened between-identity scatter. These are the eigenvectors of
    # within^-1 @ between, computed through a Cholesky factor and a thin
    # singular value decomposition rather than a general eigendecomposition.
    factor = np.linalg.cholesky(within)
    whitened = np.linalg.solve(factor, offsets)
    left, _, _ = np.linalg.svd(whitened, full_matrices=False)
    directions = np.linalg.solve(factor.T, left[:, :keep])
    norms = np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-300)
    return np.ascontiguousarray(directions / norms)


def arm_keys(arm: str) -> tuple[str, ...]:
    """Which extracted layers an arm reads."""

    if arm in (FINAL_LAYER_ALONE, FINAL_LAYER_PROJECTED):
        return ("embedding.mean",)
    if arm in (LAYERS_CONCATENATED, LEARNED_COMBINATION):
        return LAYER_KEYS
    raise ValueError(f"unknown arm: {arm}")


def prepare_arm(
    arm: str,
    vectors: dict[str, Any],
    *,
    enrollment_rows: Sequence[int],
    kept_rows: Sequence[int],
) -> dict[str, Any]:
    """Everything about an arm that no label can change.

    The permutation control refits a thousand times per layer, and the
    normalisation, the enrolment mean and the total scatter are identical under
    every shuffle. Computing them once is the difference between a run that
    takes minutes and one that takes hours; it changes no number.

    ``kept_rows`` are the clips that will be scored. Only those are projected,
    because the rest are never read.
    """

    import numpy as np

    rows = list(enrollment_rows)
    kept = list(kept_rows)
    fitted = arm in FITTED_ARMS
    layers = []
    for key in arm_keys(arm):
        normalised = _normalised(vectors[key])
        centre = normalised[rows].mean(axis=0, keepdims=True)
        enrollment = normalised[rows] - centre
        # An arm that fits nothing hands the plain normalised vectors on, and
        # the scoring path standardises and centres them exactly as it does for
        # every other figure in this project. Centring them here as well would
        # make the scoring path renormalise centred vectors, which rescales each
        # clip; the cosine heads cannot see that and the ridge head can. It was
        # doing that, and it read the Great Tit final embedding at 0.475 where
        # the frozen probe reads 0.4437 on the same clips.
        layers.append(
            {
                "enrollment": enrollment,
                "kept": normalised[kept] - centre if fitted else normalised[kept],
                "total": total_scatter(enrollment) if fitted else None,
            }
        )
    return {"arm": arm, "layers": layers, "kept": np.asarray(kept)}


def apply_arm(prepared: dict[str, Any], labels: Sequence[str]) -> Any:
    """The arm's features over the kept clips, fitted on these enrolment labels."""

    import numpy as np

    arm = prepared["arm"]
    if arm not in FITTED_ARMS:
        return np.hstack([layer["kept"] for layer in prepared["layers"]])
    blocks = []
    for layer in prepared["layers"]:
        projection = discriminant_projection(layer["enrollment"], labels, total=layer["total"])
        blocks.append(layer["kept"] @ projection)
    return np.hstack(blocks)


def build_arm(
    arm: str,
    vectors: dict[str, Any],
    *,
    enrollment_rows: Sequence[int],
    labels: Sequence[str],
) -> Any:
    """One arm's features over every clip, with anything fitted fitted on enrolment.

    ``labels`` are the enrolment labels in the order of ``enrollment_rows``. A
    permutation passes shuffled labels here and gets a differently fitted
    projection, which is the point.
    """

    width = len(next(iter(vectors.values())))
    prepared = prepare_arm(arm, vectors, enrollment_rows=enrollment_rows, kept_rows=range(width))
    return apply_arm(prepared, labels)


def _accuracy(
    features: Any,
    *,
    enrollment_rows: Sequence[int],
    query_rows: Sequence[int],
    enrollment_labels: Any,
    query_labels: Any,
    identities: int,
    head: str,
) -> tuple[float, Any, Any]:
    import numpy as np

    train, query, _ = prepare_features(
        features[list(enrollment_rows)],
        features[list(query_rows)],
        "learned-combination",
        standardisation="per-clip L2",
    )
    operator = head_operator(head, train=train, query=query, ridge_lambda=RIDGE_LAMBDA)
    identity_matrix = np.eye(identities, dtype=np.float64)
    if head == CLASS_MEAN:
        counts = np.bincount(enrollment_labels, minlength=identities).astype(np.float64)
        identity_matrix = identity_matrix / np.maximum(counts, 1.0)[:, None]
    scores = operator @ identity_matrix[enrollment_labels]
    predicted = np.argmax(scores, axis=1)
    return float(np.mean(predicted == query_labels)), predicted, scores


def evaluate_arm(
    arm: str,
    vectors: dict[str, Any],
    *,
    records: list[Any],
    enrollment_condition: str,
    query_condition: str,
    head: str | Sequence[str],
    permutations: int = PERMUTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> Any:
    """One arm and one pairing, over one head or several, with the null the arm needs.

    A fitted arm refits its projection under every shuffle, so the p-value is
    about the whole procedure and not about the head alone. An unfitted arm has
    nothing to refit and its null is the ordinary one, which the result says.

    Several heads are scored together because the refit does not depend on the
    head. Doing them one at a time repeats a thousand refits per head and
    changes no number. A single head name returns one result, a sequence of
    names returns one per head.
    """

    import numpy as np

    identities = sorted({str(record.identity) for record in records})
    index_of = {name: number for number, name in enumerate(identities)}

    def rows_for(condition: str, split: str) -> list[int]:
        return [
            row
            for row, record in enumerate(records)
            if str(record.context.get("condition", "foreground")) == condition
            and record.split == split
        ]

    enrollment_rows = rows_for(enrollment_condition, "enrollment")
    query_rows = rows_for(query_condition, "query")
    if not enrollment_rows or not query_rows:
        raise ValueError(f"no clips for {enrollment_condition} to {query_condition}")
    enrollment_names = [str(records[row].identity) for row in enrollment_rows]
    enrollment_labels = np.asarray([index_of[name] for name in enrollment_names])
    query_labels = np.asarray([index_of[str(records[row].identity)] for row in query_rows])

    # Only the clips that are scored are projected, and everything a label
    # cannot change is computed once. The rows below index into the kept block
    # rather than into the endpoint.
    kept_rows = list(enrollment_rows) + list(query_rows)
    prepared = prepare_arm(arm, vectors, enrollment_rows=enrollment_rows, kept_rows=kept_rows)
    kept_enrollment = list(range(len(enrollment_rows)))
    kept_query = list(range(len(enrollment_rows), len(kept_rows)))

    single = isinstance(head, str)
    heads: list[str] = [head] if isinstance(head, str) else [str(name) for name in head]

    features = apply_arm(prepared, enrollment_names)
    measured = {
        name: _accuracy(
            features,
            enrollment_rows=kept_enrollment,
            query_rows=kept_query,
            enrollment_labels=enrollment_labels,
            query_labels=query_labels,
            identities=len(identities),
            head=name,
        )
        for name in heads
    }

    refits = arm in FITTED_ARMS
    rng = np.random.default_rng(seed)
    permuted: dict[str, list[float]] = {name: [] for name in heads}
    for _ in range(permutations):
        shuffled = rng.permutation(enrollment_labels)
        shuffled_features = (
            apply_arm(prepared, [identities[value] for value in shuffled]) if refits else features
        )
        for name in heads:
            permuted[name].append(
                _accuracy(
                    shuffled_features,
                    enrollment_rows=kept_enrollment,
                    query_rows=kept_query,
                    enrollment_labels=shuffled,
                    query_labels=query_labels,
                    identities=len(identities),
                    head=name,
                )[0]
            )

    groups = [np.flatnonzero(query_labels == number) for number in range(len(identities))]
    results: dict[str, Any] = {}
    for name in heads:
        _, predicted, scores = measured[name]
        classification = classification_metrics(query_labels, predicted)
        verification = verification_metrics(scores, query_labels)
        exceedances = sum(value >= classification["accuracy"] for value in permuted[name])
        correct = predicted == query_labels
        bootstrap_rng = np.random.default_rng(seed + 1)
        bootstrapped = []
        for _ in range(bootstrap_replicates):
            sampled = bootstrap_rng.integers(0, len(identities), size=len(identities))
            bootstrapped.append(
                float(np.mean(correct[np.concatenate([groups[number] for number in sampled])]))
            )
        results[name] = {
            "arm": arm,
            "head": name,
            "enrollment_condition": enrollment_condition,
            "query_condition": query_condition,
            "enrollment_calls": len(enrollment_rows),
            "query_calls": len(query_rows),
            "width": int(np.asarray(features).shape[1]),
            "classification": classification,
            "verification": verification,
            "identity_block_bootstrap_accuracy_95": [
                float(value) for value in np.quantile(bootstrapped, [0.025, 0.975])
            ],
            "identity_accuracy": {
                label: float(np.mean(correct[query_labels == number]))
                for number, label in enumerate(identities)
            },
            "permutation_control": {
                "method": (
                    "enrolment labels shuffled, projection refitted under each shuffle"
                    if refits
                    else "enrolment labels shuffled; this arm fits nothing to refit"
                ),
                "refits_the_representation": refits,
                "permutations": permutations,
                "mean_accuracy": float(np.mean(permuted[name])) if permutations else None,
                "maximum_accuracy": float(np.max(permuted[name])) if permutations else None,
                "p_value_plus_one": (exceedances + 1) / (permutations + 1),
            },
            "predictions": [
                {
                    "query_id": str(records[row].filename),
                    "actual_label": identities[query_labels[number]],
                    "predicted_label": identities[predicted[number]],
                }
                for number, row in enumerate(query_rows)
            ],
        }
    return results[heads[0]] if single else results


def run_learned_combination(
    endpoint: Any,
    vectors: dict[str, Any],
    *,
    permutations: int = PERMUTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Every arm, every head, every pairing the endpoint's conditions allow."""

    missing = [key for key in LAYER_KEYS if key not in vectors]
    if missing:
        raise ValueError(f"the extraction cache holds no {missing}")
    records = list(endpoint.records)
    conditions = {str(record.context.get("condition", "foreground")) for record in records}
    pairings = [("foreground", "foreground")]
    if "background" in conditions:
        pairings += [
            ("background", "background"),
            ("foreground", "background"),
            ("background", "foreground"),
        ]
    results: dict[str, Any] = {}
    for arm in ARMS:
        results[arm] = {head: {} for head in HEADS}
        for enrollment_condition, query_condition in pairings:
            by_head = evaluate_arm(
                arm,
                vectors,
                records=records,
                enrollment_condition=enrollment_condition,
                query_condition=query_condition,
                head=HEADS,
                permutations=permutations,
                bootstrap_replicates=bootstrap_replicates,
            )
            for head, value in by_head.items():
                results[arm][head][f"{enrollment_condition}_to_{query_condition}"] = value
            print(
                f"{endpoint.name} {arm} {enrollment_condition}->{query_condition} done",
                file=sys.stderr,
                flush=True,
            )
    family = len(ARMS) * len(HEADS)
    return {
        "experiment": "PA-LEARNED-COMBINATION",
        "endpoint": endpoint.name,
        "identities": len(endpoint.identities),
        "clips": len(records),
        "chance_accuracy": 1.0 / len(endpoint.identities),
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "config": {
            "layers": list(LAYER_KEYS),
            "common_width": COMMON_WIDTH,
            "within_identity_ridge": WITHIN_IDENTITY_RIDGE,
            "ridge_lambda": RIDGE_LAMBDA,
            "seed": SEED,
            "permutations": permutations,
            "bootstrap_replicates": bootstrap_replicates,
            "multiplicity_family": f"{len(ARMS)} arms x {len(HEADS)} heads within a pairing",
        },
        "arms": results,
        "p_bonferroni_over_family": {
            arm: {
                head: {
                    pairing: min(1.0, value["permutation_control"]["p_value_plus_one"] * family)
                    for pairing, value in by_head.items()
                }
                for head, by_head in by_arm.items()
            }
            for arm, by_arm in results.items()
        },
        "claim_boundary": (
            "Frozen BirdNET layers combined by a discriminant fitted on enrolment clips "
            "only. No encoder was retrained, no sequence model is involved, and the "
            "selection of the common width was fixed before any score here was read."
        ),
    }
