"""The shared evaluation primitives, used by every representation.

These are the statistics every result in this project is reported in, and they
are why a handcrafted layer and a pretrained embedding can sit in one table:
both go through the identical head, interval and null.

What is here:

- ``classification_metrics``: accuracy and macro recall.
- ``verification_metrics``: ROC-AUC, equal error rate, true-accept at a fixed
  false-accept rate.
- ``prepare_features`` and ``ridge_projection``: the fixed dual-ridge head.
- ``evaluate_endpoint``: one endpoint end to end, with the identity-block
  bootstrap and the label-permutation control.
- ``paired_difference_interval``: the identity-block interval on the difference
  between two endpoints, which is how a representation is compared with a
  nuisance control.

The identity-block bootstrap resamples identities, not clips. With five to
sixteen individuals that is the unit the interval has to respect; resampling
clips would treat several calls from one bird as independent evidence and
report an interval several times too narrow.
"""

from __future__ import annotations

from typing import Any


def classification_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    import numpy as np

    labels = np.unique(actual)
    recalls = [float(np.mean(predicted[actual == label] == label)) for label in labels]
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "macro_recall": float(np.mean(recalls)),
    }


def verification_metrics(scores: Any, actual: Any) -> dict[str, float]:
    import numpy as np

    targets = (np.arange(scores.shape[1])[None, :] == actual[:, None]).reshape(-1)
    values = scores.reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive = targets.astype(bool)
    positives = int(positive.sum())
    negatives = len(targets) - positives
    auc = (float(ranks[positive].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    descending = np.argsort(-values, kind="mergesort")
    ordered = positive[descending]
    tpr = np.cumsum(ordered) / positives
    fpr = np.cumsum(~ordered) / negatives
    fnr = 1 - tpr
    equal = int(np.argmin(np.abs(fpr - fnr)))
    permitted = tpr[fpr <= 0.01]
    return {
        "roc_auc": auc,
        "eer": float((fpr[equal] + fnr[equal]) / 2),
        "tar_at_far_0_01": float(permitted.max()) if len(permitted) else 0.0,
    }


CONSTANT_TARGET_TOLERANCE = 1e-9

PER_DIMENSION = "enrollment-only per-dimension mean/std"
PER_CLIP_L2 = "per-clip L2"


def standardisation_for(width: int) -> str:
    """The rule that suits a representation of this width.

    A one- or three-number control has no direction to normalise: dividing it
    by its own length makes every clip identical, or nearly so. A wide
    embedding does have one, and per-clip L2 is the rule the classifier uses on
    it. Both the classifier and the probe fitted to the same features call this,
    so a probe cannot silently describe features the classifier never saw.
    """

    if width <= 3:
        return PER_DIMENSION
    return PER_CLIP_L2


def prepare_features(
    train: Any, query: Any, representation: str, *, standardisation: str
) -> tuple[Any, Any, dict[str, Any]]:
    """Standardise enrollment and query features, then centre on the enrollment mean.

    The caller names the rule. It used to be inferred from the representation
    name when left unset, against a list of three names no caller produces, and
    the default that inference fell through to was the wrong one for a
    handcrafted layer: its dimensions are frequencies, decibels and seconds, and
    dividing such a vector by its own length mixes those units into one number,
    or, for a one-number control, makes every clip identical. A wrong default
    behind a branch that only fires when a caller forgets is worse than no
    default.
    """

    import numpy as np

    train = np.asarray(train, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    metadata: dict[str, Any] = {}
    if standardisation not in {PER_DIMENSION, PER_CLIP_L2}:
        raise ValueError(f"unknown standardisation: {standardisation}")
    if standardisation == PER_DIMENSION:
        mean = train.mean(axis=0, keepdims=True)
        scale = train.std(axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        train = (train - mean) / scale
        query = (query - mean) / scale
        metadata["standardization"] = PER_DIMENSION
    else:
        train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        metadata["standardization"] = PER_CLIP_L2
    center = train.mean(axis=0, keepdims=True)
    return train - center, query - center, {**metadata, "centering": "enrollment mean"}


def ridge_projection(train: Any, query: Any, ridge_lambda: float) -> Any:
    import numpy as np

    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    # The primal and dual expressions are algebraically identical. A duration
    # control has one feature but can have thousands of enrolment clips; solving
    # its sample kernel wastes quadratic memory without changing the model.
    if train.shape[1] < train.shape[0]:
        regularized = train.T @ train + ridge_lambda * np.eye(train.shape[1])
        return query @ np.linalg.solve(regularized, train.T)
    kernel = train @ train.T
    regularized = kernel + ridge_lambda * np.eye(kernel.shape[0])
    query_kernel = query @ train.T
    return np.linalg.solve(regularized, query_kernel.T).T


KERNEL_RIDGE = "kernel ridge to one-hot"
CLASS_MEAN = "mean cosine similarity to each identity"
NEAREST_NEIGHBOUR = "single nearest enrollment clip, cosine"
HEADS: tuple[str, ...] = (KERNEL_RIDGE, CLASS_MEAN, NEAREST_NEIGHBOUR)


def head_operator(name: str, *, train: Any, query: Any, ridge_lambda: float) -> Any:
    """The label-independent half of a head, as a matrix.

    Every head here reaches its scores as ``operator @ one_hot``, where
    ``one_hot`` carries the enrollment labels. Splitting a head this way is not
    a convenience: the permutation control shuffles those labels 999 times, and
    if a head recomputed its label-independent half each time the control would
    cost as much as the measurement and would tempt a shortcut that weakens it.

    The three heads differ in what they assume. Kernel ridge fits a regularised
    linear map and has one free number. The second scores an identity by the
    mean cosine similarity to its enrollment clips, fits nothing and has no free
    number. The third ignores class shape entirely and copies the label of the
    single closest enrollment clip. A finding that holds under all three is not
    a property of one classifier's inductive bias.

    The third head returns a one-hot score matrix, so its verification area is
    computed over ties and carries no information. Its accuracy does.
    """

    import numpy as np

    if name == KERNEL_RIDGE:
        return ridge_projection(train, query, ridge_lambda)
    if name not in {CLASS_MEAN, NEAREST_NEIGHBOUR}:
        raise ValueError(f"unknown head: {name}")
    train_unit = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query_unit = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query_unit @ train_unit.T
    if name == CLASS_MEAN:
        # The caller divides the one-hot columns by each identity's enrollment
        # count, so `similarity @ one_hot` is the mean cosine similarity to that
        # identity's clips. Permuting the labels does not change the counts, so
        # they are folded in once rather than 999 times.
        return similarity
    nearest = np.argmax(similarity, axis=1)
    selector = np.zeros_like(similarity)
    selector[np.arange(len(nearest)), nearest] = 1.0
    return selector


def evaluate_endpoint(
    *,
    records: list[dict[str, Any]],
    vectors: Any,
    representation: str,
    enrollment_condition: str,
    query_condition: str,
    ridge_lambda: float,
    seed: int,
    permutations: int,
    bootstrap_replicates: int,
    standardisation: str,
    head: str = KERNEL_RIDGE,
    extended_nulls: bool = True,
) -> dict[str, Any]:
    """One representation through one head, with its nulls and intervals.

    ``extended_nulls`` adds the within-session permutation and the three-stage
    bootstrap. A sweep that scores hundreds of candidate layers only to choose
    one turns it off, because those nulls are read on the reported figure and
    not on every candidate the selection rule looked at.
    """

    import numpy as np

    if head not in HEADS:
        raise ValueError(f"unknown head: {head}")

    identities = sorted({str(record["identity"]) for record in records})
    identity_index = {identity: index for index, identity in enumerate(identities)}
    train_rows = [
        index
        for index, record in enumerate(records)
        if record["condition"] == enrollment_condition and record["split"] == "enrollment"
    ]
    query_rows = [
        index
        for index, record in enumerate(records)
        if record["condition"] == query_condition and record["split"] == "query"
    ]
    train_ids = np.asarray(
        [identity_index[str(records[index]["identity"])] for index in train_rows]
    )
    query_ids = np.asarray(
        [identity_index[str(records[index]["identity"])] for index in query_rows]
    )
    train, query, processing = prepare_features(
        vectors[train_rows], vectors[query_rows], representation, standardisation=standardisation
    )
    projection = head_operator(head, train=train, query=query, ridge_lambda=ridge_lambda)
    identity_matrix = np.eye(len(identities), dtype=np.float64)
    if head == CLASS_MEAN:
        counts = np.bincount(train_ids, minlength=len(identities)).astype(np.float64)
        identity_matrix = identity_matrix / np.maximum(counts, 1.0)[:, None]
    one_hot = identity_matrix[train_ids]
    scores = projection @ one_hot
    predicted = np.argmax(scores, axis=1)
    classification = classification_metrics(query_ids, predicted)
    verification = verification_metrics(scores, query_ids)

    rng = np.random.default_rng(seed)
    permuted_accuracy: list[float] = []
    for _ in range(permutations):
        shuffled = rng.permutation(train_ids)
        shuffled_scores = projection @ identity_matrix[shuffled]
        permuted_accuracy.append(float(np.mean(np.argmax(shuffled_scores, axis=1) == query_ids)))
    exceedances = sum(value >= classification["accuracy"] for value in permuted_accuracy)

    groups = [np.flatnonzero(query_ids == index) for index in range(len(identities))]
    correct = predicted == query_ids
    bootstrap_rng = np.random.default_rng(seed + 1)
    bootstrap_accuracy: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled = bootstrap_rng.integers(0, len(identities), size=len(identities))
        rows = np.concatenate([groups[index] for index in sampled])
        bootstrap_accuracy.append(float(np.mean(correct[rows])))

    # --- vA3 additions. Every existing key above is untouched and every new
    # generator is seeded separately, so no number this function already
    # produced can move. ---

    # The chance column. 1/n is what a random guesser scores *on average*; it is
    # not what a random guesser scores. The permutation distribution is already
    # computed above, so its upper tail is free and is the figure a result has
    # to beat.
    permutation_tail = (
        {
            "percentile_95": float(np.quantile(permuted_accuracy, 0.95)),
            "percentile_99": float(np.quantile(permuted_accuracy, 0.99)),
            "uniform_chance": 1.0 / len(identities),
        }
        if permuted_accuracy
        else None
    )

    # The within-session null. Shuffling labels across the whole enrolment pile
    # breaks the clip-to-label link and the clip-to-recording link at the same
    # time, so clearing it does not separate identity from session. Shuffling
    # within a recording holds the recording constant. It is constructible only
    # where a recording holds more than one individual.
    sessions = [records[index].get("session") for index in train_rows] if extended_nulls else []
    within_session = None
    if all(session is not None for session in sessions) and sessions:
        session_array = np.asarray([str(session) for session in sessions])
        blocks = [np.flatnonzero(session_array == value) for value in np.unique(session_array)]
        usable = [block for block in blocks if len(np.unique(train_ids[block])) > 1]
        covered = int(sum(len(block) for block in usable))
        if usable:
            within_rng = np.random.default_rng(seed + 2)
            within_accuracy: list[float] = []
            for _ in range(permutations):
                shuffled = train_ids.copy()
                for block in usable:
                    shuffled[block] = within_rng.permutation(train_ids[block])
                shuffled_scores = projection @ identity_matrix[shuffled]
                within_accuracy.append(
                    float(np.mean(np.argmax(shuffled_scores, axis=1) == query_ids))
                )
            within_exceedances = sum(
                value >= classification["accuracy"] for value in within_accuracy
            )
            within_session = {
                "method": "enrolment labels shuffled within each recording",
                "permutations": permutations,
                "sessions": len(blocks),
                "sessions_holding_two_or_more_identities": len(usable),
                "enrollment_clips_in_those_sessions": covered,
                "enrollment_clips": len(train_rows),
                "mean_accuracy": float(np.mean(within_accuracy)),
                "percentile_95": float(np.quantile(within_accuracy, 0.95)),
                "maximum_accuracy": float(np.max(within_accuracy)),
                "p_value_plus_one": (within_exceedances + 1) / (permutations + 1),
            }
        else:
            within_session = {
                "constructible": False,
                "reason": "no recording in the enrolment split holds two or more individuals",
                "sessions": len(blocks),
                "enrollment_clips": len(train_rows),
            }
    elif sessions:
        within_session = {
            "constructible": False,
            "reason": "this endpoint's manifest carries no recording or session field",
        }

    # The hierarchical bootstrap. Resampling identities alone treats each
    # animal's recordings as fixed, which understates the uncertainty where an
    # animal contributes one or two. Three stages: identities, then recordings
    # within the sampled identity, then clips within the sampled recording.
    hierarchical = None
    query_sessions = (
        [records[index].get("session") for index in query_rows] if extended_nulls else []
    )
    if all(session is not None for session in query_sessions) and query_sessions:
        query_session_array = np.asarray([str(session) for session in query_sessions])
        by_identity: dict[int, list[Any]] = {}
        for index in range(len(identities)):
            rows_here = np.flatnonzero(query_ids == index)
            if not len(rows_here):
                continue
            by_identity[index] = [
                rows_here[query_session_array[rows_here] == value]
                for value in np.unique(query_session_array[rows_here])
            ]
        if by_identity:
            hierarchical_rng = np.random.default_rng(seed + 3)
            present = sorted(by_identity)
            draws: list[float] = []
            for _ in range(bootstrap_replicates):
                picked: list[Any] = []
                for choice in hierarchical_rng.integers(0, len(present), size=len(present)):
                    blocks_here = by_identity[present[int(choice)]]
                    for block_choice in hierarchical_rng.integers(
                        0, len(blocks_here), size=len(blocks_here)
                    ):
                        block = blocks_here[int(block_choice)]
                        picked.append(
                            block[hierarchical_rng.integers(0, len(block), size=len(block))]
                        )
                draws.append(float(np.mean(correct[np.concatenate(picked)])))
            hierarchical = {
                "method": (
                    "identities, then recordings within identity, then clips within recording"
                ),
                "replicates": bootstrap_replicates,
                "recordings_per_identity_median": float(
                    np.median([len(value) for value in by_identity.values()])
                ),
                "accuracy_95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            }

    predictions = [
        {
            "query_id": str(records[row]["filename"]),
            "actual_label": identities[query_ids[index]],
            "predicted_label": identities[predicted[index]],
            "scores": {
                identity: float(scores[index, identity_index[identity]]) for identity in identities
            },
        }
        for index, row in enumerate(query_rows)
    ]
    return {
        "enrollment_condition": enrollment_condition,
        "query_condition": query_condition,
        "enrollment_calls": len(train_rows),
        "query_calls": len(query_rows),
        "identities": len(identities),
        "head": head,
        "feature_processing": processing,
        "classification": classification,
        "verification": verification,
        "identity_block_bootstrap_accuracy_95": (
            [float(value) for value in np.quantile(bootstrap_accuracy, [0.025, 0.975])]
            if bootstrap_accuracy
            else None
        ),
        "identity_accuracy": {
            identity: float(np.mean(correct[query_ids == index]))
            for index, identity in enumerate(identities)
        },
        "permutation_upper_tail": permutation_tail,
        "within_session_permutation": within_session,
        "hierarchical_bootstrap": hierarchical,
        "permutation_control": {
            "method": "training labels shuffled across source clips",
            "permutations": permutations,
            "mean_accuracy": float(np.mean(permuted_accuracy)) if permuted_accuracy else None,
            "maximum_accuracy": float(np.max(permuted_accuracy)) if permuted_accuracy else None,
            "p_value_plus_one": (
                (exceedances + 1) / (permutations + 1) if permuted_accuracy else None
            ),
        },
        "predictions": predictions,
    }


def paired_difference_interval(
    foreground: dict[str, Any],
    background: dict[str, Any],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    import numpy as np

    foreground_values = foreground["identity_accuracy"]
    background_values = background["identity_accuracy"]
    identities = sorted(set(foreground_values) & set(background_values))
    if len(identities) != foreground["identities"]:
        raise ValueError("foreground/background bootstrap identities differ")
    differences = np.asarray(
        [foreground_values[identity] - background_values[identity] for identity in identities]
    )
    rng = np.random.default_rng(seed)
    values = [
        float(np.mean(differences[rng.integers(0, len(differences), size=len(differences))]))
        for _ in range(replicates)
    ]
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def leave_one_identity_out(
    *,
    features: Any,
    identities: Any,
    targets: dict[str, Any],
    categorical: frozenset[str] = frozenset(),
    standardisation: str,
    ridge_lambda: float,
) -> dict[str, dict[str, float]]:
    """Ask how much of a nuisance target can be read back out of the features.

    Ridge in the dual, fitted on every identity but one and scored on the
    held-out one, so a target that is predictable only because the same bird
    appears on both sides cannot look predictable here.

    A continuous target is scored by ``max(0, R²)``. A categorical target is
    scored by accuracy against the majority-class rate, because an R² over an
    arbitrary numeric code for a category means nothing. Either way the number
    can come out at the floor, which is what makes this a probe rather than a
    description.

    A target whose values never repeat across identities is refused: the held-out
    identity's value cannot appear in training, so no method could predict it and
    the probe would report a floor score whatever the features held.

    ``standardisation`` names the same rule the classifier applied to these
    features, and the caller gets it from :func:`standardisation_for`. Handing a
    narrow control per-clip L2 divides each row by its own length, which for a
    single number leaves the same value in every row, so the probe reports a
    floor score on a target the features are.
    """

    import numpy as np

    if standardisation not in {PER_DIMENSION, PER_CLIP_L2}:
        raise ValueError(f"unknown standardisation: {standardisation}")
    features = np.asarray(features, dtype=np.float64)
    identities = np.asarray(identities)
    unique_identities = sorted(set(identities.tolist()))
    if len(unique_identities) < 2:
        raise ValueError("leave-one-identity-out needs at least two identities")
    results: dict[str, dict[str, float]] = {}
    for name, raw in sorted(targets.items()):
        values = np.asarray(raw)
        if len(values) != len(features):
            raise ValueError(f"target {name!r} has {len(values)} values for {len(features)} rows")
        is_categorical = name in categorical
        if is_categorical:
            # Only a categorical target needs this. Its prediction is a choice
            # among the classes seen in training, so a class that never appears
            # outside one identity can never be chosen for that identity, and
            # the probe would report a floor score whatever the features held.
            # A continuous target does not have the problem: the regression
            # interpolates, and a held-out value need not have been seen.
            shared = {
                value
                for value in set(values.tolist())
                if len(
                    {
                        identity
                        for identity, other in zip(identities, values, strict=True)
                        if other == value
                    }
                )
                > 1
            }
            if not shared:
                raise ValueError(
                    f"target {name!r} takes no value shared by two identities, "
                    "so a held-out identity's value can never appear in training"
                )
        if is_categorical:
            classes = sorted(set(values.tolist()))
            target = np.eye(len(classes))[[classes.index(value) for value in values.tolist()]]
        else:
            numeric = np.asarray(values, dtype=np.float64)
            # A constant target has no variance to explain, and the R² below
            # would divide the residual by the floating-point dust left in
            # sum((y - mean)^2), which comes out at 1.0. That reads as a perfect
            # probe on a target that carries nothing. Refuse it here instead.
            if float(numeric.std()) <= CONSTANT_TARGET_TOLERANCE * max(
                1.0, float(np.abs(numeric.mean()))
            ):
                raise ValueError(
                    f"target {name!r} is constant, so there is nothing for a probe to predict"
                )
            target = numeric.reshape(-1, 1)
        prediction = np.zeros_like(target)
        for identity in unique_identities:
            test = identities == identity
            train = ~test
            train_x = features[train].copy()
            test_x = features[test].copy()
            if standardisation == PER_DIMENSION:
                mean = train_x.mean(axis=0, keepdims=True)
                scale = train_x.std(axis=0, keepdims=True)
                scale[scale == 0] = 1
                train_x = (train_x - mean) / scale
                test_x = (test_x - mean) / scale
            else:
                train_x /= np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
                test_x /= np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12)
            center = train_x.mean(axis=0, keepdims=True)
            train_x -= center
            test_x -= center
            target_mean = target[train].mean(axis=0, keepdims=True)
            centered_target = target[train] - target_mean
            if train_x.shape[1] < train_x.shape[0]:
                weights = np.linalg.solve(
                    train_x.T @ train_x + ridge_lambda * np.eye(train_x.shape[1]),
                    train_x.T @ centered_target,
                )
                prediction[test] = test_x @ weights + target_mean
            else:
                dual = np.linalg.solve(
                    train_x @ train_x.T + ridge_lambda * np.eye(int(train.sum())),
                    centered_target,
                )
                prediction[test] = test_x @ train_x.T @ dual + target_mean
        if is_categorical:
            truth = np.argmax(target, axis=1)
            counts = np.bincount(truth, minlength=target.shape[1])
            results[name] = {
                "accuracy": float(np.mean(np.argmax(prediction, axis=1) == truth)),
                "majority_class_rate": float(counts.max() / counts.sum()),
                "classes": float(target.shape[1]),
            }
        else:
            flat = target[:, 0]
            denominator = float(np.sum((flat - flat.mean()) ** 2))
            r_squared = 1 - float(np.sum((flat - prediction[:, 0]) ** 2)) / denominator
            results[name] = {"r_squared": max(0.0, r_squared)}
    return results
