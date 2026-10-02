"""Architecture two: the time inside a call, and a metric learned without it.

Every BirdNET figure in this project comes from averaging a layer over time. The
extractor has been keeping the time positions themselves since 2026-09-13 and
nothing has read them. This module asks what they are worth: a small attentive
pooling learned over those positions, against the average of the same positions,
on the same clips and splits.

What makes the comparison safe is where the model is fitted. The identities are
split into a development group and an evaluation group by a fixed salted rule
before anything is read. The model sees only development clips, at any split.
Every number reported comes from evaluation identities the model has never seen,
so the downstream test conditions on a representation fitted without them, and
the ordinary label permutation is a valid control for it.

The gradients here are written out rather than taken from a framework, so that
the project's numerical environment stays numpy and the tests can check them.
`tests/test_sequence_metric.py` compares every one against a finite difference.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

#: Layers whose retained sequence carries more than one time position per
#: three-second segment. `post_convolution` and `embedding` carry one position
#: per segment, which is a sequence over segments and not over the call, so a
#: pooling learned on them would have almost nothing to weight.
SEQUENCE_LAYERS: tuple[str, ...] = (
    "stage1.sequence",
    "stage2.sequence",
    "stage3.sequence",
    "stage4.sequence",
    "post_activation.sequence",
)

#: The width of the learned embedding, the temperature of the contrastive loss,
#: the optimiser's step, how many passes, and the batch. All fixed here, before
#: any score on real data was read, and none of them is chosen by a result.
EMBEDDING_WIDTH = 64
TEMPERATURE = 0.07
LEARNING_RATE = 3e-3
PASSES = 60
BATCH = 64
SEED = 17

#: How the identities are divided. The salt fixes the division before any score
#: is read and makes it reproducible without storing a list.
IDENTITY_SALT = "xyy-architecture-two-20260914"
DEVELOPMENT_SHARE = 0.4
STRANGER_SHARE = 0.3

#: The stranger budget the open-set threshold is chosen at, on development
#: identities, and reported at, on evaluation identities.
STRANGER_BUDGET = 0.10
NO_FEW_SHOT_PERMUTATION = (
    "not run: this arm measures how accuracy falls as the gallery shrinks, and "
    "whether the representation carries identity at all is answered by the "
    "closed-set arm on the same probe clips"
)

#: Enrolment sizes for the few-shot arm.
SHOTS: tuple[int, ...] = (1, 2, 5)


def _rank(identity: str, purpose: str) -> str:
    return hashlib.sha256(f"{IDENTITY_SALT}:{purpose}:{identity}".encode()).hexdigest()


def divide_identities(identities: Sequence[str]) -> dict[str, list[str]]:
    """Development, enrolled and stranger groups, from the names alone.

    Nothing about any clip, vector or score reaches this. The development group
    is what the model is fitted on; the enrolled and stranger groups are drawn
    from what is left and are never seen by the fit. Each of development and
    evaluation is split again into enrolled and stranger, so the open-set
    threshold can be chosen on one group and reported on the other.
    """

    names = sorted(set(identities))
    if len(names) < 4:
        raise ValueError(f"an identity division needs at least four identities, not {len(names)}")
    ordered = sorted(names, key=lambda name: _rank(name, "development"))
    take = max(2, round(len(names) * DEVELOPMENT_SHARE))
    take = min(take, len(names) - 2)
    development, evaluation = ordered[:take], ordered[take:]
    groups: dict[str, list[str]] = {}
    for label, members in (("development", development), ("evaluation", evaluation)):
        by_stranger = sorted(members, key=lambda name: _rank(name, "stranger"))
        strangers = max(1, round(len(members) * STRANGER_SHARE))
        strangers = min(strangers, len(members) - 1)
        groups[f"{label}_strangers"] = sorted(by_stranger[:strangers])
        groups[f"{label}_enrolled"] = sorted(by_stranger[strangers:])
    groups["development"] = sorted(development)
    groups["evaluation"] = sorted(evaluation)
    return groups


def pool(sequence: Any, attention_vector: Any) -> tuple[Any, dict[str, Any]]:
    """Attentive statistics over the time positions of one clip.

    The weights come from the positions themselves through one vector, so the
    model has as many attention parameters as the layer has channels. A flat
    weighting reproduces the average this project already uses, which puts the
    baseline inside the model rather than beside it.
    """

    import numpy as np

    positions = np.asarray(sequence, dtype=np.float64)
    scores = positions @ attention_vector
    scores = scores - scores.max()
    weights = np.exp(scores)
    weights = weights / weights.sum()
    mean = weights @ positions
    variance = weights @ (positions - mean) ** 2
    spread = np.sqrt(variance + 1e-8)
    return np.concatenate([mean, spread]), {
        "weights": weights,
        "positions": positions,
        "mean": mean,
        "spread": spread,
    }


def pool_batch(sequences: Sequence[Any], attention_vector: Any) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    pooled, caches = [], []
    for sequence in sequences:
        vector, cache = pool(sequence, attention_vector)
        pooled.append(vector)
        caches.append(cache)
    return np.asarray(pooled), caches


def project(pooled: Any, weight: Any) -> tuple[Any, dict[str, Any]]:
    """One linear map to the embedding width, then L2, which is what is compared."""

    import numpy as np

    raw = pooled @ weight.T
    norms = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    return raw / norms, {"raw": raw, "norms": norms}


def contrastive_loss(
    embeddings: Any, positives: Any, *, temperature: float = TEMPERATURE
) -> tuple[float, Any]:
    """Cross-entropy over cosine similarities, against a mask of allowed positives.

    ``positives`` is a boolean matrix: entry (i, j) is True when clip j is a
    positive for clip i. Which pairs are positive is where the hierarchy enters,
    and it is decided by the caller, not here. A row with no positive is skipped
    rather than contributing a loss with no target.

    Returns the loss and its gradient with respect to the embeddings.
    """

    import numpy as np

    rows = np.asarray(embeddings, dtype=np.float64)
    count = len(rows)
    similarity = rows @ rows.T / temperature
    np.fill_diagonal(similarity, -np.inf)
    mask = np.asarray(positives, dtype=bool).copy()
    np.fill_diagonal(mask, False)
    usable = mask.any(axis=1)
    if not usable.any():
        return 0.0, np.zeros_like(rows)

    shifted = (
        similarity - np.max(np.where(np.isfinite(similarity), similarity, -np.inf), axis=1)[:, None]
    )
    exponential = np.where(np.isfinite(similarity), np.exp(shifted), 0.0)
    denominator = exponential.sum(axis=1)
    numerator = np.where(mask, exponential, 0.0).sum(axis=1)
    loss_rows = -np.log(np.maximum(numerator, 1e-300) / np.maximum(denominator, 1e-300))
    loss = float(loss_rows[usable].mean())

    softmax = exponential / np.maximum(denominator, 1e-300)[:, None]
    positive_softmax = np.where(mask, exponential, 0.0) / np.maximum(numerator, 1e-300)[:, None]
    gradient_similarity = np.zeros((count, count))
    gradient_similarity[usable] = (softmax - positive_softmax)[usable]
    gradient_similarity /= usable.sum() * temperature
    symmetric = gradient_similarity + gradient_similarity.T
    return loss, symmetric @ rows


def backward(
    gradient_embeddings: Any,
    *,
    pooled: Any,
    weight: Any,
    projection_cache: dict[str, Any],
    pooling_caches: list[dict[str, Any]],
) -> tuple[Any, Any]:
    """The gradient of the loss with respect to the projection and the attention.

    Written out because the project's numerical environment is numpy. Every term
    here is checked against a finite difference in the tests, which is the only
    reason to trust it.
    """

    import numpy as np

    raw, norms = projection_cache["raw"], projection_cache["norms"]
    upstream = np.asarray(gradient_embeddings, dtype=np.float64)
    inner = np.sum(upstream * raw, axis=1, keepdims=True)
    gradient_raw = upstream / norms - raw * inner / norms**3
    gradient_weight = gradient_raw.T @ pooled
    gradient_pooled = gradient_raw @ weight

    width = pooled.shape[1] // 2
    gradient_attention = np.zeros(pooling_caches[0]["positions"].shape[1])
    for index, cache in enumerate(pooling_caches):
        positions = cache["positions"]
        weights, mean, spread = cache["weights"], cache["mean"], cache["spread"]
        gradient_mean = gradient_pooled[index, :width]
        gradient_spread = gradient_pooled[index, width:]
        gradient_variance = gradient_spread / (2.0 * spread)
        offsets = positions - mean
        # d/d weights, holding the mean's own dependence on the weights in mind:
        # the variance term's derivative through the mean cancels, because
        # sum(weights * offsets) is zero.
        through_mean = offsets @ gradient_mean
        through_variance = (offsets**2) @ gradient_variance
        gradient_weights = through_mean + through_variance
        centred = gradient_weights - weights @ gradient_weights
        gradient_scores = weights * centred
        gradient_attention += positions.T @ gradient_scores
    return gradient_weight, gradient_attention


def positive_mask(
    identities: Sequence[str], groups: Sequence[str] | None, *, cross_group_only: bool
) -> Any:
    """Which pairs the contrastive loss is allowed to pull together.

    The hierarchy is: two clips of one individual recorded in different sessions
    are the pair worth learning from; two clips of one individual from the same
    session share a recording as well as a bird, so pulling them together
    teaches the recording. Where the endpoint publishes a session or recording
    label this restricts positives to different sessions. Where it does not, the
    restriction cannot be applied and the caller is told so rather than being
    given a mask that silently ignores the level.
    """

    import numpy as np

    names = np.asarray(identities)
    same_identity = names[:, None] == names[None, :]
    if not cross_group_only:
        return same_identity
    if groups is None:
        raise ValueError("this endpoint carries no session label, so pairs cannot cross one")
    sessions = np.asarray(groups)
    return same_identity & (sessions[:, None] != sessions[None, :])


def train(
    sequences: Sequence[Any],
    identities: Sequence[str],
    groups: Sequence[str] | None,
    *,
    cross_group_only: bool,
    width: int = EMBEDDING_WIDTH,
    passes: int = PASSES,
    batch: int = BATCH,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit the attention and the projection on these clips and nothing else.

    Adam with the fixed step above, a fixed number of passes and no early
    stopping. Stopping on a score would need a partition held out from these
    clips for that purpose, and the one held out here is held out from the fit
    entirely and is where the result is read.
    """

    import numpy as np

    if not sequences:
        raise ValueError("nothing to fit")
    channels = int(np.asarray(sequences[0]).shape[1])
    generator = np.random.default_rng(seed)
    attention = np.zeros(channels)
    weight = generator.normal(size=(width, 2 * channels)) / np.sqrt(2 * channels)
    mask = positive_mask(identities, groups, cross_group_only=cross_group_only)
    usable = np.flatnonzero(mask.sum(axis=1) - np.diag(mask))
    if not len(usable):
        raise ValueError("no clip has a positive partner under this hierarchy")

    moments = {
        "weight": (np.zeros_like(weight), np.zeros_like(weight)),
        "attention": (np.zeros_like(attention), np.zeros_like(attention)),
    }
    history: list[float] = []
    step_number = 0
    order = np.arange(len(sequences))
    for _ in range(passes):
        generator.shuffle(order)
        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            if len(rows) < 4:
                continue
            chosen = [sequences[row] for row in rows]
            submask = mask[np.ix_(rows, rows)]
            pooled, caches = pool_batch(chosen, attention)
            embeddings, projection_cache = project(pooled, weight)
            loss, gradient_embeddings = contrastive_loss(embeddings, submask)
            if loss == 0.0:
                continue
            gradient_weight, gradient_attention = backward(
                gradient_embeddings,
                pooled=pooled,
                weight=weight,
                projection_cache=projection_cache,
                pooling_caches=caches,
            )
            step_number += 1
            for name, parameter, gradient in (
                ("weight", weight, gradient_weight),
                ("attention", attention, gradient_attention),
            ):
                first, second = moments[name]
                first *= 0.9
                first += 0.1 * gradient
                second *= 0.999
                second += 0.001 * gradient**2
                corrected_first = first / (1 - 0.9**step_number)
                corrected_second = second / (1 - 0.999**step_number)
                parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            history.append(loss)
    return {
        "attention": attention,
        "weight": weight,
        "loss_first": history[0] if history else None,
        "loss_last": history[-1] if history else None,
        "steps": step_number,
        "clips": len(sequences),
        "cross_session_positives": cross_group_only,
    }


def embed(sequences: Sequence[Any], fit: dict[str, Any]) -> Any:
    """The learned representation of every clip, under a fit made without them."""

    pooled, _ = pool_batch(sequences, fit["attention"])
    embeddings, _ = project(pooled, fit["weight"])
    return embeddings


def _scores(gallery: Any, gallery_labels: Sequence[str], probes: Any) -> tuple[Any, list[str]]:
    """Cosine to each enrolled identity's mean, and the identity names in order."""

    import numpy as np

    names = sorted(set(gallery_labels))
    labels = np.asarray(gallery_labels)
    centres = np.vstack([gallery[labels == name].mean(axis=0) for name in names])
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
    return probes @ centres.T, names


def closed_set(
    gallery: Any,
    gallery_labels: Sequence[str],
    probes: Any,
    probe_labels: Sequence[str],
    *,
    permutations: int = 999,
    bootstrap_replicates: int = 2000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Which enrolled identity is this, with the ordinary label permutation.

    The permutation is valid here because the representation was fitted without
    any of these identities: it is fixed before the labels are shuffled, in the
    way `evaluate_endpoint`'s control assumes and a representation fitted on the
    evaluation identities would violate.
    """

    import numpy as np

    scores, names = _scores(gallery, gallery_labels, probes)
    index_of = {name: number for number, name in enumerate(names)}
    truth = np.asarray([index_of[str(name)] for name in probe_labels])
    predicted = np.argmax(scores, axis=1)
    correct = predicted == truth
    accuracy = float(correct.mean())

    generator = np.random.default_rng(seed)
    gallery_index = np.asarray([index_of[str(name)] for name in gallery_labels])
    permuted: list[float] = []
    for _ in range(permutations):
        shuffled = generator.permutation(gallery_index)
        shuffled_names = [names[value] for value in shuffled]
        shuffled_scores, shuffled_order = _scores(gallery, shuffled_names, probes)
        lookup = {name: number for number, name in enumerate(shuffled_order)}
        shuffled_truth = np.asarray([lookup[str(name)] for name in probe_labels])
        permuted.append(float(np.mean(np.argmax(shuffled_scores, axis=1) == shuffled_truth)))
    exceedances = sum(value >= accuracy for value in permuted)
    control: dict[str, Any] | None = None
    if permutations:
        control = {
            "method": "gallery labels shuffled; the representation is fixed before the shuffle",
            "permutations": permutations,
            "mean_accuracy": float(np.mean(permuted)),
            "p_value_plus_one": (exceedances + 1) / (permutations + 1),
        }

    groups = [np.flatnonzero(truth == number) for number in range(len(names))]
    bootstrap_generator = np.random.default_rng(seed + 1)
    bootstrapped = []
    for _ in range(bootstrap_replicates):
        sampled = bootstrap_generator.integers(0, len(names), size=len(names))
        bootstrapped.append(
            float(np.mean(correct[np.concatenate([groups[number] for number in sampled])]))
        )
    return {
        "identities": len(names),
        "chance_accuracy": 1.0 / len(names),
        "gallery_clips": len(gallery),
        "probe_clips": len(probes),
        "accuracy": accuracy,
        "identity_block_bootstrap_accuracy_95": [
            float(value) for value in np.quantile(bootstrapped, [0.025, 0.975])
        ],
        "permutation_control": control,
    }


def stranger_threshold(
    gallery: Any,
    gallery_labels: Sequence[str],
    probes: Any,
    is_stranger: Sequence[bool],
    *,
    budget: float = STRANGER_BUDGET,
) -> float:
    """The highest-score cut that lets through at most ``budget`` of the strangers.

    Chosen on one group of identities and applied to another. Returned as a
    number so that what was chosen where is visible in the result rather than
    implied by an accuracy.
    """

    import numpy as np

    scores, _ = _scores(gallery, gallery_labels, probes)
    best = scores.max(axis=1)
    stranger_scores = np.sort(best[np.asarray(is_stranger, dtype=bool)])
    if not len(stranger_scores):
        raise ValueError("a stranger budget needs stranger clips to measure it on")
    keep = int(np.floor(budget * len(stranger_scores)))
    index = max(0, len(stranger_scores) - keep - 1)
    return float(stranger_scores[index])


def open_set(
    gallery: Any,
    gallery_labels: Sequence[str],
    probes: Any,
    probe_labels: Sequence[str],
    is_stranger: Sequence[bool],
    *,
    threshold: float,
) -> dict[str, Any]:
    """What the fixed threshold does on identities it was not chosen on."""

    import numpy as np

    scores, names = _scores(gallery, gallery_labels, probes)
    index_of = {name: number for number, name in enumerate(names)}
    stranger = np.asarray(is_stranger, dtype=bool)
    best = scores.max(axis=1)
    accepted = best >= threshold
    predicted = np.argmax(scores, axis=1)
    known = ~stranger
    truth = np.asarray(
        [
            index_of[str(name)] if not flag else -1
            for name, flag in zip(probe_labels, stranger, strict=True)
        ]
    )
    correct_known = accepted & known & (predicted == truth)
    return {
        "threshold": threshold,
        "declared_stranger_budget": STRANGER_BUDGET,
        "known_clips": int(known.sum()),
        "stranger_clips": int(stranger.sum()),
        "stranger_acceptance": float(accepted[stranger].mean()) if stranger.any() else None,
        "known_acceptance": float(accepted[known].mean()) if known.any() else None,
        "correct_known_acceptance": float(correct_known.sum() / max(known.sum(), 1)),
    }


def _gallery(
    embeddings: Any,
    records: Sequence[Any],
    rows: Sequence[int],
    *,
    shots: int | None,
    salt: str,
) -> tuple[Any, list[str]]:
    """The enrolled clips, optionally cut to a fixed number per identity.

    Which clips survive a cut is decided by hashing the filename, so the
    few-shot arm is the same clips every time it runs and no score chooses them.
    """

    import numpy as np

    chosen = list(rows)
    if shots is not None:
        by_identity: dict[str, list[int]] = {}
        for row in chosen:
            by_identity.setdefault(str(records[row].identity), []).append(row)
        chosen = []
        for _identity, members in sorted(by_identity.items()):
            ordered = sorted(
                members,
                key=lambda row: hashlib.sha256(
                    f"{IDENTITY_SALT}:{salt}:{records[row].filename}".encode()
                ).hexdigest(),
            )
            chosen.extend(ordered[:shots])
        chosen.sort()
    return np.asarray(embeddings)[chosen], [str(records[row].identity) for row in chosen]


def run_sequence_metric(
    endpoint: Any,
    sequences: dict[str, Any],
    means: dict[str, Any],
    *,
    sessions: Sequence[str] | None,
    permutations: int = 999,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Architecture two on one endpoint: every sequence layer, learned and averaged.

    ``sessions`` is a label per clip that says which recording session it came
    from, or None where the endpoint publishes none. With it, the contrastive
    loss is restricted to pairs of one individual from different sessions. Its
    absence is recorded rather than worked around, because a loss that pulls
    same-session clips together on an endpoint with no session label is learning
    the recording and cannot say it is not.
    """

    import numpy as np

    records = list(endpoint.records)
    foreground = [
        row
        for row, record in enumerate(records)
        if str(record.context.get("condition", "foreground")) == "foreground"
    ]
    groups = divide_identities([str(records[row].identity) for row in foreground])
    development = set(groups["development"])
    development_rows = [row for row in foreground if str(records[row].identity) in development]
    if not development_rows:
        raise ValueError("the development group holds no clips")

    def rows_in(names: set[str], split: str | None, condition: str = "foreground") -> list[int]:
        return [
            row
            for row, record in enumerate(records)
            if str(record.context.get("condition", "foreground")) == condition
            and str(record.identity) in names
            and (split is None or record.split == split)
        ]

    enrolled = set(groups["evaluation_enrolled"])
    strangers = set(groups["evaluation_strangers"])
    gallery_rows = rows_in(enrolled, "enrollment")
    probe_rows = rows_in(enrolled, "query") + rows_in(strangers, "query")
    closed_probe_rows = rows_in(enrolled, "query")
    if not gallery_rows or not closed_probe_rows:
        raise ValueError("the evaluation group holds no enrolment or no query clips")

    calibration_gallery = rows_in(set(groups["development_enrolled"]), "enrollment")
    calibration_probes = rows_in(set(groups["development_enrolled"]), "query") + rows_in(
        set(groups["development_strangers"]), "query"
    )

    # The validity control this project applies to every representation that
    # carries a claim: score the same evaluation identities on the ambient
    # recordings of their locations, with no bird in them. The model is still
    # fitted on development foreground clips alone; only what it is scored on
    # changes. An endpoint that publishes no background cannot take it.
    background_gallery = rows_in(enrolled, "enrollment", "background")
    background_probes = rows_in(enrolled, "query", "background")

    arms: dict[str, Any] = {}
    for layer in SEQUENCE_LAYERS:
        if layer not in sequences:
            continue
        clips = sequences[layer]
        width = int(np.asarray(clips[0]).shape[1])
        mean_key = layer.replace(".sequence", ".mean")
        fit = train(
            [clips[row] for row in development_rows],
            [str(records[row].identity) for row in development_rows],
            None if sessions is None else [str(sessions[row]) for row in development_rows],
            cross_group_only=sessions is not None,
            passes=PASSES,
            batch=BATCH,
        )
        learned = embed([clips[row] for row in range(len(records))], fit)
        averaged = np.asarray(means[mean_key], dtype=np.float64)
        averaged = averaged / np.maximum(np.linalg.norm(averaged, axis=1, keepdims=True), 1e-12)
        # The unweighted pooling of the same positions the learned arm sees.
        # Without it the comparison confounds three things at once: the
        # attention, the learned projection, and the fact that `<layer>.mean`
        # averages the whole time-frequency grid while a retained sequence is
        # cropped to the real-audio fraction of its last segment. This arm fits
        # nothing and differs from the learned one only by the attention and the
        # projection; `<layer>.mean` is kept beside it because it is the
        # project's existing reading of that layer.
        plain = np.vstack([pool(clips[row], np.zeros(width))[0] for row in range(len(records))])
        plain = plain / np.maximum(np.linalg.norm(plain, axis=1, keepdims=True), 1e-12)
        for name, vectors in (
            ("learned sequence", learned),
            ("sequence average", plain),
            ("layer average", averaged),
        ):
            gallery, gallery_labels = _gallery(
                vectors, records, gallery_rows, shots=None, salt="all"
            )
            entry: dict[str, Any] = {
                "fit": {
                    key: value for key, value in fit.items() if key not in ("attention", "weight")
                },
                "closed_set": closed_set(
                    gallery,
                    gallery_labels,
                    np.asarray(vectors)[closed_probe_rows],
                    [str(records[row].identity) for row in closed_probe_rows],
                    permutations=permutations,
                    bootstrap_replicates=bootstrap_replicates,
                ),
                "few_shot": {},
            }
            calibration_strangers = [
                row
                for row in calibration_probes
                if str(records[row].identity) in set(groups["development_strangers"])
            ]
            evaluation_strangers = [
                row for row in probe_rows if str(records[row].identity) in strangers
            ]
            if not (calibration_gallery and calibration_strangers and evaluation_strangers):
                entry["open_set"] = {
                    "unavailable": (
                        "the open set needs enrolment clips and stranger query clips in both "
                        "the development and the evaluation group; this endpoint's division "
                        f"gives {len(calibration_gallery)} calibration gallery clips, "
                        f"{len(calibration_strangers)} calibration stranger clips and "
                        f"{len(evaluation_strangers)} evaluation stranger clips"
                    )
                }
            else:
                calibration_vectors, calibration_labels = _gallery(
                    vectors, records, calibration_gallery, shots=None, salt="all"
                )
                threshold = stranger_threshold(
                    calibration_vectors,
                    calibration_labels,
                    np.asarray(vectors)[calibration_probes],
                    [
                        str(records[row].identity) in set(groups["development_strangers"])
                        for row in calibration_probes
                    ],
                )
                entry["open_set"] = open_set(
                    gallery,
                    gallery_labels,
                    np.asarray(vectors)[probe_rows],
                    [str(records[row].identity) for row in probe_rows],
                    [str(records[row].identity) in strangers for row in probe_rows],
                    threshold=threshold,
                )
                entry["open_set"]["threshold_chosen_on"] = "development identities"
            if background_gallery and background_probes:
                background_vectors, background_labels = _gallery(
                    vectors, records, background_gallery, shots=None, salt="all"
                )
                entry["background_only"] = closed_set(
                    background_vectors,
                    background_labels,
                    np.asarray(vectors)[background_probes],
                    [str(records[row].identity) for row in background_probes],
                    permutations=permutations,
                    bootstrap_replicates=bootstrap_replicates,
                )
            else:
                entry["background_only"] = None
            for shots in SHOTS:
                few_gallery, few_labels = _gallery(
                    vectors, records, gallery_rows, shots=shots, salt=f"shots{shots}"
                )
                if len(set(few_labels)) < 2:
                    continue
                # No permutation here, for the reason the result carries with it.
                measurement = closed_set(
                    few_gallery,
                    few_labels,
                    np.asarray(vectors)[closed_probe_rows],
                    [str(records[row].identity) for row in closed_probe_rows],
                    permutations=0,
                    bootstrap_replicates=bootstrap_replicates,
                )
                measurement["permutation_control_absent_because"] = NO_FEW_SHOT_PERMUTATION
                entry["few_shot"][str(shots)] = measurement
            arms.setdefault(layer, {})[name] = entry

    return {
        "experiment": "PA-SEQUENCE-METRIC",
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "identity_groups": groups,
        "session_label": "published" if sessions is not None else "absent",
        "background_control": bool(background_gallery and background_probes),
        "config": {
            "layers": list(SEQUENCE_LAYERS),
            "embedding_width": EMBEDDING_WIDTH,
            "temperature": TEMPERATURE,
            "learning_rate": LEARNING_RATE,
            "passes": PASSES,
            "batch": BATCH,
            "seed": SEED,
            "identity_salt": IDENTITY_SALT,
            "development_share": DEVELOPMENT_SHARE,
            "stranger_share": STRANGER_SHARE,
            "declared_stranger_budget": STRANGER_BUDGET,
            "shots": list(SHOTS),
            "permutations": permutations,
            "bootstrap_replicates": bootstrap_replicates,
        },
        "arms": arms,
        "claim_boundary": (
            "A pooling and a linear metric learned over frozen BirdNET time positions, "
            "fitted on development identities only. No encoder was retrained. The "
            "stranger threshold was chosen on development identities and applied "
            "unchanged to evaluation identities the fit never saw."
        ),
    }
