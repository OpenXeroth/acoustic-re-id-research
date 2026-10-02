"""Tests for the shared evaluation primitives.

These use synthetic vectors, so a failure points at the statistics rather than
at an endpoint. The identity-block test checks the property the module's
docstring claims, which is the one that would silently overstate every interval
in the programme if it were wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xinyenyana.evaluation import (
    CLASS_MEAN,
    HEADS,
    KERNEL_RIDGE,
    NEAREST_NEIGHBOUR,
    PER_CLIP_L2,
    PER_DIMENSION,
    classification_metrics,
    evaluate_endpoint,
    head_operator,
    leave_one_identity_out,
    paired_difference_interval,
    prepare_features,
    ridge_projection,
    standardisation_for,
    verification_metrics,
)


def test_classification_metrics_on_a_known_case() -> None:
    actual = np.array(["a", "a", "b", "b"])
    predicted = np.array(["a", "b", "b", "b"])
    metrics = classification_metrics(actual, predicted)
    assert metrics["accuracy"] == pytest.approx(0.75)
    # a: 1 of 2 correct, b: 2 of 2, so the macro mean is 0.75
    assert metrics["macro_recall"] == pytest.approx(0.75)


def test_classification_macro_recall_differs_from_accuracy_when_classes_are_unbalanced() -> None:
    actual = np.array(["a", "b", "b", "b", "b"])
    predicted = np.array(["b", "b", "b", "b", "b"])
    metrics = classification_metrics(actual, predicted)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["macro_recall"] == pytest.approx(0.5)


def test_verification_metrics_separate_a_perfect_and_a_useless_scorer() -> None:
    # Two queries, two classes, correct class scored far higher.
    perfect = np.array([[9.0, 0.0], [0.0, 9.0]])
    actual = np.array([0, 1])
    good = verification_metrics(perfect, actual)
    assert good["roc_auc"] == pytest.approx(1.0)
    assert good["eer"] == pytest.approx(0.0)

    inverted = np.array([[0.0, 9.0], [9.0, 0.0]])
    bad = verification_metrics(inverted, actual)
    assert bad["roc_auc"] == pytest.approx(0.0)
    assert bad["eer"] == pytest.approx(1.0)

    tied = np.array([[1.0, 1.0], [1.0, 1.0]])
    chance = verification_metrics(tied, actual)
    assert chance["roc_auc"] == pytest.approx(0.5)


def test_ridge_projection_recovers_the_training_targets_at_a_small_penalty() -> None:
    train = np.eye(3)
    query = np.eye(3)
    projected = ridge_projection(train, query, 1e-6)
    assert projected.shape == (3, 3)
    assert np.allclose(projected, np.eye(3), atol=1e-4)


def test_prepare_features_centres_on_the_enrollment_mean() -> None:
    train = np.array([[0.0, 1.0], [2.0, 1.0]])
    query = np.array([[10.0, 1.0]])
    centred_train, centred_query, metadata = prepare_features(
        train, query, "duration_only", standardisation=PER_DIMENSION
    )
    # the enrollment mean, not the query mean, is what is subtracted
    assert centred_train.mean(axis=0)[0] == pytest.approx(0.0)
    assert centred_query[0][0] > centred_train.max()
    assert metadata["centering"] == "enrollment mean"
    assert "mean/std" in metadata["standardization"]


def test_prepare_features_l2_normalises_an_embedding_representation() -> None:
    train = np.array([[3.0, 4.0], [0.0, 5.0]])
    query = np.array([[6.0, 8.0]])
    _, _, metadata = prepare_features(train, query, "birdnet_v2_4", standardisation=PER_CLIP_L2)
    assert metadata["standardization"] == "per-clip L2"


def test_prepare_features_does_not_leak_the_query_rows_into_the_scaling() -> None:
    train = np.array([[0.0, 1.0], [2.0, 1.0]])
    near = prepare_features(
        train, np.array([[1.0, 1.0]]), "duration_only", standardisation=PER_DIMENSION
    )
    far = prepare_features(
        train, np.array([[500.0, 1.0]]), "duration_only", standardisation=PER_DIMENSION
    )
    assert np.allclose(near[0], far[0])


def _records(identities: list[str], per_identity: int) -> list[dict[str, Any]]:
    records = []
    for identity in identities:
        for index in range(per_identity):
            records.append(
                {
                    "identity": identity,
                    "condition": "foreground",
                    "split": "enrollment" if index == 0 else "query",
                    "filename": f"{identity}-{index}.wav",
                }
            )
    return records


def _separable_vectors(identities: list[str], per_identity: int, *, noise: float) -> Any:
    rng = np.random.default_rng(0)
    rows = []
    for position, _ in enumerate(identities):
        centre = np.zeros(len(identities))
        centre[position] = 1.0
        for _ in range(per_identity):
            rows.append(centre + rng.normal(0.0, noise, size=len(identities)))
    return np.asarray(rows)


def _evaluate(vectors: Any, records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return evaluate_endpoint(
        records=records,
        vectors=vectors,
        representation="synthetic",
        enrollment_condition="foreground",
        query_condition="foreground",
        ridge_lambda=1.0,
        seed=11,
        standardisation=PER_CLIP_L2,
        **{"permutations": 199, "bootstrap_replicates": 499, **kwargs},
    )


def test_a_separable_endpoint_scores_high_with_a_significant_permutation_test() -> None:
    identities = ["a", "b", "c", "d", "e"]
    records = _records(identities, 4)
    result = _evaluate(_separable_vectors(identities, 4, noise=0.05), records)
    assert result["classification"]["accuracy"] == pytest.approx(1.0)
    # 0.05 is the alpha the protocols register; with 199 permutations the
    # smallest attainable p-value is 0.005.
    assert result["permutation_control"]["p_value_plus_one"] <= 0.05
    assert result["identity_block_bootstrap_accuracy_95"][0] > 0.2


def test_an_endpoint_with_no_identity_signal_is_not_significant() -> None:
    identities = ["a", "b", "c", "d", "e"]
    records = _records(identities, 4)
    rng = np.random.default_rng(3)
    noise_only = rng.normal(size=(len(records), len(identities)))
    result = _evaluate(noise_only, records)
    assert result["permutation_control"]["p_value_plus_one"] > 0.05
    assert result["identity_block_bootstrap_accuracy_95"][0] <= 0.2


def test_the_bootstrap_resamples_identities_and_not_clips() -> None:
    """The interval must widen when identities disagree, however many clips there are.

    Four identities are perfect and one is hopeless. Resampling clips would
    average that away and report a narrow interval; resampling identities has to
    keep a lower bound that reflects a whole bird being wrong.
    """

    identities = ["a", "b", "c", "d", "e"]
    per_identity = 10
    records = _records(identities, per_identity)
    vectors = _separable_vectors(identities, per_identity, noise=0.02)
    # Every *query* clip of identity "e" is moved onto identity "a"'s centre,
    # leaving "e"'s enrollment clip where it was, so "e" enrolls correctly and
    # then every one of its nine queries is answered "a".
    a_centre = np.zeros(len(identities))
    a_centre[0] = 1.0
    for row, record in enumerate(records):
        if record["identity"] == "e" and record["split"] == "query":
            vectors[row] = a_centre

    result = _evaluate(vectors, records)
    lower, upper = result["identity_block_bootstrap_accuracy_95"]
    assert result["identity_accuracy"]["e"] == pytest.approx(0.0)
    assert result["classification"]["accuracy"] == pytest.approx(0.8)
    assert upper == pytest.approx(1.0)
    # Resampling the 50 clips of this same result gives [0.700, 0.920].
    # Resampling identities gives [0.400, 1.000], because losing one bird out of
    # five is a thing that can happen. This bound fails if the unit ever changes
    # back to the clip.
    assert lower < 0.7


def test_paired_difference_interval_is_positive_when_one_endpoint_dominates() -> None:
    identities = ["a", "b", "c", "d", "e"]
    records = _records(identities, 4)
    strong = _evaluate(_separable_vectors(identities, 4, noise=0.05), records)
    rng = np.random.default_rng(5)
    weak = _evaluate(rng.normal(size=(len(records), len(identities))), records)
    interval = paired_difference_interval(strong, weak, seed=7, replicates=499)
    assert interval[0] > 0.0
    assert interval[1] <= 1.0


def test_paired_difference_interval_refuses_mismatched_identity_sets() -> None:
    identities = ["a", "b", "c", "d", "e"]
    strong = _evaluate(_separable_vectors(identities, 4, noise=0.05), _records(identities, 4))
    fewer = ["a", "b", "c", "d"]
    other = _evaluate(_separable_vectors(fewer, 4, noise=0.05), _records(fewer, 4))
    with pytest.raises(ValueError, match="bootstrap identities differ"):
        paired_difference_interval(strong, other, seed=7, replicates=99)


# --- the probe reads what it is handed -------------------------------------


def _level_case(width: int) -> tuple[Any, Any, dict[str, Any]]:
    """Features that carry the level, and the level as the target.

    Eight identities, five clips each, levels spread so no identity's range
    covers another's: the probe holds one identity out, so a target it can
    only reach by interpolating from the rest is the harder case, not the
    easier one.
    """

    rng = np.random.default_rng(11)
    identities = np.asarray([f"bird{index // 5}" for index in range(40)])
    level = np.linspace(-40.0, -8.0, 40) + rng.normal(0.0, 0.2, size=40)
    if width == 1:
        features = level.reshape(-1, 1)
    else:
        columns = [level] + [rng.normal(0.0, 1.0, size=40) for _ in range(width - 1)]
        features = np.stack(columns, axis=1)
    return features, identities, {"rms_dbfs": level}


def test_the_probe_reads_a_one_number_target_it_is_handed_as_its_feature() -> None:
    """The falsification case for the probe itself.

    A representation that is the level, probed for the level, must come out at
    the ceiling. It came out at 0.000 for every model in the A5 runs, because
    the probe divided each row by its own length: for one number that leaves
    the same value in every row and the level is gone. A probe that cannot
    report a target it is handed as its own feature reports nothing about any
    other representation either.
    """

    features, identities, targets = _level_case(1)
    result = leave_one_identity_out(
        features=features,
        identities=identities,
        targets=targets,
        standardisation=standardisation_for(features.shape[1]),
        ridge_lambda=1.0,
    )
    assert result["rms_dbfs"]["r_squared"] > 0.99


def test_the_probe_reads_a_target_held_in_one_column_of_a_narrow_control() -> None:
    features, identities, targets = _level_case(3)
    result = leave_one_identity_out(
        features=features,
        identities=identities,
        targets=targets,
        standardisation=standardisation_for(features.shape[1]),
        ridge_lambda=1.0,
    )
    assert result["rms_dbfs"]["r_squared"] > 0.99


def test_per_clip_l2_loses_a_one_number_target_which_is_why_the_rule_is_chosen_by_width() -> None:
    """The wrong rule on the same data, to show the rule is what decides."""

    features, identities, targets = _level_case(1)
    result = leave_one_identity_out(
        features=features,
        identities=identities,
        targets=targets,
        standardisation=PER_CLIP_L2,
        ridge_lambda=1.0,
    )
    assert result["rms_dbfs"]["r_squared"] == pytest.approx(0.0)


def test_the_width_rule_is_the_one_the_classifier_uses() -> None:
    assert [standardisation_for(width) for width in (1, 2, 3)] == [PER_DIMENSION] * 3
    assert [standardisation_for(width) for width in (4, 128, 1024)] == [PER_CLIP_L2] * 3


def test_the_probe_refuses_a_standardisation_it_does_not_know() -> None:
    features, identities, targets = _level_case(1)
    with pytest.raises(ValueError, match="unknown standardisation"):
        leave_one_identity_out(
            features=features,
            identities=identities,
            targets=targets,
            standardisation="whitening",
            ridge_lambda=1.0,
        )


# --- the three heads -------------------------------------------------------


def _two_identity_endpoint() -> tuple[list[dict[str, Any]], Any]:
    """Two identities, separated, with one query clip planted next to a neighbour."""

    records = [
        {"filename": f"{name}-{index}.wav", "identity": name, "split": split, "condition": "fg"}
        for name, split in (
            ("a", "enrollment"),
            ("b", "enrollment"),
            ("a", "query"),
            ("b", "query"),
        )
        for index in range(4)
    ]
    rng = np.random.default_rng(5)
    vectors = np.asarray(
        [
            rng.normal(0.0, 0.05, size=6)
            + (3.0 if record["identity"] == "a" else -3.0) * np.eye(6)[0]
            for record in records
        ]
    )
    return records, vectors


def _run(head: str) -> dict[str, Any]:
    records, vectors = _two_identity_endpoint()
    return evaluate_endpoint(
        records=records,
        vectors=vectors,
        representation="synthetic",
        enrollment_condition="fg",
        query_condition="fg",
        ridge_lambda=1.0,
        seed=3,
        permutations=199,
        bootstrap_replicates=200,
        standardisation=PER_CLIP_L2,
        head=head,
    )


def test_every_head_names_itself_and_separates_a_separable_endpoint() -> None:
    for head in HEADS:
        result = _run(head)
        assert result["head"] == head
        assert result["classification"]["accuracy"] == pytest.approx(1.0), head


def test_the_default_head_is_the_one_every_earlier_figure_used() -> None:
    records, vectors = _two_identity_endpoint()
    shared = {
        "records": records,
        "vectors": vectors,
        "representation": "synthetic",
        "enrollment_condition": "fg",
        "query_condition": "fg",
        "ridge_lambda": 1.0,
        "seed": 3,
        "permutations": 199,
        "bootstrap_replicates": 200,
        "standardisation": PER_CLIP_L2,
    }
    assert evaluate_endpoint(**shared)["head"] == KERNEL_RIDGE
    default = evaluate_endpoint(**shared)["classification"]["accuracy"]
    named = evaluate_endpoint(**shared, head=KERNEL_RIDGE)["classification"]["accuracy"]
    assert default == named


def test_the_heads_are_not_the_same_computation() -> None:
    """Three names for one classifier would make agreement between them empty."""

    records, vectors = _two_identity_endpoint()
    prepared = prepare_features(
        vectors[[0, 1, 2, 3, 4, 5, 6, 7]],
        vectors[[8, 9, 10, 11]],
        "synthetic",
        standardisation=PER_CLIP_L2,
    )
    operators = {
        head: head_operator(head, train=prepared[0], query=prepared[1], ridge_lambda=1.0)
        for head in HEADS
    }
    assert operators[NEAREST_NEIGHBOUR].sum() == pytest.approx(4.0)  # one clip chosen per query
    assert not np.allclose(operators[KERNEL_RIDGE], operators[CLASS_MEAN])
    assert not np.allclose(operators[CLASS_MEAN], operators[NEAREST_NEIGHBOUR])


def test_a_head_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown head"):
        _run("random forest")


def test_the_class_mean_head_scores_the_mean_similarity_to_each_identity() -> None:
    """Checked against the arithmetic done directly, not against itself."""

    records, vectors = _two_identity_endpoint()
    train_rows = [index for index, record in enumerate(records) if record["split"] == "enrollment"]
    query_rows = [index for index, record in enumerate(records) if record["split"] == "query"]
    train, query, _ = prepare_features(
        vectors[train_rows], vectors[query_rows], "synthetic", standardisation=PER_CLIP_L2
    )
    unit = lambda block: block / np.linalg.norm(block, axis=1, keepdims=True)  # noqa: E731
    similarity = unit(query) @ unit(train).T
    names = [records[index]["identity"] for index in train_rows]
    expected = np.stack(
        [
            similarity[:, [i for i, name in enumerate(names) if name == identity]].mean(axis=1)
            for identity in sorted(set(names))
        ],
        axis=1,
    )
    operator = head_operator(CLASS_MEAN, train=train, query=query, ridge_lambda=1.0)
    counts = np.bincount([sorted(set(names)).index(name) for name in names]).astype(float)
    one_hot = (np.eye(2) / counts[:, None])[[sorted(set(names)).index(name) for name in names]]
    assert np.allclose(operator @ one_hot, expected)


def test_asking_for_no_permutations_or_bootstrap_reports_nothing_rather_than_a_number() -> None:
    """A caller that wants only the predictions must not get invented statistics.

    `run-split-difference` scores both splits with `permutations=0` and
    `bootstrap_replicates=0`, because its own paired interval is computed from
    the retained predictions afterwards. Before this test the quantile of an
    empty list raised, and the permutation p-value would have read exactly 1.000
    from zero shuffles, which is a fabricated value wearing a measurement's
    name.
    """

    identities = ["a", "b", "c", "d", "e"]
    records = _records(identities, 4)
    result = _evaluate(
        _separable_vectors(identities, 4, noise=0.05),
        records,
        permutations=0,
        bootstrap_replicates=0,
    )
    assert result["classification"]["accuracy"] == pytest.approx(1.0)
    assert result["identity_block_bootstrap_accuracy_95"] is None
    assert result["permutation_control"]["permutations"] == 0
    assert result["permutation_control"]["p_value_plus_one"] is None
    assert result["permutation_control"]["mean_accuracy"] is None
    assert result["permutation_control"]["maximum_accuracy"] is None
    assert len(result["predictions"]) == len(identities) * 3
