"""Architecture one, and the null that a fitted representation needs.

The permutation control here refits the discriminant under every shuffle, so it
tests the whole fitted procedure rather than the head alone. The last test
compares it against the cheaper control `evaluate_endpoint` uses, which holds
the discriminant fitted with the true labels and shuffles only the head's: on
planted noise at four sizes the two agree, so no case has been found here where
the cheaper one misleads. The refitting control is what this module uses anyway,
because its validity does not rest on that agreement holding in general.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.evaluation import KERNEL_RIDGE
from xinyenyana.learned_combination import (
    FINAL_LAYER_ALONE,
    FITTED_ARMS,
    LAYER_KEYS,
    LEARNED_COMBINATION,
    build_arm,
    discriminant_projection,
    evaluate_arm,
    run_learned_combination,
)

WIDTH = 24


def _endpoint(*, identities: int, per_split: int, conditions: tuple[str, ...]) -> Endpoint:
    records = []
    for number in range(identities):
        for condition in conditions:
            for split in ("enrollment", "query"):
                for index in range(per_split):
                    name = f"bird{number}-{condition}-{split}-{index}.wav"
                    records.append(
                        ClipRecord(
                            filename=name,
                            path=Path(name),
                            identity=f"bird{number}",
                            split=split,
                            context={"condition": condition},
                        )
                    )
    return Endpoint(
        name="synthetic",
        records=tuple(records),
        categorical_targets=("identity",),
        manifest_sha256="0" * 64,
        source_document="a synthetic endpoint written by this test",
    )


def _layers(endpoint: Endpoint, *, carries_identity: bool, seed: int = 3) -> dict[str, Any]:
    """One matrix per layer key. Identity is planted only where asked for."""

    generator = np.random.default_rng(seed)
    signatures = {name: generator.normal(size=WIDTH) * 3.0 for name in endpoint.identities}
    vectors: dict[str, Any] = {}
    for key in LAYER_KEYS:
        rows = []
        for record in endpoint.records:
            noise = generator.normal(size=WIDTH)
            rows.append(noise + signatures[record.identity] if carries_identity else noise)
        vectors[key] = np.asarray(rows)
    return vectors


def test_the_projection_returns_at_most_one_fewer_direction_than_identities() -> None:
    generator = np.random.default_rng(5)
    features = generator.normal(size=(40, 30))
    labels = [f"bird{index % 4}" for index in range(40)]
    assert discriminant_projection(features, labels, width=16).shape == (30, 3)


def test_the_projection_refuses_a_single_identity() -> None:
    generator = np.random.default_rng(5)
    with pytest.raises(ValueError):
        discriminant_projection(generator.normal(size=(10, 4)), ["one"] * 10)


def test_a_fitted_arm_moves_with_the_labels_and_an_unfitted_one_does_not() -> None:
    endpoint = _endpoint(identities=4, per_split=8, conditions=("foreground",))
    vectors = _layers(endpoint, carries_identity=True)
    rows = [i for i, r in enumerate(endpoint.records) if r.split == "enrollment"]
    true = [endpoint.records[row].identity for row in rows]
    shuffled = list(np.random.default_rng(1).permutation(true))
    for arm, moves in ((LEARNED_COMBINATION, True), (FINAL_LAYER_ALONE, False)):
        first = build_arm(arm, vectors, enrollment_rows=rows, labels=true)
        second = build_arm(arm, vectors, enrollment_rows=rows, labels=shuffled)
        assert (not np.allclose(first, second)) is moves


def test_the_refitting_null_is_not_fooled_by_a_discriminant_fitted_to_noise() -> None:
    """Few clips, many dimensions, no identity in the features at all.

    The discriminant can still separate the enrolment clips, because with this
    many directions it can separate anything. The refitting null sees that the
    same thing happens under a shuffle and returns a p-value near uniform.
    """

    endpoint = _endpoint(identities=4, per_split=6, conditions=("foreground",))
    vectors = _layers(endpoint, carries_identity=False, seed=11)
    result = evaluate_arm(
        LEARNED_COMBINATION,
        vectors,
        records=list(endpoint.records),
        enrollment_condition="foreground",
        query_condition="foreground",
        head=KERNEL_RIDGE,
        permutations=199,
        bootstrap_replicates=50,
    )
    assert result["permutation_control"]["refits_the_representation"] is True
    assert result["permutation_control"]["p_value_plus_one"] > 0.05


def test_the_refitting_null_still_finds_a_real_signal() -> None:
    endpoint = _endpoint(identities=4, per_split=12, conditions=("foreground",))
    vectors = _layers(endpoint, carries_identity=True, seed=13)
    result = evaluate_arm(
        LEARNED_COMBINATION,
        vectors,
        records=list(endpoint.records),
        enrollment_condition="foreground",
        query_condition="foreground",
        head=KERNEL_RIDGE,
        permutations=199,
        bootstrap_replicates=50,
    )
    assert result["classification"]["accuracy"] > 0.8
    assert result["permutation_control"]["p_value_plus_one"] <= 0.01


def test_every_arm_and_head_is_scored_on_all_four_pairings_where_backgrounds_exist() -> None:
    endpoint = _endpoint(identities=3, per_split=6, conditions=("foreground", "background"))
    vectors = _layers(endpoint, carries_identity=True, seed=17)
    summary = run_learned_combination(endpoint, vectors, permutations=19, bootstrap_replicates=20)
    assert set(summary["arms"]) == set(summary["p_bonferroni_over_family"])
    for arm, by_arm in summary["arms"].items():
        for head, by_head in by_arm.items():
            assert sorted(by_head) == [
                "background_to_background",
                "background_to_foreground",
                "foreground_to_background",
                "foreground_to_foreground",
            ], (arm, head)
            for value in by_head.values():
                assert value["permutation_control"]["refits_the_representation"] is (
                    arm in FITTED_ARMS
                )


def test_a_missing_layer_stops_the_run_rather_than_shrinking_it() -> None:
    endpoint = _endpoint(identities=3, per_split=4, conditions=("foreground",))
    vectors = _layers(endpoint, carries_identity=True)
    del vectors[LAYER_KEYS[2]]
    with pytest.raises(ValueError, match="extraction cache holds no"):
        run_learned_combination(endpoint, vectors, permutations=5, bootstrap_replicates=5)


def test_the_two_nulls_agree_on_planted_noise_at_four_sizes() -> None:
    """What the cheaper control would have said, measured rather than asserted.

    Nothing in these features carries identity, and the enrolment side is small
    relative to the width, which is where a fitted representation has the most
    room to invent a structure. The discriminant separates the enrolment clips
    and none of that reaches the held-out clips, so both controls sit at chance
    and neither reports a result.
    """

    from xinyenyana.evaluation import evaluate_endpoint

    for identities, per_split, width in ((4, 6, 24), (4, 4, 48), (6, 3, 64), (8, 3, 96)):
        endpoint = _endpoint(identities=identities, per_split=per_split, conditions=("foreground",))
        generator = np.random.default_rng(11)
        vectors = {
            key: np.asarray([generator.normal(size=width) for _ in endpoint.records])
            for key in LAYER_KEYS
        }
        rows = [i for i, r in enumerate(endpoint.records) if r.split == "enrollment"]
        features = build_arm(
            LEARNED_COMBINATION,
            vectors,
            enrollment_rows=rows,
            labels=[endpoint.records[row].identity for row in rows],
        )
        head_only = evaluate_endpoint(
            records=[record.as_evaluation_record() for record in endpoint.records],
            vectors=features,
            representation="learned-combination",
            enrollment_condition="foreground",
            query_condition="foreground",
            ridge_lambda=1.0,
            seed=17,
            permutations=199,
            bootstrap_replicates=20,
            standardisation="per-clip L2",
            head=KERNEL_RIDGE,
        )
        refitting = evaluate_arm(
            LEARNED_COMBINATION,
            vectors,
            records=list(endpoint.records),
            enrollment_condition="foreground",
            query_condition="foreground",
            head=KERNEL_RIDGE,
            permutations=199,
            bootstrap_replicates=20,
        )
        chance = 1.0 / identities
        assert abs(refitting["classification"]["accuracy"] - chance) < 0.15
        assert (
            abs(
                refitting["permutation_control"]["mean_accuracy"]
                - head_only["permutation_control"]["mean_accuracy"]
            )
            < 0.05
        )
        assert refitting["permutation_control"]["p_value_plus_one"] > 0.05
        assert head_only["permutation_control"]["p_value_plus_one"] > 0.05


def _general_eigen_projection(features, labels, *, width, ridge):
    """The same construction through a general eigendecomposition.

    This is what `discriminant_projection` did before the Cholesky and thin
    singular value decomposition replaced it. It is kept here as the thing the
    faster path is checked against, because the faster path is only worth having
    if it computes the same directions.
    """

    values = np.asarray(features, dtype=np.float64)
    names = np.asarray(labels)
    unique = np.unique(names)
    overall = values.mean(axis=0)
    centred = values - overall
    total = centred.T @ centred
    between = np.zeros_like(total)
    for identity in unique:
        rows = values[names == identity]
        offset = (rows.mean(axis=0) - overall)[:, None]
        between += len(rows) * (offset @ offset.T)
    within = total - between
    floor = ridge * max(float(np.mean(np.diag(within))), 1e-30)
    within = within + floor * np.eye(within.shape[0])
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(within, between))
    order = np.argsort(-eigenvalues.real)
    keep = min(width, len(unique) - 1, values.shape[1])
    return np.ascontiguousarray(eigenvectors.real[:, order[:keep]])


@pytest.mark.parametrize(
    ("clips", "dimensions", "identities", "width"),
    [(60, 12, 4, 3), (40, 30, 5, 4), (90, 8, 3, 2), (50, 25, 6, 5)],
)
def test_the_fast_discriminant_spans_the_same_space_as_the_general_one(
    clips: int, dimensions: int, identities: int, width: int
) -> None:
    from xinyenyana.learned_combination import WITHIN_IDENTITY_RIDGE, discriminant_projection

    generator = np.random.default_rng(clips + dimensions)
    signatures = generator.normal(size=(identities, dimensions)) * 2.0
    labels = [f"bird{index % identities}" for index in range(clips)]
    features = np.vstack(
        [
            signatures[index % identities] + generator.normal(size=dimensions)
            for index in range(clips)
        ]
    )
    fast = discriminant_projection(features, labels, width=width, ridge=WITHIN_IDENTITY_RIDGE)
    slow = _general_eigen_projection(features, labels, width=width, ridge=WITHIN_IDENTITY_RIDGE)
    assert fast.shape == slow.shape
    # Same subspace: each projects the other onto itself with nothing left over.
    for matrix, other in ((fast, slow), (slow, fast)):
        basis = np.linalg.qr(other)[0]
        residual = matrix - basis @ (basis.T @ matrix)
        assert np.max(np.abs(residual)) < 1e-6 * max(1.0, float(np.max(np.abs(matrix))))


def test_the_fast_discriminant_returns_unit_directions() -> None:
    from xinyenyana.learned_combination import discriminant_projection

    generator = np.random.default_rng(3)
    labels = [f"bird{index % 4}" for index in range(48)]
    features = generator.normal(size=(48, 20))
    projection = discriminant_projection(features, labels, width=3)
    assert np.allclose(np.linalg.norm(projection, axis=0), 1.0)


def test_an_unfitted_arm_hands_on_the_normalised_vectors_untouched() -> None:
    """The invariant the baselines depend on, checked directly.

    An arm that fits nothing must hand the scoring path exactly what
    `evaluate_endpoint` would be handed, so that its number is the project's
    established number for that representation. A first version subtracted the
    enrolment mean here as well, which made the scoring path renormalise centred
    vectors and rescale every clip. On the Great Tit that read the final
    embedding at 0.475 where the frozen probe reads 0.4437 on the same clips;
    the cosine heads could not see it because they renormalise, and the ridge
    head could.
    """

    from xinyenyana.learned_combination import (
        FINAL_LAYER_ALONE,
        LAYERS_CONCATENATED,
        _normalised,
        apply_arm,
        prepare_arm,
    )

    endpoint = _endpoint(identities=6, per_split=8, conditions=("foreground",))
    vectors = _layers(endpoint, carries_identity=True, seed=29)
    rows = [index for index, r in enumerate(endpoint.records) if r.split == "enrollment"]
    labels = [endpoint.records[row].identity for row in rows]
    kept = list(range(len(endpoint.records)))
    expected = {
        FINAL_LAYER_ALONE: _normalised(vectors["embedding.mean"]),
        LAYERS_CONCATENATED: np.hstack([_normalised(vectors[key]) for key in LAYER_KEYS]),
    }
    for arm, wanted in expected.items():
        prepared = prepare_arm(arm, vectors, enrollment_rows=rows, kept_rows=kept)
        assert np.array_equal(apply_arm(prepared, labels), wanted), arm
        assert np.allclose(np.linalg.norm(apply_arm(prepared, labels), axis=1), 1.0) or (
            arm == LAYERS_CONCATENATED
        )


def _offset_layers(endpoint, *, seed: int = 23, offset: float = 5.0) -> dict[str, Any]:
    """Vectors with a large direction every clip shares, as a real encoder has.

    Without it this test cannot fail. BirdNET embeddings sit far from the
    origin, so subtracting the enrolment mean moves every vector a long way and
    renormalising afterwards rescales the clips by different amounts. Isotropic
    noise around zero does not, and the defect this test exists for is invisible
    on it.
    """

    generator = np.random.default_rng(seed)
    common = np.abs(generator.normal(size=WIDTH)) * offset
    signatures = {name: generator.normal(size=WIDTH) for name in endpoint.identities}
    vectors: dict[str, Any] = {}
    rows = [
        common + signatures[record.identity] + generator.normal(size=WIDTH) * 1.5
        for record in endpoint.records
    ]
    matrix = np.asarray(rows)
    for key in LAYER_KEYS:
        vectors[key] = matrix
    return vectors


def test_the_unfitted_arms_reproduce_the_project_pipeline_exactly() -> None:
    """A baseline that disagrees with the established figure is not a baseline.

    `final layer alone` must give the number `evaluate_endpoint` gives on the
    same vectors, and `layers concatenated` the number it gives on those layers
    stacked. A first version centred the unfitted arms before handing them on,
    so the scoring path renormalised centred vectors and rescaled every clip.
    The cosine heads could not see it and the ridge head could: on the Great Tit
    it read the final embedding at 0.475 where the frozen probe reads 0.4437 on
    the same clips.
    """

    from xinyenyana.evaluation import HEADS, evaluate_endpoint
    from xinyenyana.learned_combination import (
        FINAL_LAYER_ALONE,
        LAYERS_CONCATENATED,
        _normalised,
        evaluate_arm,
    )

    endpoint = _endpoint(identities=8, per_split=12, conditions=("foreground",))
    vectors = _offset_layers(endpoint)
    records = [record.as_evaluation_record() for record in endpoint.records]
    expected = {
        FINAL_LAYER_ALONE: _normalised(vectors["embedding.mean"]),
        LAYERS_CONCATENATED: np.hstack([_normalised(vectors[key]) for key in LAYER_KEYS]),
    }
    for arm, features in expected.items():
        mine = evaluate_arm(
            arm,
            vectors,
            records=list(endpoint.records),
            enrollment_condition="foreground",
            query_condition="foreground",
            head=HEADS,
            permutations=9,
            bootstrap_replicates=20,
        )
        for head in HEADS:
            theirs = evaluate_endpoint(
                records=records,
                vectors=features,
                representation="baseline",
                enrollment_condition="foreground",
                query_condition="foreground",
                ridge_lambda=1.0,
                seed=17,
                permutations=9,
                bootstrap_replicates=20,
                standardisation="per-clip L2",
                head=head,
            )
            assert mine[head]["classification"]["accuracy"] == pytest.approx(
                theirs["classification"]["accuracy"]
            ), (arm, head)
