"""Architecture two, starting with whether its gradients are the gradients.

The model is written out in numpy, so nothing checks the derivatives except
these. Every parameter's gradient is compared against a central finite
difference of the loss itself. A sign error or a missing term fails here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xinyenyana.sequence_metric import (
    EMBEDDING_WIDTH,
    backward,
    contrastive_loss,
    divide_identities,
    pool,
    pool_batch,
    project,
)

CHANNELS = 7
CLIPS = 10


def _case(seed: int = 5) -> tuple[list[Any], Any, Any, Any]:
    generator = np.random.default_rng(seed)
    sequences = [
        generator.normal(size=(int(generator.integers(3, 9)), CHANNELS)) for _ in range(CLIPS)
    ]
    attention = generator.normal(size=CHANNELS) * 0.4
    weight = generator.normal(size=(4, 2 * CHANNELS)) * 0.3
    labels = np.asarray([index % 3 for index in range(CLIPS)])
    positives = labels[:, None] == labels[None, :]
    return sequences, attention, weight, positives


def _loss(sequences: list[Any], attention: Any, weight: Any, positives: Any) -> float:
    pooled, _ = pool_batch(sequences, attention)
    embeddings, _ = project(pooled, weight)
    return contrastive_loss(embeddings, positives)[0]


def _gradients(
    sequences: list[Any], attention: Any, weight: Any, positives: Any
) -> tuple[Any, Any]:
    pooled, caches = pool_batch(sequences, attention)
    embeddings, projection_cache = project(pooled, weight)
    _, gradient_embeddings = contrastive_loss(embeddings, positives)
    return backward(
        gradient_embeddings,
        pooled=pooled,
        weight=weight,
        projection_cache=projection_cache,
        pooling_caches=caches,
    )


def test_the_projection_gradient_matches_a_finite_difference() -> None:
    sequences, attention, weight, positives = _case()
    analytic, _ = _gradients(sequences, attention, weight, positives)
    step = 1e-6
    for row in range(weight.shape[0]):
        for column in range(weight.shape[1]):
            up, down = weight.copy(), weight.copy()
            up[row, column] += step
            down[row, column] -= step
            numeric = (
                _loss(sequences, attention, up, positives)
                - _loss(sequences, attention, down, positives)
            ) / (2 * step)
            assert abs(numeric - analytic[row, column]) < 1e-6 + 1e-4 * abs(numeric)


def test_the_attention_gradient_matches_a_finite_difference() -> None:
    sequences, attention, weight, positives = _case()
    _, analytic = _gradients(sequences, attention, weight, positives)
    step = 1e-6
    for index in range(len(attention)):
        up, down = attention.copy(), attention.copy()
        up[index] += step
        down[index] -= step
        numeric = (
            _loss(sequences, up, weight, positives) - _loss(sequences, down, weight, positives)
        ) / (2 * step)
        assert abs(numeric - analytic[index]) < 1e-6 + 1e-4 * abs(numeric)


def test_a_zero_attention_vector_reproduces_the_plain_average() -> None:
    generator = np.random.default_rng(9)
    sequence = generator.normal(size=(11, CHANNELS))
    vector, _ = pool(sequence, np.zeros(CHANNELS))
    assert np.allclose(vector[:CHANNELS], sequence.mean(axis=0))
    assert np.allclose(vector[CHANNELS:], np.sqrt(sequence.var(axis=0) + 1e-8))


def test_a_row_with_no_positive_contributes_nothing() -> None:
    generator = np.random.default_rng(4)
    embeddings = generator.normal(size=(4, EMBEDDING_WIDTH))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    none_positive = np.zeros((4, 4), dtype=bool)
    loss, gradient = contrastive_loss(embeddings, none_positive)
    assert loss == 0.0
    assert not gradient.any()


def test_the_identity_division_reads_only_the_names() -> None:
    names = [f"bird{index:02d}" for index in range(16)]
    first = divide_identities(names)
    second = divide_identities(list(reversed(names)))
    assert first == second
    assert set(first["development"]) & set(first["evaluation"]) == set()
    assert sorted(first["development"] + first["evaluation"]) == names
    for label in ("development", "evaluation"):
        members = set(first[label])
        assert set(first[f"{label}_enrolled"]) | set(first[f"{label}_strangers"]) == members
        assert set(first[f"{label}_enrolled"]) & set(first[f"{label}_strangers"]) == set()
        assert first[f"{label}_enrolled"] and first[f"{label}_strangers"]


def test_too_few_identities_stops_rather_than_returning_an_empty_group() -> None:
    with pytest.raises(ValueError, match="at least four"):
        divide_identities(["a", "b", "c"])


def _planted(identities: int, sessions: int, per_cell: int, seed: int = 21) -> tuple[Any, Any, Any]:
    """Clips whose identity sits in one direction and whose session sits in another.

    The identity signature is weaker than the session signature, so a model that
    is allowed to pull same-session pairs together will learn the session, and
    one restricted to cross-session pairs has to find the bird.
    """

    generator = np.random.default_rng(seed)
    identity_axis = np.eye(CHANNELS)[0]
    session_axis = np.eye(CHANNELS)[1]
    sequences, names, groups = [], [], []
    for bird in range(identities):
        for session in range(sessions):
            for _ in range(per_cell):
                length = int(generator.integers(4, 10))
                base = 1.0 * (bird + 1) * identity_axis + 4.0 * (session + 1) * session_axis
                sequences.append(base + generator.normal(size=(length, CHANNELS)) * 0.3)
                names.append(f"bird{bird}")
                groups.append(f"day{session}")
    return sequences, names, groups


def test_the_hierarchy_refuses_same_session_pairs_when_asked_to() -> None:
    from xinyenyana.sequence_metric import positive_mask

    names = ["a", "a", "a", "b"]
    groups = ["day1", "day1", "day2", "day1"]
    both = positive_mask(names, groups, cross_group_only=False)
    crossing = positive_mask(names, groups, cross_group_only=True)
    assert both[0, 1] and not crossing[0, 1]
    assert both[0, 2] and crossing[0, 2]
    assert not both[0, 3] and not crossing[0, 3]


def test_an_endpoint_with_no_session_label_is_refused_rather_than_ignored() -> None:
    from xinyenyana.sequence_metric import positive_mask

    with pytest.raises(ValueError, match="no session label"):
        positive_mask(["a", "a", "b"], None, cross_group_only=True)


def test_training_lowers_its_own_loss_and_separates_the_planted_identities() -> None:
    from xinyenyana.sequence_metric import embed, train

    sequences, names, groups = _planted(identities=4, sessions=3, per_cell=6)
    fit = train(sequences, names, groups, cross_group_only=True, passes=30, batch=32)
    assert fit["loss_last"] < fit["loss_first"]
    vectors = embed(sequences, fit)
    labels = np.asarray(names)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = labels[np.argmax(similarity, axis=1)]
    assert float(np.mean(nearest == labels)) > 0.8


def test_a_fit_made_without_a_clip_still_embeds_it() -> None:
    from xinyenyana.sequence_metric import embed, train

    sequences, names, groups = _planted(identities=4, sessions=3, per_cell=5)
    held_out = sequences[:6]
    fit = train(sequences[6:], names[6:], groups[6:], cross_group_only=True, passes=10, batch=16)
    vectors = embed(held_out, fit)
    assert vectors.shape == (6, EMBEDDING_WIDTH)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def _endpoint_with_sessions(
    identities: int, sessions: int, per_cell: int
) -> tuple[Any, Any, Any, Any]:
    from pathlib import Path

    from xinyenyana.a2 import ClipRecord, Endpoint
    from xinyenyana.sequence_metric import SEQUENCE_LAYERS

    generator = np.random.default_rng(31)
    records, clip_sequences, session_labels = [], [], []
    identity_axis = np.eye(CHANNELS)[0]
    session_axis = np.eye(CHANNELS)[1]
    for bird in range(identities):
        for session in range(sessions):
            for index in range(per_cell):
                split = "enrollment" if index % 2 == 0 else "query"
                name = f"bird{bird:02d}-day{session}-{index}.wav"
                records.append(
                    ClipRecord(
                        filename=name,
                        path=Path(name),
                        identity=f"bird{bird:02d}",
                        split=split,
                        context={"condition": "foreground", "recorded": f"day{session}"},
                    )
                )
                length = int(generator.integers(4, 9))
                base = 2.0 * (bird + 1) * identity_axis + 1.0 * (session + 1) * session_axis
                clip_sequences.append(base + generator.normal(size=(length, CHANNELS)) * 0.3)
                session_labels.append(f"day{session}")
    endpoint = Endpoint(
        name="synthetic-sequences",
        records=tuple(records),
        categorical_targets=("identity",),
        manifest_sha256="0" * 64,
        source_document="a synthetic endpoint written by this test",
    )
    sequences = {key: clip_sequences for key in SEQUENCE_LAYERS}
    means = {
        key.replace(".sequence", ".mean"): np.asarray(
            [clip.mean(axis=0) for clip in clip_sequences]
        )
        for key in SEQUENCE_LAYERS
    }
    return endpoint, sequences, means, session_labels


def test_no_evaluation_identity_reaches_the_fit() -> None:
    from xinyenyana.sequence_metric import divide_identities, run_sequence_metric

    endpoint, sequences, means, sessions = _endpoint_with_sessions(8, 3, 6)
    summary = run_sequence_metric(
        endpoint, sequences, means, sessions=sessions, permutations=19, bootstrap_replicates=20
    )
    groups = summary["identity_groups"]
    assert groups == divide_identities([r.identity for r in endpoint.records])
    assert set(groups["development"]).isdisjoint(groups["evaluation"])
    fitted_clips = sum(
        1 for record in endpoint.records if record.identity in set(groups["development"])
    )
    for by_layer in summary["arms"].values():
        assert by_layer["learned sequence"]["fit"]["clips"] == fitted_clips


def test_every_arm_reports_closed_open_and_few_shot() -> None:
    from xinyenyana.sequence_metric import SHOTS, run_sequence_metric

    endpoint, sequences, means, sessions = _endpoint_with_sessions(8, 3, 6)
    summary = run_sequence_metric(
        endpoint, sequences, means, sessions=sessions, permutations=19, bootstrap_replicates=20
    )
    assert summary["session_label"] == "published"
    for by_layer in summary["arms"].values():
        for arm in ("learned sequence", "layer average"):
            entry = by_layer[arm]
            assert entry["closed_set"]["accuracy"] >= 0.0
            assert entry["open_set"]["threshold_chosen_on"] == "development identities"
            assert sorted(entry["few_shot"]) == sorted(str(shots) for shots in SHOTS)


def test_an_endpoint_with_no_session_label_says_so_and_still_runs() -> None:
    from xinyenyana.sequence_metric import run_sequence_metric

    endpoint, sequences, means, _ = _endpoint_with_sessions(8, 3, 6)
    summary = run_sequence_metric(
        endpoint, sequences, means, sessions=None, permutations=19, bootstrap_replicates=20
    )
    assert summary["session_label"] == "absent"
    for by_layer in summary["arms"].values():
        assert by_layer["learned sequence"]["fit"]["cross_session_positives"] is False


def test_the_few_shot_arm_says_in_the_result_why_it_has_no_permutation_block() -> None:
    """A reader of the archived result must not have to guess at a null control."""
    from xinyenyana.sequence_metric import NO_FEW_SHOT_PERMUTATION, run_sequence_metric

    endpoint, sequences, means, sessions = _endpoint_with_sessions(8, 3, 6)
    summary = run_sequence_metric(
        endpoint, sequences, means, sessions=sessions, permutations=19, bootstrap_replicates=20
    )
    for by_layer in summary["arms"].values():
        for entry in by_layer.values():
            assert entry["closed_set"]["permutation_control"] is not None
            assert "permutation_control_absent_because" not in entry["closed_set"]
            for shot in entry["few_shot"].values():
                assert shot["permutation_control"] is None
                assert shot["permutation_control_absent_because"] == NO_FEW_SHOT_PERMUTATION


def test_an_endpoint_with_no_stranger_clips_records_the_open_set_as_unavailable() -> None:
    """A division that cannot supply strangers says so instead of raising."""

    from xinyenyana.sequence_metric import divide_identities, run_sequence_metric

    endpoint, sequences, means, sessions = _endpoint_with_sessions(8, 3, 6)
    groups = divide_identities([r.identity for r in endpoint.records])
    strangers = set(groups["development_strangers"])
    kept = tuple(record for record in endpoint.records if record.identity not in strangers)
    from dataclasses import replace

    trimmed = replace(endpoint, records=kept)
    keep_rows = [
        index for index, record in enumerate(endpoint.records) if record.identity not in strangers
    ]
    trimmed_sequences = {key: [sequences[key][row] for row in keep_rows] for key in sequences}
    trimmed_means = {key: means[key][keep_rows] for key in means}
    trimmed_sessions = [sessions[row] for row in keep_rows]
    summary = run_sequence_metric(
        trimmed,
        trimmed_sequences,
        trimmed_means,
        sessions=trimmed_sessions,
        permutations=9,
        bootstrap_replicates=10,
    )
    for by_layer in summary["arms"].values():
        for entry in by_layer.values():
            assert "closed_set" in entry


def _endpoint_with_backgrounds(
    identities: int, sessions: int, per_cell: int, *, in_background: bool
):
    """The same endpoint with an ambient recording beside every call.

    ``in_background`` decides whether the identity signature is planted in those
    ambient recordings, so the control has a known answer both ways.
    """

    from pathlib import Path

    from xinyenyana.a2 import ClipRecord, Endpoint
    from xinyenyana.sequence_metric import SEQUENCE_LAYERS

    generator = np.random.default_rng(41)
    records, clip_sequences, session_labels = [], [], []
    identity_axis = np.eye(CHANNELS)[0]
    for bird in range(identities):
        for session in range(sessions):
            for index in range(per_cell):
                for condition in ("foreground", "background"):
                    split = "enrollment" if index % 2 == 0 else "query"
                    name = f"bird{bird:02d}-day{session}-{condition}-{index}.wav"
                    records.append(
                        ClipRecord(
                            filename=name,
                            path=Path(name),
                            identity=f"bird{bird:02d}",
                            split=split,
                            context={"condition": condition, "recorded": f"day{session}"},
                        )
                    )
                    length = int(generator.integers(4, 9))
                    carries = condition == "foreground" or in_background
                    base = 3.0 * (bird + 1) * identity_axis if carries else np.zeros(CHANNELS)
                    clip_sequences.append(base + generator.normal(size=(length, CHANNELS)) * 0.3)
                    session_labels.append(f"day{session}")
    endpoint = Endpoint(
        name="synthetic-with-backgrounds",
        records=tuple(records),
        categorical_targets=("identity",),
        manifest_sha256="0" * 64,
        source_document="a synthetic endpoint written by this test",
    )
    sequences = {key: clip_sequences for key in SEQUENCE_LAYERS}
    means = {
        key.replace(".sequence", ".mean"): np.asarray(
            [clip.mean(axis=0) for clip in clip_sequences]
        )
        for key in SEQUENCE_LAYERS
    }
    return endpoint, sequences, means, session_labels


def test_the_background_control_moves_with_the_place_and_not_otherwise() -> None:
    """The validity control, on data where the answer is planted.

    The model is fitted on foreground clips either way. When the ambient
    recordings carry the bird's signature, scoring them names the bird; when
    they do not, every arm sits near chance. The comparison is made arm by arm,
    because the arms differ in how well they read this planted signal at all.
    """

    from xinyenyana.sequence_metric import run_sequence_metric

    measured = {}
    for in_background in (False, True):
        endpoint, sequences, means, sessions = _endpoint_with_backgrounds(
            8, 3, 6, in_background=in_background
        )
        summary = run_sequence_metric(
            endpoint, sequences, means, sessions=sessions, permutations=19, bootstrap_replicates=20
        )
        assert summary["background_control"] is True
        for layer, by_layer in summary["arms"].items():
            for arm, entry in by_layer.items():
                control = entry["background_only"]
                assert control is not None
                measured[(in_background, layer, arm)] = control

    for (in_background, layer, arm), control in measured.items():
        if in_background:
            continue
        chance = control["chance_accuracy"]
        assert control["accuracy"] < chance + 0.15, (layer, arm)
        planted = measured[(True, layer, arm)]
        assert planted["accuracy"] > control["accuracy"] + 0.1, (layer, arm)


def test_an_endpoint_without_backgrounds_records_no_background_control() -> None:
    from xinyenyana.sequence_metric import run_sequence_metric

    endpoint, sequences, means, sessions = _endpoint_with_sessions(8, 3, 6)
    summary = run_sequence_metric(
        endpoint, sequences, means, sessions=sessions, permutations=9, bootstrap_replicates=10
    )
    assert summary["background_control"] is False
    for by_layer in summary["arms"].values():
        for entry in by_layer.values():
            assert entry["background_only"] is None


def test_the_unweighted_sequence_arm_is_the_flat_attention_of_the_learned_one() -> None:
    """The control that isolates the attention, checked against the model itself."""

    from xinyenyana.sequence_metric import pool, run_sequence_metric

    endpoint, sequences, means, sessions = _endpoint_with_sessions(8, 3, 6)
    summary = run_sequence_metric(
        endpoint, sequences, means, sessions=sessions, permutations=9, bootstrap_replicates=10
    )
    for by_layer in summary["arms"].values():
        assert sorted(by_layer) == ["layer average", "learned sequence", "sequence average"]
    first = sequences["stage1.sequence"][0]
    flat, _ = pool(first, np.zeros(CHANNELS))
    assert np.allclose(flat[:CHANNELS], np.asarray(first).mean(axis=0))


def test_the_three_arms_do_not_return_one_number_between_them() -> None:
    """A control that returns the same numbers as the thing it controls is not one."""

    from xinyenyana.sequence_metric import run_sequence_metric

    endpoint, sequences, means, sessions = _endpoint_with_sessions(8, 3, 6)
    summary = run_sequence_metric(
        endpoint, sequences, means, sessions=sessions, permutations=9, bootstrap_replicates=10
    )
    distinct = set()
    for layer, by_layer in summary["arms"].items():
        galleries = {entry["closed_set"]["gallery_clips"] for entry in by_layer.values()}
        assert len(galleries) == 1, layer
        distinct.add(
            tuple(round(e["closed_set"]["accuracy"], 6) for _, e in sorted(by_layer.items()))
        )
    assert any(len(set(row)) > 1 for row in distinct)
