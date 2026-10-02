"""Tests for the validity gate, the measurement of how much is the place."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.evaluation import HEADS, KERNEL_RIDGE
from xinyenyana.validity import PAIRINGS, run_validity_gate

IDENTITIES = ("alpha", "beta", "gamma")
ENROLLMENT_PER_CELL = 20
QUERY_PER_CELL = 20
WIDTH = 8


def _records(*, conditions: tuple[str, ...]) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for identity in IDENTITIES:
        for condition in conditions:
            for split, count in (("enrollment", ENROLLMENT_PER_CELL), ("query", QUERY_PER_CELL)):
                for index in range(count):
                    name = f"{identity}-{condition}-{split}-{index}.wav"
                    records.append(
                        ClipRecord(
                            filename=name,
                            path=Path(name),
                            identity=identity,
                            split=split,
                            context={"condition": condition},
                        )
                    )
    return records


def _endpoint(*, conditions: tuple[str, ...] = ("foreground", "background")) -> Endpoint:
    return Endpoint(
        name="synthetic",
        records=tuple(_records(conditions=conditions)),
        categorical_targets=("identity",),
        manifest_sha256="0" * 64,
        source_document="a synthetic endpoint written by this test",
    )


def _vectors(endpoint: Endpoint, *, identity_in_background: bool, signature: float = 4.0) -> Any:
    """One vector per clip, with the identity signature switched per condition.

    ``signature`` is how far apart the identities sit against a fixed noise
    level. The default is far enough that every head reads the fixture
    perfectly; a smaller value is what separates the heads from each other.
    """

    generator = np.random.default_rng(11)
    signatures = {
        identity: generator.normal(size=WIDTH) * signature for identity in endpoint.identities
    }
    rows = []
    for record in endpoint.records:
        noise = generator.normal(size=WIDTH) * 0.1
        carries = record.context["condition"] == "foreground" or identity_in_background
        rows.append(signatures[record.identity] + noise if carries else noise)
    return np.asarray(rows, dtype=np.float64)


def _run(*, identity_in_background: bool) -> dict[str, Any]:
    endpoint = _endpoint()
    return run_validity_gate(
        endpoint=endpoint,
        vectors=_vectors(endpoint, identity_in_background=identity_in_background),
        representation="synthetic",
    )


# --- what the gate refuses to score --------------------------------------------


def test_an_endpoint_without_backgrounds_is_refused() -> None:
    """Without the place recorded on its own, the place cannot be subtracted."""

    endpoint = _endpoint(conditions=("foreground",))
    with pytest.raises(ValueError, match="carries no background clips"):
        run_validity_gate(
            endpoint=endpoint,
            vectors=_vectors(endpoint, identity_in_background=False),
            representation="synthetic",
        )


# --- what the gate reports -----------------------------------------------------


def test_every_pairing_of_conditions_is_scored() -> None:
    result = _run(identity_in_background=False)
    assert set(result["pairings"]) == {f"{enrollment}_to_{query}" for enrollment, query in PAIRINGS}


def test_each_pairing_carries_its_interval_its_p_value_and_its_counts() -> None:
    result = _run(identity_in_background=False)
    for scored in result["pairings"].values():
        assert set(scored) == {
            "accuracy",
            "identity_block_bootstrap_accuracy_95",
            "permutation_p",
            "roc_auc",
            "enrollment_calls",
            "query_calls",
        }
        assert scored["enrollment_calls"] == len(IDENTITIES) * ENROLLMENT_PER_CELL
        assert scored["query_calls"] == len(IDENTITIES) * QUERY_PER_CELL


def test_the_summary_names_the_endpoint_its_identities_and_its_manifest() -> None:
    result = _run(identity_in_background=False)
    assert result["endpoint"] == "synthetic"
    assert result["representation"] == "synthetic"
    assert result["identities"] == len(IDENTITIES)
    assert result["clips"] == len(IDENTITIES) * 2 * (ENROLLMENT_PER_CELL + QUERY_PER_CELL)
    assert result["chance_accuracy"] == pytest.approx(1.0 / len(IDENTITIES))
    assert result["manifest_sha256"] == "0" * 64


# --- what the gate separates ---------------------------------------------------


def test_a_signature_carried_only_by_the_call_leaves_the_backgrounds_at_chance() -> None:
    """The result a method has to produce: the animal scores, the place does not."""

    result = _run(identity_in_background=False)
    assert result["pairings"]["foreground_to_foreground"]["accuracy"] == 1.0
    assert result["pairings"]["background_to_background"]["accuracy"] < 0.5
    assert result["background_only_over_chance"] < 1.5


def test_a_signature_carried_by_the_place_is_read_off_the_backgrounds_alone() -> None:
    """The control a method has to fail, made to fail here on purpose."""

    result = _run(identity_in_background=True)
    assert result["pairings"]["background_to_background"]["accuracy"] == 1.0
    assert result["background_only_over_chance"] == pytest.approx(float(len(IDENTITIES)))
    assert result["pairings"]["foreground_to_background"]["accuracy"] == 1.0


def test_every_pairing_is_scored_under_every_head() -> None:
    result = _run(identity_in_background=False)
    assert set(result["pairings_by_head"]) == set(HEADS)
    for scored in result["pairings_by_head"].values():
        assert set(scored) == {f"{a}_to_{b}" for a, b in PAIRINGS}


def test_adding_the_other_heads_does_not_move_the_fitted_head_result() -> None:
    """Five gate results are already on record under the fitted head alone.
    The keys they carry, and the numbers in them, have to be what this still
    reports, or the new heads have rewritten published figures."""

    result = _run(identity_in_background=False)
    for name, scored in result["pairings"].items():
        under_ridge = result["pairings_by_head"][KERNEL_RIDGE][name]
        assert set(scored) == set(under_ridge) - {"macro_recall"}
        for key, value in scored.items():
            assert under_ridge[key] == value


def test_the_three_heads_read_a_hard_case_differently() -> None:
    """If the three heads always agreed, running three of them would cost time
    and say nothing. The default fixture is far too easy to show anything: every
    head reads it perfectly. Where the identities barely separate, they differ."""

    endpoint = _endpoint()
    result = run_validity_gate(
        endpoint=endpoint,
        vectors=_vectors(endpoint, identity_in_background=False, signature=0.12),
        representation="synthetic",
    )
    readings = {
        head: result["pairings_by_head"][head]["foreground_to_foreground"]["accuracy"]
        for head in HEADS
    }
    assert len(set(readings.values())) > 1


def test_the_seed_is_recorded_and_the_run_repeats() -> None:
    first = _run(identity_in_background=False)
    second = _run(identity_in_background=False)
    assert first["seed"] == second["seed"]
    assert first["pairings"] == second["pairings"]


def test_the_enrolment_fisher_ratio_reads_only_foreground_enrolment_clips() -> None:
    """Where one endpoint is built in several arms, as the right whale is at
    eleven frequency shifts, this names the reported arm. It must not move when
    a query clip or a background clip moves, or the choice is made on the
    answer."""

    endpoint = _endpoint()
    vectors = _vectors(endpoint, identity_in_background=False)
    baseline = run_validity_gate(endpoint=endpoint, vectors=vectors, representation="synthetic")[
        "foreground_enrolment_fisher_ratio"
    ]

    generator = np.random.default_rng(3)
    disturbed = vectors.copy()
    for index, record in enumerate(endpoint.records):
        if record.split == "query" or record.context["condition"] == "background":
            disturbed[index] = generator.normal(size=WIDTH) * 50.0
    moved = run_validity_gate(endpoint=endpoint, vectors=disturbed, representation="synthetic")[
        "foreground_enrolment_fisher_ratio"
    ]
    assert moved == pytest.approx(baseline)


def test_the_enrolment_fisher_ratio_separates_a_clean_arm_from_a_noisy_one() -> None:
    """If it read the same on every arm it could not choose between them."""

    endpoint = _endpoint()
    clean = run_validity_gate(
        endpoint=endpoint,
        vectors=_vectors(endpoint, identity_in_background=False, signature=4.0),
        representation="synthetic",
    )["foreground_enrolment_fisher_ratio"]
    noisy = run_validity_gate(
        endpoint=endpoint,
        vectors=_vectors(endpoint, identity_in_background=False, signature=0.12),
        representation="synthetic",
    )["foreground_enrolment_fisher_ratio"]
    assert clean > noisy


def test_requested_bootstrap_count_reaches_every_pairing_and_head(monkeypatch) -> None:
    from xinyenyana import validity

    counts = []

    def evaluate(**kwargs):
        counts.append(kwargs["bootstrap_replicates"])
        return {
            "classification": {"accuracy": 0.5, "macro_recall": 0.5},
            "identity_block_bootstrap_accuracy_95": [0.4, 0.6],
            "permutation_control": {"p_value_plus_one": 0.1},
            "verification": {"roc_auc": 0.5},
            "enrollment_calls": 1,
            "query_calls": 1,
        }

    monkeypatch.setattr(validity, "evaluate_endpoint", evaluate)
    endpoint = _endpoint()
    result = run_validity_gate(
        endpoint=endpoint,
        vectors=_vectors(endpoint, identity_in_background=False),
        representation="synthetic",
        bootstrap_replicates=10000,
    )
    assert counts == [10000] * (len(PAIRINGS) * len(HEADS))
    assert result["bootstrap_replicates"] == 10000
