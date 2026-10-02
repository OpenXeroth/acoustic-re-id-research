"""Tests for the open-set protocol: known bird, or stranger?"""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.open_set import (
    ALLOCATION_SEED,
    ALLOCATIONS,
    COHORT_SIZES,
    ROLES,
    STRANGER_BUDGETS,
    acceptance_curve,
    allocate_roles,
    calibrate_threshold,
    evaluate_observations,
    known_vs_unknown_auroc,
    normalise,
    run_open_set_endpoint,
    wilson_interval,
    write_masked_clip,
)

SAMPLE_RATE = 22_050


def _cohorts() -> dict[str, list[str]]:
    birds = iter(f"bird{index:02d}" for index in range(16))
    return {cohort: [next(birds) for _ in range(size)] for cohort, size in COHORT_SIZES.items()}


# --- how the birds are split into roles -------------------------------------


def test_every_allocation_is_a_partition_of_the_birds() -> None:
    cohorts = _cohorts()
    everyone = {name for names in cohorts.values() for name in names}
    for allocation in range(ALLOCATIONS):
        roles = allocate_roles(cohorts, seed=ALLOCATION_SEED, allocation=allocation)
        assert sorted(roles) == sorted(ROLES)
        assert all(len(roles[role]) == 4 for role in ROLES)
        dealt = [name for role in ROLES for name in roles[role]]
        assert len(dealt) == len(set(dealt)) == len(everyone)


def test_every_bird_holds_every_role_once_within_a_group_of_four() -> None:
    cohorts = _cohorts()
    for group in range(4):
        seen: dict[str, set[str]] = {}
        for shift in range(4):
            roles = allocate_roles(cohorts, seed=ALLOCATION_SEED, allocation=group * 4 + shift)
            for role, names in roles.items():
                for name in names:
                    seen.setdefault(name, set()).add(role)
        assert all(held == set(ROLES) for held in seen.values())


def test_the_year_pair_cohorts_are_balanced_across_the_roles() -> None:
    cohorts = _cohorts()
    largest = set(cohorts["2021-2022"])
    for allocation in range(ALLOCATIONS):
        roles = allocate_roles(cohorts, seed=ALLOCATION_SEED, allocation=allocation)
        for role in ROLES:
            assert len(largest & set(roles[role])) == 2


def test_the_sixteen_allocations_are_not_all_the_same() -> None:
    cohorts = _cohorts()
    drawn = {
        tuple(sorted(allocate_roles(cohorts, seed=ALLOCATION_SEED, allocation=index)["test_known"]))
        for index in range(ALLOCATIONS)
    }
    assert len(drawn) > 1


def test_a_sample_that_is_not_the_frozen_cohorts_is_refused() -> None:
    cohorts = _cohorts()
    cohorts["2020-2022"].append(cohorts["2021-2022"].pop())
    with pytest.raises(ValueError, match="frozen cohorts"):
        allocate_roles(cohorts, seed=ALLOCATION_SEED, allocation=0)


def test_an_allocation_outside_the_sixteen_is_refused() -> None:
    with pytest.raises(ValueError, match="allocation must be"):
        allocate_roles(_cohorts(), seed=ALLOCATION_SEED, allocation=ALLOCATIONS)


# --- the threshold ----------------------------------------------------------


def test_the_threshold_admits_no_more_strangers_than_the_target_allows() -> None:
    strangers = [0.1 * index for index in range(20)]
    threshold = calibrate_threshold(strangers, target_far=0.05)
    accepted = sum(value >= threshold for value in strangers) / len(strangers)
    assert accepted <= 0.05


def test_the_threshold_is_the_lowest_one_that_meets_the_target() -> None:
    strangers = [0.1 * index for index in range(20)]
    threshold = calibrate_threshold(strangers, target_far=0.05)
    lower = max(value for value in strangers if value < threshold)
    assert sum(value >= lower for value in strangers) / len(strangers) > 0.05


def test_a_target_of_zero_puts_the_threshold_above_every_stranger() -> None:
    strangers = [0.4, 0.6, 0.9]
    assert calibrate_threshold(strangers, target_far=0.0) > max(strangers)


def test_calibration_without_stranger_scores_is_refused() -> None:
    with pytest.raises(ValueError, match="stranger scores"):
        calibrate_threshold([], target_far=0.05)


# --- the statistics ---------------------------------------------------------


def test_a_rate_of_zero_still_has_an_interval_a_check_can_fail_on() -> None:
    low, high = wilson_interval(0, 40)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high > 0.05


def test_the_interval_stays_inside_zero_and_one_at_both_ends() -> None:
    for successes in (0, 40):
        low, high = wilson_interval(successes, 40)
        assert low >= -1e-12 and high <= 1 + 1e-12


def test_separated_scores_give_the_highest_possible_separation() -> None:
    assert known_vs_unknown_auroc([0.9, 0.8], [0.2, 0.1]) == 1.0


def test_identical_scores_give_a_coin() -> None:
    assert known_vs_unknown_auroc([0.5, 0.5], [0.5, 0.5]) == 0.5


# --- what counts as getting it right ----------------------------------------


def _observation(*, known: bool, score: float, top: str, identity: str) -> dict[str, object]:
    return {
        "partition": "test",
        "clip": f"{identity}-{score}",
        "identity": identity,
        "cohort": "2020-2021",
        "is_known": known,
        "top_identity": top,
        "maximum_score": score,
        "clipped_over_slice_threshold": False,
    }


def test_accepting_a_call_and_naming_the_wrong_bird_is_not_a_correct_accept() -> None:
    """At a waterhole this is a different failure from rejecting a bird you know."""

    observations = [
        _observation(known=True, score=0.9, top="wren", identity="robin"),
        _observation(known=False, score=0.1, top="wren", identity="stranger"),
    ]
    metrics = evaluate_observations(observations, threshold=0.5)
    assert metrics["known_correct_accepts"] == 0
    assert metrics["known_misidentifications"] == 1
    assert metrics["known_false_rejects"] == 0


def test_a_rejected_call_from_an_enrolled_bird_is_a_false_reject() -> None:
    observations = [
        _observation(known=True, score=0.2, top="robin", identity="robin"),
        _observation(known=False, score=0.1, top="robin", identity="stranger"),
    ]
    metrics = evaluate_observations(observations, threshold=0.5)
    assert metrics["known_false_rejects"] == 1
    assert metrics["known_misidentifications"] == 0


def test_balanced_accuracy_is_a_coin_when_everything_is_accepted() -> None:
    observations = [
        _observation(known=True, score=0.9, top="robin", identity="robin"),
        _observation(known=False, score=0.9, top="robin", identity="stranger"),
    ]
    metrics = evaluate_observations(observations, threshold=0.5)
    assert metrics["open_set_balanced_accuracy"] == 0.5


def test_evaluation_without_strangers_is_refused() -> None:
    with pytest.raises(ValueError, match="enrolled and stranger"):
        evaluate_observations(
            [_observation(known=True, score=0.9, top="robin", identity="robin")], threshold=0.5
        )


# --- the acceptance curve ---------------------------------------------------


def test_the_curve_reports_a_number_for_every_budget() -> None:
    observations = [
        _observation(known=True, score=0.9, top="robin", identity="robin"),
        _observation(known=False, score=0.1, top="robin", identity="stranger"),
    ]
    curve = acceptance_curve(observations)
    assert set(curve) == {f"{budget:.2f}" for budget in STRANGER_BUDGETS}


def test_separable_scores_reach_full_acceptance_at_every_budget() -> None:
    """Every enrolled call scores above every stranger, so no budget binds."""

    observations = [
        _observation(known=True, score=0.90 + index / 100, top=f"b{index}", identity=f"b{index}")
        for index in range(10)
    ] + [
        _observation(known=False, score=index / 100, top="b0", identity="stranger")
        for index in range(10)
    ]
    curve = acceptance_curve(observations)
    assert all(value == 1.0 for value in curve.values())


def test_a_tighter_budget_can_never_buy_more_acceptance() -> None:
    """The curve must be non-decreasing in the budget, or it is not a curve."""

    rng = np.random.default_rng(11)
    observations = [
        _observation(
            known=True,
            score=float(rng.normal(0.6, 0.2)),
            top=f"b{index % 4}",
            identity=f"b{index % 4}",
        )
        for index in range(40)
    ] + [
        _observation(known=False, score=float(rng.normal(0.5, 0.2)), top="b0", identity="stranger")
        for index in range(40)
    ]
    curve = acceptance_curve(observations)
    values = [curve[f"{budget:.2f}"] for budget in sorted(STRANGER_BUDGETS)]
    assert values == sorted(values)


def test_overlapping_scores_do_not_reach_full_acceptance_at_a_tight_budget() -> None:
    """Ten strangers score above every enrolled call, so a 0.01 budget admits none."""

    observations = [
        _observation(known=True, score=0.3, top=f"b{index}", identity=f"b{index}")
        for index in range(10)
    ] + [_observation(known=False, score=0.9, top="b0", identity="stranger") for _ in range(10)]
    curve = acceptance_curve(observations)
    assert curve["0.01"] == 0.0


def test_naming_the_wrong_bird_does_not_count_towards_the_curve() -> None:
    """Accepted but misnamed is not a correct accept anywhere in this module."""

    observations = [
        _observation(known=True, score=0.9, top="wren", identity="robin") for _ in range(4)
    ] + [_observation(known=False, score=0.1, top="wren", identity="stranger") for _ in range(4)]
    assert all(value == 0.0 for value in acceptance_curve(observations).values())


def test_a_curve_without_strangers_is_refused() -> None:
    with pytest.raises(ValueError, match="enrolled and stranger"):
        acceptance_curve([_observation(known=True, score=0.9, top="robin", identity="robin")])


# --- no query clip scales itself --------------------------------------------


def test_the_scaling_never_sees_a_clip_it_will_be_scored_on() -> None:
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [500.0, 600.0]])
    near = normalise(vectors, [0, 1], standardise=True)
    vectors[2] = [-9000.0, 9000.0]
    far = normalise(vectors, [0, 1], standardise=True)
    assert np.allclose(near[:2], far[:2])


# --- the mask ---------------------------------------------------------------


def _write_tone(path: Path, *, seconds: float = 0.4, frequency: float = 800.0) -> None:
    frames = int(SAMPLE_RATE * seconds)
    samples = array(
        "h",
        (
            int(12_000 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
            for index in range(frames)
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


def _record(path: Path, *, identity: str, split: str, onsets: list[float]) -> ClipRecord:
    return ClipRecord(
        filename=f"{identity}-{split}-{path.stem}",
        path=path,
        identity=identity,
        split=split,
        context={
            "year": "2020",
            "cohort": "2020-2021",
            "nest_x": 1.0,
            "nest_y": 2.0,
            "quality": {
                "duration_seconds": 0.4,
                "rms_dbfs": -30.0,
                "peak_dbfs": -10.0,
                "clipped_sample_fraction": 0.0,
                "absolute_dc_fraction": 0.0,
            },
            "annotation": {
                "onsets": onsets,
                "offsets": [value + 0.05 for value in onsets],
                "unit_durations": [0.05 for _ in onsets],
                "silence_durations": [0.02 for _ in onsets[:-1]],
                "lower_frequency_hz": 3000.0,
                "upper_frequency_hz": 5000.0,
            },
        },
    )


def test_the_mask_silences_the_annotated_notes_and_keeps_the_rest(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    _write_tone(source)
    record = _record(source, identity="robin", split="query", onsets=[0.1])
    masked = write_masked_clip(record, tmp_path / "masked" / "clip.wav")

    with wave.open(str(source), "rb") as handle:
        original = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    with wave.open(str(masked), "rb") as handle:
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnchannels() == 1
        after = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")

    assert len(after) == len(original)
    start = int(math.floor((0.1 - 0.02) * SAMPLE_RATE))
    end = int(math.ceil((0.15 + 0.02) * SAMPLE_RATE))
    assert not after[start:end].any()
    assert np.array_equal(after[:start], original[:start])
    assert np.array_equal(after[end:], original[end:])


# --- the runner, end to end -------------------------------------------------


def _sixteen_bird_endpoint(tmp_path: Path) -> Endpoint:
    names = [f"bird{index:02d}" for index in range(16)]
    cursor = 0
    cohorts: dict[str, list[str]] = {}
    for cohort, size in COHORT_SIZES.items():
        cohorts[cohort] = names[cursor : cursor + size]
        cursor += size
    records = []
    for cohort, members in cohorts.items():
        for identity in members:
            for index in range(20):
                split = "enrollment" if index < 10 else "query"
                path = tmp_path / identity / f"{index}.wav"
                _write_tone(path, frequency=300.0 + 40.0 * names.index(identity))
                record = _record(path, identity=identity, split=split, onsets=[0.1, 0.25])
                records.append(
                    ClipRecord(
                        filename=f"{identity}-{index}",
                        path=record.path,
                        identity=identity,
                        split=split,
                        context={**record.context, "cohort": cohort},
                    )
                )
    return Endpoint(
        name="synthetic-open-set",
        records=tuple(records),
        categorical_targets=("year", "cohort"),
        manifest_sha256="0" * 64,
        source_document="tests",
    )


def test_the_runner_scores_every_representation_on_all_sixteen_allocations(
    tmp_path: Path,
) -> None:
    """Drives the command's own entry point, not a stand-in for it.

    The embedding representations are left out because they need a downloaded
    model; every rule this module adds is exercised by the handcrafted three.
    """

    endpoint = _sixteen_bird_endpoint(tmp_path)
    summary = run_open_set_endpoint(
        endpoint=endpoint,
        scratch=tmp_path / "scratch",
        representations=("acquisition-context", "level-duration-clipping"),
    )
    assert summary["allocations"] == ALLOCATIONS
    assert summary["identities"] == 16
    assert summary["clips"] == 320
    for name in ("acquisition-context", "level-duration-clipping"):
        block = summary["representations"][name]
        assert len(block["allocations"]) == ALLOCATIONS
        for entry in block["allocations"]:
            observations = entry["observations"]
            assert {row["partition"] for row in observations["calibration"]} == {"calibration"}
            assert {row["partition"] for row in observations["test"]} == {"test"}
            assert {row["identity"] for row in observations["test"]} == set(
                entry["roles"]["test_known"] + entry["roles"]["test_unknown"]
            )
            assert {row["top_identity"] for row in observations["test"]} <= set(
                entry["roles"]["test_known"]
            )
            known = [row for row in observations["test"] if row["is_known"]]
            assert entry["closed_set_accuracy"] == sum(
                row["top_identity"] == row["identity"] for row in known
            ) / len(known)
            assert set(entry["best_acceptance_at_budget"]) == {
                f"{budget:.2f}" for budget in STRANGER_BUDGETS
            }
            assert set(entry["calibrated_on_other_birds"]) == {
                f"{budget:.2f}" for budget in STRANGER_BUDGETS
            }
            for budget in STRANGER_BUDGETS:
                point = entry["calibrated_on_other_birds"][f"{budget:.2f}"]
                assert (
                    evaluate_observations(observations["test"], threshold=point["threshold"])
                    == point["test"]
                )
                assert point["test"]["known_queries"] == 40
                assert point["test"]["unknown_queries"] == 40
                # a calibrated point can never beat the best the curve allows
                assert (
                    point["test"]["known_correct_accept_rate"]
                    <= entry["best_acceptance_at_budget"][f"{budget:.2f}"] + 1e-12
                    or point["test"]["unknown_false_accept_rate"] > budget
                )
        spread = block["allocation_sensitivity"]["known_vs_unknown_auroc"]
        assert spread["minimum"] <= spread["median"] <= spread["maximum"]
    assert summary["stranger_budgets"] == list(STRANGER_BUDGETS)
    assert "passing_allocations" not in summary["representations"]["acquisition-context"]


def test_an_endpoint_whose_birds_do_not_all_have_the_same_query_count_is_refused(
    tmp_path: Path,
) -> None:
    endpoint = _sixteen_bird_endpoint(tmp_path)
    trimmed = tuple(
        record
        for index, record in enumerate(endpoint.records)
        if not (record.identity == "bird00" and record.split == "query" and index % 2 == 0)
    )
    with pytest.raises(ValueError, match="same number of query clips"):
        run_open_set_endpoint(
            endpoint=Endpoint(
                name=endpoint.name,
                records=trimmed,
                categorical_targets=endpoint.categorical_targets,
                manifest_sha256=endpoint.manifest_sha256,
                source_document=endpoint.source_document,
            ),
            scratch=tmp_path / "scratch",
            representations=("acquisition-context",),
        )


def test_even_allocation_gives_every_bird_every_role_once_per_group() -> None:
    from xinyenyana.open_set import ALLOCATIONS, ROLES, allocate_roles_evenly

    birds = [f"rook{i:02d}" for i in range(11)]
    for group in range(ALLOCATIONS // 4):
        held: dict[str, set[str]] = {bird: set() for bird in birds}
        for shift in range(4):
            roles = allocate_roles_evenly(birds, seed=317, allocation=group * 4 + shift)
            assert sorted(b for role in ROLES for b in roles[role]) == birds
            assert max(len(v) for v in roles.values()) - min(len(v) for v in roles.values()) <= 1
            for role, members in roles.items():
                for bird in members:
                    held[bird].add(role)
        assert all(value == set(ROLES) for value in held.values())


def test_even_allocation_refuses_too_few_birds() -> None:
    import pytest

    from xinyenyana.open_set import allocate_roles_evenly

    with pytest.raises(ValueError):
        allocate_roles_evenly([f"b{i}" for i in range(7)], seed=317, allocation=0)


def test_equal_error_rate_is_zero_when_separated_and_a_half_when_not() -> None:
    from xinyenyana.open_set import equal_error_rate

    assert equal_error_rate([0.9, 0.8, 0.95], [0.1, 0.2, 0.3]) == 0.0
    assert equal_error_rate([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.5


def test_true_accept_holds_strangers_to_the_budget() -> None:
    from xinyenyana.open_set import true_accept_at_false_accept

    known = [0.9, 0.8, 0.7, 0.6, 0.5]
    unknown = [0.1, 0.2, 0.3, 0.4, 0.75]
    # With no stranger allowed, the threshold must sit above 0.75: two of five known pass.
    assert true_accept_at_false_accept(known, unknown, budget=0.0) == 0.4
    # One stranger in five allowed: the threshold can drop to 0.5 and all five pass.
    assert true_accept_at_false_accept(known, unknown, budget=0.2) == 1.0
