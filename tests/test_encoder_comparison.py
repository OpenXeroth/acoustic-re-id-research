"""The paired encoder test and the rank-consistency test, each able to fail."""

from __future__ import annotations

import numpy as np

from xinyenyana.encoder_comparison import holm, paired_bootstrap, rank_consistency


def _flags(values: list[int]) -> str:
    return "".join(str(v) for v in values)


def _animals(count: int, clips: int) -> list[str]:
    return [f"bird{i:02d}" for i in range(count) for _ in range(clips)]


def test_an_encoder_identical_to_the_reference_is_not_different() -> None:
    identities = _animals(10, 20)
    rng = np.random.default_rng(1)
    same = _flags(list(rng.integers(0, 2, size=len(identities))))
    result = paired_bootstrap(identities=identities, correct={"birdnet-v2.4": same, "copy": same})
    entry = result["comparisons"]["copy"]
    assert entry["difference_from_reference"] == 0.0
    assert entry["p_two_sided"] == 1.0
    assert not entry["differs_at_0.05_after_holm"]


def test_an_encoder_better_on_every_animal_is_found_better() -> None:
    identities = _animals(10, 20)
    reference = _flags([1 if i % 4 == 0 else 0 for i in range(len(identities))])
    better = _flags([1 if i % 4 != 3 else 0 for i in range(len(identities))])
    result = paired_bootstrap(
        identities=identities, correct={"birdnet-v2.4": reference, "better": better}
    )
    entry = result["comparisons"]["better"]
    assert entry["difference_from_reference"] > 0.4
    assert entry["difference_95"][0] > 0
    assert entry["differs_at_0.05_after_holm"]


def test_a_difference_carried_by_one_animal_does_not_survive() -> None:
    """Better on one bird of ten and identical elsewhere: the animals decide, not the clips."""

    identities = _animals(10, 40)
    reference = [0] * len(identities)
    lucky = list(reference)
    for i in range(40):
        lucky[i] = 1
    result = paired_bootstrap(
        identities=identities,
        correct={"birdnet-v2.4": _flags(reference), "lucky": _flags(lucky)},
    )
    assert result["comparisons"]["lucky"]["difference_95"][0] == 0.0
    assert not result["comparisons"]["lucky"]["differs_at_0.05_after_holm"]


def test_holm_never_lowers_a_p_value_and_keeps_the_order() -> None:
    comparisons = {name: {"p_two_sided": p} for name, p in [("a", 0.01), ("b", 0.02), ("c", 0.5)]}
    holm(comparisons)
    assert comparisons["a"]["p_holm"] == 0.03
    assert comparisons["b"]["p_holm"] == 0.04
    assert comparisons["c"]["p_holm"] == 0.5
    assert all(comparisons[n]["p_holm"] >= comparisons[n]["p_two_sided"] for n in comparisons)


def test_identical_rankings_everywhere_give_full_concordance() -> None:
    accuracy = {
        f"endpoint{e}": {"a": 0.9 - e * 0.01, "b": 0.7 - e * 0.01, "c": 0.5, "d": 0.3}
        for e in range(6)
    }
    result = rank_consistency(accuracy)
    assert result["kendall_w"] == 1.0
    assert result["friedman_p"] < 0.01


def test_rankings_that_disagree_give_low_concordance() -> None:
    orders = [
        ("a", "b", "c", "d"),
        ("d", "c", "b", "a"),
        ("b", "d", "a", "c"),
        ("c", "a", "d", "b"),
    ]
    accuracy = {
        f"endpoint{e}": {name: 1.0 - 0.1 * position for position, name in enumerate(order)}
        for e, order in enumerate(orders)
    }
    result = rank_consistency(accuracy)
    assert result["kendall_w"] < 0.2
    assert result["friedman_p"] > 0.5


def test_a_representation_missing_from_one_endpoint_is_named_and_left_out() -> None:
    accuracy = {
        "one": {"a": 0.9, "b": 0.5, "c": 0.3, "x": 0.1},
        "two": {"a": 0.8, "b": 0.6, "c": 0.2},
    }
    result = rank_consistency(accuracy)
    assert result["left_out_not_scored_everywhere"] == ["x"]
    assert "x" not in result["representations"]
