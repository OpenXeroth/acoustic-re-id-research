"""A maximal minimum-entropy index need not identify an animal uniquely."""

import pytest

from xinyenyana.confound_index import axis_reading, verdicts


@pytest.mark.parametrize("reverse", [False, True])
def test_nested_variables_have_maximal_index_without_being_interchangeable(reverse: bool) -> None:
    pairs = [("a", "first"), ("b", "first"), ("c", "second"), ("d", "second")]
    if reverse:
        pairs = [(b, a) for a, b in pairs]
    row = axis_reading(pairs)
    assert row["nmi"] == 1.0
    assert row["identities"] != row["levels"]
    assert row["levels_with_two_or_more_identities"] == (0 if reverse else 2)
    assert row["identities_on_two_or_more_levels"] == (2 if reverse else 0)
    sentence = verdicts({"axes": {"cohort": {"all": row}}})["cohort"]
    assert "same variable" not in sentence
    assert "no result" not in sentence
    assert "index alone does not establish separability" in sentence


def test_constant_recorded_microphone_does_not_rule_out_unrecorded_differences() -> None:
    row = axis_reading([("a", "mic"), ("b", "mic")])
    assert row["nmi"] == 0.0
    sentence = verdicts({"axes": {"microphone": {"all": row}}})["microphone"]
    assert "association cannot be estimated" in sentence
    assert "unrecorded differences remain possible" in sentence
