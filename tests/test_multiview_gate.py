"""Association must recover a known delay and refuse separate emissions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xinyenyana.multiview_gate import (
    Window,
    correlation_peak,
    eligible_windows,
    infer_groups,
    pair_association,
    single_onset_channel,
)


def test_gain_and_delay_are_recovered_without_changing_the_waveform() -> None:
    rate = 48000
    a = np.random.default_rng(18).normal(size=rate)
    b = np.r_[np.zeros(12), a[:-12]] * 0.25
    peak = correlation_peak(a, b, rate)
    assert peak["correlation"] > 0.99
    assert peak["lag_seconds"] == pytest.approx(-12 / rate)
    assert pair_association(a, b, rate)["accepted"]


def test_different_noise_emissions_are_not_associated() -> None:
    rng = np.random.default_rng(17)
    result = pair_association(rng.normal(size=4800), rng.normal(size=4800), 48000)
    assert not result["accepted"]


def test_periodic_ambiguity_fails_the_competing_peak_rule() -> None:
    signal = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
    result = pair_association(signal, signal, 48000)
    assert result["correlation"] > 0.99
    assert not result["accepted"]


def test_changing_delay_across_halves_is_refused() -> None:
    a = np.random.default_rng(23).normal(size=48000)
    b = np.r_[a[:24000], np.zeros(240), a[24000:-240]]
    assert not pair_association(a, b, 48000)["accepted"]


def test_zero_energy_cannot_create_a_spurious_perfect_match() -> None:
    assert not pair_association(np.zeros(4800), np.zeros(4800), 48000)["accepted"]


def test_two_separated_envelope_onsets_fail_the_single_onset_screen() -> None:
    rng = np.random.default_rng(3)
    signal = np.zeros(48000)
    signal[4800:12000] = rng.uniform(-0.1, 0.1, 7200)
    assert single_onset_channel(signal, 48000)["accepted"]
    signal[28800:36000] = rng.uniform(-0.1, 0.1, 7200)
    result = single_onset_channel(signal, 48000)
    assert result["dominant_onset_runs"] == 2
    assert not result["accepted"]


def test_unknown_identity_is_not_required_by_the_annotation_gate() -> None:
    windows = [Window(2, 1, 2, False), Window(3, 3, 4, False)]
    eligible, reasons = eligible_windows(windows)
    assert eligible == windows
    assert reasons == {}


def test_overlap_margin_and_multiple_emitters_are_counted() -> None:
    windows = [
        Window(2, 1, 2, False),
        Window(3, 2.03, 3, False),
        Window(4, 4, 5, True),
        Window(5, 6, 7, False),
    ]
    eligible, reasons = eligible_windows(windows)
    assert [w.row for w in eligible] == [5]
    assert set(reasons) == {2, 3, 4}


def test_invalid_interval_is_an_error_not_a_failed_scientific_gate() -> None:
    with pytest.raises(ValueError, match="interval"):
        eligible_windows([Window(2, 2, 1, False)])


def _matrix() -> Any:
    matrix = np.ones((4, 4)) * 0.1
    np.fill_diagonal(matrix, 1.0)
    matrix[0, 2] = matrix[2, 0] = 0.7
    matrix[1, 3] = matrix[3, 1] = 0.6
    return matrix


def test_grouping_recovers_nonadjacent_pairs_without_claiming_hardware() -> None:
    result = infer_groups(_matrix())
    assert result["accepted"]
    assert result["inferred_pairs"] == [[0, 2], [1, 3]]
    assert not result["hardware_mapping_verified"]


def test_ambiguous_groups_are_not_resolved_by_channel_order() -> None:
    assert not infer_groups(np.ones((4, 4)))["accepted"]


def test_two_channels_have_no_competing_partition_but_still_need_correlation() -> None:
    assert infer_groups(np.ones((2, 2)))["accepted"]
    assert not infer_groups(np.eye(2))["accepted"]


def test_a_geometry_rejection_names_the_number_it_missed() -> None:
    """A rejection that carries a null reason tells nobody anything.

    Sixteen of the eighty-one recordings in the first full run were rejected,
    and ten of those carried no reason at all.
    """

    from xinyenyana.multiview_gate import CONFIG, infer_groups

    minimum = float(CONFIG["geometry_pair_median_minimum"])
    margin = float(CONFIG["geometry_pairing_margin"])

    weak = np.full((4, 4), 0.1)
    weak[0, 1] = weak[1, 0] = minimum - 0.05
    weak[2, 3] = weak[3, 2] = minimum - 0.05
    rejected = infer_groups(weak)
    assert rejected["accepted"] is False
    assert "below" in rejected["reason"]
    assert f"{minimum:.2f}" in rejected["reason"]

    close = np.full((4, 4), 0.80)
    close[0, 1] = close[1, 0] = 0.80 + margin / 4
    ambiguous = infer_groups(close)
    assert ambiguous["accepted"] is False
    assert "leads the next by" in ambiguous["reason"]

    clear = np.full((4, 4), 0.2)
    clear[0, 1] = clear[1, 0] = 0.9
    clear[2, 3] = clear[3, 2] = 0.9
    accepted = infer_groups(clear)
    assert accepted["accepted"] is True
    assert accepted["reason"] is None
    assert accepted["weakest_pair_correlation"] == pytest.approx(0.9)
