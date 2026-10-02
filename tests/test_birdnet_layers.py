"""Tests for reading BirdNET's inner layers.

The parts that need the model and the published package are measured on the
machine that holds them, not here. What is tested here is the audio cutting and
the pooling, which are pure and are where a mistake would go unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from xinyenyana.representations import (
    BIRDNET_LAYERS,
    BIRDNET_SEGMENT_SAMPLES,
    birdnet_pad_segment,
    birdnet_segment_bounds,
    pool_layer,
)

# --- cutting the audio -------------------------------------------------------


def test_spans_are_cut_in_the_recording_own_rate_not_after_resampling() -> None:
    """Cutting after resampling reaches only a cosine of 0.999 against the package."""

    assert birdnet_segment_bounds(66_150, 22_050) == [(0, 66_150)]
    assert birdnet_segment_bounds(72_765, 22_050) == [(0, 66_150), (66_150, 72_765)]


def test_a_clip_shorter_than_a_span_is_one_span() -> None:
    assert birdnet_segment_bounds(1000, 22_050) == [(0, 1000)]


def test_a_clip_of_exactly_two_spans_is_two() -> None:
    assert len(birdnet_segment_bounds(132_300, 22_050)) == 2


def test_an_empty_signal_has_no_spans() -> None:
    with pytest.raises(ValueError, match="no segments"):
        birdnet_segment_bounds(0, 22_050)


def test_a_sample_rate_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        birdnet_segment_bounds(100, 0)


def test_a_short_span_is_padded_with_silence() -> None:
    """Padding is what the package does; tiling and stretching disagree with it."""

    signal = np.ones(1000, dtype=np.float32)
    padded = birdnet_pad_segment(signal)
    assert padded.shape == (BIRDNET_SEGMENT_SAMPLES,)
    assert np.array_equal(padded[:1000], signal)
    assert not padded[1000:].any()


def test_a_span_of_exactly_one_segment_is_unchanged() -> None:
    signal = np.linspace(-1.0, 1.0, BIRDNET_SEGMENT_SAMPLES, dtype=np.float32)
    assert np.array_equal(birdnet_pad_segment(signal), signal)


def test_a_stereo_array_is_refused_rather_than_silently_mixed() -> None:
    with pytest.raises(ValueError, match="one channel"):
        birdnet_pad_segment(np.zeros((2, 100), dtype=np.float32))


def test_an_empty_span_is_refused() -> None:
    with pytest.raises(ValueError, match="no segments"):
        birdnet_pad_segment(np.zeros(0, dtype=np.float32))


# --- pooling a layer ---------------------------------------------------------


def test_a_layer_is_averaged_over_time_and_frequency_leaving_its_channels() -> None:
    activation = np.arange(1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    pooled = pool_layer(activation, standard_deviation=False)
    assert pooled.shape == (4,)
    assert pooled == pytest.approx(activation.reshape(-1, 4).mean(axis=0))


def test_carrying_the_spread_doubles_the_width() -> None:
    activation = np.arange(1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    pooled = pool_layer(activation, standard_deviation=True)
    assert pooled.shape == (8,)
    assert pooled[:4] == pytest.approx(activation.reshape(-1, 4).mean(axis=0))
    assert pooled[4:] == pytest.approx(activation.reshape(-1, 4).std(axis=0))


def test_a_constant_layer_has_no_spread() -> None:
    """A layer that never varies must not look informative once the spread is added."""

    activation = np.full((1, 5, 2, 3), 7.0, dtype=np.float32)
    pooled = pool_layer(activation, standard_deviation=True)
    assert pooled[:3] == pytest.approx([7.0, 7.0, 7.0])
    assert pooled[3:] == pytest.approx([0.0, 0.0, 0.0])


def test_a_layer_the_network_already_pooled_passes_through() -> None:
    activation = np.arange(6, dtype=np.float32).reshape(1, 6)
    assert pool_layer(activation, standard_deviation=False) == pytest.approx(np.arange(6))


def test_a_layer_with_the_wrong_number_of_axes_is_refused() -> None:
    with pytest.raises(ValueError, match="4-axis grid"):
        pool_layer(np.zeros((1, 2, 3), dtype=np.float32), standard_deviation=False)


# --- the ladder --------------------------------------------------------------


def test_the_layer_ladder_ends_at_the_embedding_every_recorded_figure_uses() -> None:
    assert BIRDNET_LAYERS[-1][0] == "embedding"
    assert BIRDNET_LAYERS[-1][1] == "model/GLOBAL_AVG_POOL/Mean"


def test_every_layer_is_named_once() -> None:
    shorts = [short for short, _ in BIRDNET_LAYERS]
    names = [name for _, name in BIRDNET_LAYERS]
    assert len(set(shorts)) == len(shorts)
    assert len(set(names)) == len(names)
