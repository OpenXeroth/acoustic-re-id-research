"""Temporal extraction preserves ordering and applies the registered uniform-grid crop."""

import numpy as np
import pytest

from xinyenyana.birdnet_reader import time_positions


def test_time_positions_average_only_frequency_and_keep_call_order() -> None:
    grid = np.arange(1 * 6 * 3 * 2, dtype=np.float32).reshape(1, 6, 3, 2)
    series = time_positions(grid, 1.0)
    assert series.shape == (6, 2)
    assert series == pytest.approx(grid[0].mean(axis=1))


def test_synthetic_tail_does_not_become_a_sequence_of_extra_observations() -> None:
    grid = np.arange(1 * 6 * 3 * 2, dtype=np.float32).reshape(1, 6, 3, 2)
    series = time_positions(grid, 0.4)
    assert series.shape == (3, 2)
    assert series == pytest.approx(grid[0, :3].mean(axis=1))


def test_a_very_short_call_keeps_one_observation() -> None:
    assert time_positions(np.ones((1, 6, 3, 2)), 0.001).shape == (1, 2)


def test_an_already_pooled_embedding_is_one_observation() -> None:
    assert time_positions(np.ones((1, 1024)), 0.1).shape == (1, 1024)


def test_wrong_grid_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="spatial activation"):
        time_positions(np.ones((2, 3, 4)), 1.0)
