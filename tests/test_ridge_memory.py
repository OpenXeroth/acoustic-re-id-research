"""The smaller ridge system must preserve scores and avoid a sample-sized solve."""

from typing import Any

import numpy as np
import pytest

from xinyenyana.evaluation import PER_DIMENSION, leave_one_identity_out, ridge_projection


@pytest.mark.parametrize("samples,width", [(40, 3), (10, 30), (15, 15)])
def test_projection_agrees_with_the_original_dual(samples: int, width: int) -> None:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(samples, width))
    query = rng.normal(size=(8, width))
    original = np.linalg.solve(train @ train.T + np.eye(samples), train @ query.T).T
    assert ridge_projection(train, query, 1.0) == pytest.approx(original, abs=1e-12)


def test_a_tall_control_never_solves_the_sample_kernel(monkeypatch: Any) -> None:
    solve = np.linalg.solve
    sizes = []

    def checked(matrix: Any, rhs: Any) -> Any:
        sizes.append(matrix.shape)
        assert matrix.shape == (3, 3)
        return solve(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", checked)
    rng = np.random.default_rng(17)
    train = rng.normal(size=(120, 3))
    result = ridge_projection(train, train[:8], 1.0)
    assert result.shape == (8, 120)
    probe = leave_one_identity_out(
        features=train,
        identities=np.repeat(np.arange(6), 20),
        targets={"planted": train[:, 0]},
        standardisation=PER_DIMENSION,
        ridge_lambda=1.0,
    )
    assert probe["planted"]["r_squared"] > 0.99
    assert len(sizes) == 7


def test_zero_regularisation_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        ridge_projection(np.zeros((4, 1)), np.zeros((1, 1)), 0.0)
