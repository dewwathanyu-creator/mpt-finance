import numpy as np
import pytest

from mpt_finance.simulate import random_portfolios

MEAN_RETURNS = np.array([0.10, 0.15, 0.08])
COV_MATRIX = np.array([
    [0.04, 0.01, 0.0],
    [0.01, 0.09, 0.02],
    [0.0, 0.02, 0.03],
])


def test_random_portfolios_returns_requested_count():
    result = random_portfolios(MEAN_RETURNS, COV_MATRIX, num_portfolios=200, seed=42)
    assert len(result) == 200
    assert set(result.columns) == {"return", "volatility", "sharpe", "weights"}


def test_random_portfolios_weights_sum_to_one():
    result = random_portfolios(MEAN_RETURNS, COV_MATRIX, num_portfolios=50, seed=1)
    for weights in result["weights"]:
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()


def test_random_portfolios_volatility_non_negative():
    result = random_portfolios(MEAN_RETURNS, COV_MATRIX, num_portfolios=50, seed=1)
    assert (result["volatility"] >= 0).all()


def test_random_portfolios_reproducible_with_seed():
    first = random_portfolios(MEAN_RETURNS, COV_MATRIX, num_portfolios=20, seed=7)
    second = random_portfolios(MEAN_RETURNS, COV_MATRIX, num_portfolios=20, seed=7)
    assert first["return"].tolist() == pytest.approx(second["return"].tolist())
