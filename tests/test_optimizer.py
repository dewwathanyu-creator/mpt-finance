import numpy as np
import pytest

from mpt_finance.optimizer import (
    max_sharpe_weights,
    min_variance_weights,
    portfolio_performance,
    efficient_frontier,
)

# Two uncorrelated assets: A (mean 0.10, var 0.04), B (mean 0.15, var 0.09).
MEAN_RETURNS = np.array([0.10, 0.15])
COV_MATRIX = np.array([[0.04, 0.0], [0.0, 0.09]])


def test_portfolio_performance_matches_hand_calculation():
    weights = np.array([0.5, 0.5])
    port_return, port_vol, sharpe = portfolio_performance(
        weights, MEAN_RETURNS, COV_MATRIX, risk_free_rate=0.0
    )
    assert port_return == pytest.approx(0.125)
    expected_var = 0.5**2 * 0.04 + 0.5**2 * 0.09
    assert port_vol == pytest.approx(np.sqrt(expected_var))
    assert sharpe == pytest.approx(port_return / port_vol)


def test_min_variance_matches_analytical_solution_for_uncorrelated_assets():
    # For uncorrelated assets, min-variance weight is inversely proportional to variance.
    weights = min_variance_weights(MEAN_RETURNS, COV_MATRIX)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.09 / 0.13, abs=1e-3)
    assert weights[1] == pytest.approx(0.04 / 0.13, abs=1e-3)


def test_max_sharpe_matches_analytical_tangency_portfolio():
    # Unconstrained tangency weight w_i proportional to (mean_i - rf) / var_i.
    weights = max_sharpe_weights(MEAN_RETURNS, COV_MATRIX, risk_free_rate=0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.6, abs=1e-3)
    assert weights[1] == pytest.approx(0.4, abs=1e-3)


def test_efficient_frontier_returns_valid_points():
    frontier = efficient_frontier(MEAN_RETURNS, COV_MATRIX, num_points=10)
    assert len(frontier) == 10
    for point in frontier:
        assert point["weights"].sum() == pytest.approx(1.0, abs=1e-4)
        assert point["volatility"] >= 0
        assert (point["weights"] >= -1e-6).all()
