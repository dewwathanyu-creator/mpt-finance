import numpy as np
import pandas as pd
import pytest

from mpt_finance.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)


def test_annualized_return_zero_for_flat_returns():
    returns = pd.Series([0.0] * 252)
    assert annualized_return(returns) == pytest.approx(0.0)


def test_annualized_return_matches_compounded_growth_over_one_year():
    daily_rate = 0.001
    returns = pd.Series([daily_rate] * 252)
    expected = (1 + daily_rate) ** 252 - 1
    assert annualized_return(returns) == pytest.approx(expected, rel=1e-9)


def test_annualized_volatility_scales_daily_std_by_sqrt_252():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_sharpe_ratio_combines_return_and_volatility():
    returns = pd.Series([0.001] * 252)
    rf = 0.02
    expected = (annualized_return(returns) - rf) / annualized_volatility(returns)
    assert sharpe_ratio(returns, risk_free_rate=rf) == pytest.approx(expected)


def test_sharpe_ratio_zero_volatility_returns_zero():
    returns = pd.Series([0.0] * 10)
    assert sharpe_ratio(returns, risk_free_rate=0.0) == 0.0


def test_max_drawdown_on_known_price_path():
    # Prices: 100 -> 120 -> 90 -> 110 gives daily returns below.
    returns = pd.Series([0.20, -0.25, 0.2222222222])
    assert max_drawdown(returns) == pytest.approx(-0.25, abs=1e-6)


def test_value_at_risk_matches_percentile_definition():
    returns = pd.Series(np.arange(1, 101))
    confidence = 0.90
    expected = -np.percentile(returns, (1 - confidence) * 100)
    assert value_at_risk(returns, confidence=confidence) == pytest.approx(expected)
