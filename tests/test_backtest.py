import numpy as np
import pandas as pd
import pytest

from mpt_finance.backtest import backtest_portfolio, cumulative_growth, portfolio_returns
from mpt_finance.metrics import annualized_return

DATES = pd.date_range("2024-01-01", periods=5)
RETURNS = pd.DataFrame({
    "A": [0.01, 0.02, -0.01, 0.0, 0.03],
    "B": [0.00, 0.01, 0.01, 0.02, -0.02],
}, index=DATES)
BENCHMARK_RETURNS = pd.Series([0.005] * 5, index=DATES)
WEIGHTS = [0.5, 0.5]


def test_portfolio_returns_is_weighted_sum():
    result = portfolio_returns(RETURNS, WEIGHTS)
    expected = [0.005, 0.015, 0.0, 0.01, 0.005]
    assert result.tolist() == pytest.approx(expected)


def test_cumulative_growth_compounds_returns():
    daily = pd.Series([0.10, -0.10])
    result = cumulative_growth(daily)
    assert result.tolist() == pytest.approx([1.10, 0.99])


def test_backtest_portfolio_returns_cumulative_and_summary():
    cumulative, summary = backtest_portfolio(RETURNS, WEIGHTS, BENCHMARK_RETURNS, risk_free_rate=0.0)

    assert list(cumulative.columns) == ["portfolio", "benchmark"]
    assert len(cumulative) == 5

    port_daily = portfolio_returns(RETURNS, WEIGHTS)
    assert summary["portfolio"]["annualized_return"] == pytest.approx(annualized_return(port_daily))
    assert set(summary["portfolio"].keys()) == {
        "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"
    }
    assert set(summary.keys()) == {"portfolio", "benchmark"}


def test_backtest_portfolio_drops_misaligned_dates():
    short_benchmark = BENCHMARK_RETURNS.iloc[:3]
    cumulative, _ = backtest_portfolio(RETURNS, WEIGHTS, short_benchmark, risk_free_rate=0.0)
    assert len(cumulative) == 3
