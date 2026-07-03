"""Backtest a fixed-weight portfolio against a benchmark over historical data."""
import numpy as np
import pandas as pd

from mpt_finance.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)


def portfolio_returns(returns, weights):
    weights = np.asarray(weights)
    return returns.dot(weights)


def cumulative_growth(daily_returns):
    return (1 + daily_returns).cumprod()


def backtest_portfolio(returns, weights, benchmark_returns, risk_free_rate=0.0):
    port_daily_returns = portfolio_returns(returns, weights)
    aligned = pd.DataFrame({
        "portfolio": port_daily_returns,
        "benchmark": benchmark_returns,
    }).dropna(how="any")

    cumulative = pd.DataFrame({
        "portfolio": cumulative_growth(aligned["portfolio"]),
        "benchmark": cumulative_growth(aligned["benchmark"]),
    })

    summary = {}
    for label in ("portfolio", "benchmark"):
        series = aligned[label]
        summary[label] = {
            "annualized_return": annualized_return(series),
            "annualized_volatility": annualized_volatility(series),
            "sharpe_ratio": sharpe_ratio(series, risk_free_rate),
            "max_drawdown": max_drawdown(series),
        }

    return cumulative, summary
