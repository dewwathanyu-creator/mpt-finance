"""Return- and risk-based performance metrics for a daily return series."""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns: pd.Series) -> float:
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0
    total_growth = (1 + daily_returns).prod()
    return total_growth ** (TRADING_DAYS_PER_YEAR / n_days) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    vol = annualized_volatility(daily_returns)
    if vol == 0:
        return 0.0
    return (annualized_return(daily_returns) - risk_free_rate) / vol


def max_drawdown(daily_returns: pd.Series) -> float:
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    return -np.percentile(daily_returns, (1 - confidence) * 100)
