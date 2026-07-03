"""Monte Carlo simulation of randomly weighted portfolios for MPT visualization."""
import numpy as np
import pandas as pd

from mpt_finance.optimizer import portfolio_performance


def random_portfolios(mean_returns, cov_matrix, risk_free_rate=0.0, num_portfolios=5000, seed=None):
    rng = np.random.default_rng(seed)
    num_assets = len(np.asarray(mean_returns))
    records = []
    for _ in range(num_portfolios):
        weights = rng.random(num_assets)
        weights /= weights.sum()
        port_return, port_volatility, sharpe = portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        records.append({
            "return": port_return,
            "volatility": port_volatility,
            "sharpe": sharpe,
            "weights": weights,
        })
    return pd.DataFrame(records)
