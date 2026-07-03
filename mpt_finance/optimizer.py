"""Modern Portfolio Theory optimization: efficient frontier, max Sharpe, min variance."""
import numpy as np
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252


def annualize_returns_and_covariance(daily_returns):
    mean_returns = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = daily_returns.cov() * TRADING_DAYS_PER_YEAR
    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    weights = np.asarray(weights)
    means = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    port_return = float(weights @ means)
    port_volatility = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0.0
    return port_return, port_volatility, sharpe


def _weight_bounds_and_sum_constraint(num_assets):
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def min_variance_weights(mean_returns, cov_matrix):
    means = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    num_assets = len(means)
    init_guess = np.repeat(1.0 / num_assets, num_assets)
    bounds, constraints = _weight_bounds_and_sum_constraint(num_assets)

    def portfolio_variance(w):
        return w @ cov @ w

    result = minimize(
        portfolio_variance, init_guess, method="SLSQP",
        bounds=bounds, constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Min-variance optimization failed: {result.message}")
    return result.x


def max_sharpe_weights(mean_returns, cov_matrix, risk_free_rate=0.0):
    means = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    num_assets = len(means)
    init_guess = np.repeat(1.0 / num_assets, num_assets)
    bounds, constraints = _weight_bounds_and_sum_constraint(num_assets)

    def negative_sharpe(w):
        _, _, sharpe = portfolio_performance(w, means, cov, risk_free_rate)
        return -sharpe

    result = minimize(
        negative_sharpe, init_guess, method="SLSQP",
        bounds=bounds, constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Max-Sharpe optimization failed: {result.message}")
    return result.x


def efficient_frontier(mean_returns, cov_matrix, num_points=50):
    means = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    num_assets = len(means)
    target_returns = np.linspace(means.min(), means.max(), num_points)

    frontier = []
    for target in target_returns:
        init_guess = np.repeat(1.0 / num_assets, num_assets)
        bounds, sum_constraint = _weight_bounds_and_sum_constraint(num_assets)
        constraints = sum_constraint + [
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, means) - target}
        ]

        def portfolio_variance(w):
            return w @ cov @ w

        result = minimize(
            portfolio_variance, init_guess, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        if result.success:
            frontier.append({
                "return": float(target),
                "volatility": float(np.sqrt(result.fun)),
                "weights": result.x,
            })
    if not frontier:
        raise ValueError(
            "Could not compute an efficient frontier: no target return produced a "
            "feasible portfolio (this can happen when all assets have very similar "
            "expected returns)."
        )
    return frontier
