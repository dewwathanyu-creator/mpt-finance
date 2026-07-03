# MPT Finance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Modern Portfolio Theory (MPT) portfolio optimizer as a Streamlit web app, backed by a tested Python package, deployable to Streamlit Community Cloud as a GitHub portfolio project.

**Architecture:** A `mpt_finance` Python package holds all calculation logic (data fetching, optimization, Monte Carlo simulation, backtesting, metrics) as small pure/testable modules. `app.py` is a thin Streamlit UI layer that only wires user input to the package and renders results — it contains no calculation logic itself.

**Tech Stack:** Python 3.13, `yfinance`, `numpy`, `pandas`, `scipy.optimize`, `plotly`, `streamlit`, `pytest`.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `mpt_finance/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
streamlit>=1.32
yfinance>=0.2.40
pandas>=2.0
numpy>=1.24
scipy>=1.11
plotly>=5.18
pytest>=7.4
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
.streamlit/secrets.toml
.pytest_cache/
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
```

This makes `mpt_finance` importable from test files regardless of which directory `pytest` is invoked from.

- [ ] **Step 4: Create the package init file**

`mpt_finance/__init__.py`:

```python
"""MPT Finance: Modern Portfolio Theory optimization toolkit."""
```

- [ ] **Step 5: Create a virtual environment and install dependencies**

Run:
```bash
cd "/c/Users/acer/Desktop/Claude-Project/MPT-Finance"
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```
Expected: dependencies install without error.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore pytest.ini mpt_finance/__init__.py
git commit -m "chore: project scaffolding"
```

---

## Task 2: Metrics Module

**Files:**
- Create: `mpt_finance/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mpt_finance.metrics'`

- [ ] **Step 3: Implement `mpt_finance/metrics.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add mpt_finance/metrics.py tests/test_metrics.py
git commit -m "feat: add performance metrics module"
```

---

## Task 3: Optimizer Module

**Files:**
- Create: `mpt_finance/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_optimizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mpt_finance.optimizer'`

- [ ] **Step 3: Implement `mpt_finance/optimizer.py`**

```python
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
    return frontier
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimizer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add mpt_finance/optimizer.py tests/test_optimizer.py
git commit -m "feat: add MPT optimizer (efficient frontier, max Sharpe, min variance)"
```

---

## Task 4: Monte Carlo Simulation Module

**Files:**
- Create: `mpt_finance/simulate.py`
- Test: `tests/test_simulate.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_simulate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mpt_finance.simulate'`

- [ ] **Step 3: Implement `mpt_finance/simulate.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add mpt_finance/simulate.py tests/test_simulate.py
git commit -m "feat: add Monte Carlo portfolio simulation"
```

---

## Task 5: Data Loader Module

**Files:**
- Create: `mpt_finance/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_data_loader.py`:

```python
import pandas as pd
import pytest

from mpt_finance import data_loader


def _fake_price_frame(values):
    dates = pd.date_range("2024-01-01", periods=len(values))
    return pd.DataFrame({"Adj Close": values}, index=dates)


def test_fetch_price_data_returns_prices_for_valid_tickers(monkeypatch):
    frames = {
        "AAPL": _fake_price_frame([100, 101, 102]),
        "MSFT": _fake_price_frame([200, 202, 201]),
    }

    def fake_download(ticker, start=None, end=None, progress=False):
        return frames[ticker]

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    prices, dropped = data_loader.fetch_price_data(["AAPL", "MSFT"], "2024-01-01", "2024-01-03")

    assert dropped == []
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert prices.shape[0] == 3


def test_fetch_price_data_drops_invalid_ticker(monkeypatch):
    frames = {
        "AAPL": _fake_price_frame([100, 101, 102]),
        "MSFT": _fake_price_frame([200, 202, 201]),
        "FAKE": pd.DataFrame(),
    }

    def fake_download(ticker, start=None, end=None, progress=False):
        return frames[ticker]

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    prices, dropped = data_loader.fetch_price_data(["AAPL", "MSFT", "FAKE"], "2024-01-01", "2024-01-03")

    assert dropped == ["FAKE"]
    assert list(prices.columns) == ["AAPL", "MSFT"]


def test_fetch_price_data_raises_when_fewer_than_two_valid_tickers(monkeypatch):
    frames = {
        "AAPL": _fake_price_frame([100, 101, 102]),
        "FAKE1": pd.DataFrame(),
        "FAKE2": pd.DataFrame(),
    }

    def fake_download(ticker, start=None, end=None, progress=False):
        return frames[ticker]

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    with pytest.raises(data_loader.InsufficientDataError):
        data_loader.fetch_price_data(["AAPL", "FAKE1", "FAKE2"], "2024-01-01", "2024-01-03")


def test_fetch_benchmark_prices_returns_series(monkeypatch):
    monkeypatch.setattr(
        data_loader.yf, "download",
        lambda ticker, start=None, end=None, progress=False: _fake_price_frame([1000, 1010, 1005]),
    )

    series = data_loader.fetch_benchmark_prices("^GSPC", "2024-01-01", "2024-01-03")

    assert series.tolist() == [1000, 1010, 1005]


def test_fetch_benchmark_prices_raises_when_empty(monkeypatch):
    monkeypatch.setattr(
        data_loader.yf, "download",
        lambda ticker, start=None, end=None, progress=False: pd.DataFrame(),
    )

    with pytest.raises(data_loader.InsufficientDataError):
        data_loader.fetch_benchmark_prices("BADTICKER", "2024-01-01", "2024-01-03")


def test_compute_daily_returns_matches_pct_change():
    prices = pd.DataFrame({"AAPL": [100, 110, 99], "MSFT": [200, 190, 209]})
    returns = data_loader.compute_daily_returns(prices)
    assert returns["AAPL"].tolist() == pytest.approx([0.10, -0.10])
    assert returns["MSFT"].tolist() == pytest.approx([-0.05, 0.10])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mpt_finance.data_loader'`

- [ ] **Step 3: Implement `mpt_finance/data_loader.py`**

```python
"""Fetch and validate price data for portfolio construction."""
import pandas as pd
import yfinance as yf


class InsufficientDataError(Exception):
    """Raised when there isn't enough valid price data to build a portfolio."""


def fetch_price_data(tickers, start, end):
    valid_prices = {}
    dropped_tickers = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty or "Adj Close" not in data.columns:
            dropped_tickers.append(ticker)
            continue
        valid_prices[ticker] = data["Adj Close"]

    if len(valid_prices) < 2:
        raise InsufficientDataError(
            f"Only {len(valid_prices)} valid ticker(s) after dropping {dropped_tickers}; need at least 2."
        )

    prices = pd.DataFrame(valid_prices).dropna(how="any")
    return prices, dropped_tickers


def fetch_benchmark_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty or "Adj Close" not in data.columns:
        raise InsufficientDataError(f"No data available for benchmark ticker '{ticker}'.")
    return data["Adj Close"]


def compute_daily_returns(prices):
    return prices.pct_change().dropna(how="any")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_loader.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add mpt_finance/data_loader.py tests/test_data_loader.py
git commit -m "feat: add price data loader with validation"
```

---

## Task 6: Backtest Module

**Files:**
- Create: `mpt_finance/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_backtest.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mpt_finance.backtest'`

- [ ] **Step 3: Implement `mpt_finance/backtest.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add mpt_finance/backtest.py tests/test_backtest.py
git commit -m "feat: add backtest module comparing portfolio to benchmark"
```

---

## Task 7: Streamlit App

**Files:**
- Create: `app.py`
- Create: `.streamlit/config.toml`

- [ ] **Step 1: Create `.streamlit/config.toml`**

```toml
[theme]
primaryColor = "#0E4C92"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

- [ ] **Step 2: Implement `app.py`**

```python
"""Streamlit UI for the MPT Finance portfolio optimizer."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mpt_finance.backtest import backtest_portfolio
from mpt_finance.data_loader import (
    InsufficientDataError,
    compute_daily_returns,
    fetch_benchmark_prices,
    fetch_price_data,
)
from mpt_finance.optimizer import (
    annualize_returns_and_covariance,
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_performance,
)
from mpt_finance.simulate import random_portfolios

st.set_page_config(page_title="MPT Finance", layout="wide")
st.title("MPT Finance — Portfolio Optimizer")
st.caption(
    "Find the asset mix that gives the best expected return for a given level of risk, "
    "based on Modern Portfolio Theory."
)

with st.sidebar:
    st.header("Portfolio Inputs")
    tickers_input = st.text_input(
        "Tickers (comma-separated)", value="PTT.BK, KBANK.BK, AOT.BK, CPALL.BK"
    )
    start_date = st.date_input("Start date", value=pd.Timestamp.today() - pd.DateOffset(years=3))
    end_date = st.date_input("End date", value=pd.Timestamp.today())
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)", value=0.02, step=0.005, format="%.3f"
    )
    benchmark_ticker = st.text_input("Benchmark ticker", value="^SET.BK")
    run_button = st.button("Build Portfolio")

if run_button:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers.")
        st.stop()

    try:
        prices, dropped = fetch_price_data(tickers, start_date, end_date)
    except InsufficientDataError as exc:
        st.error(str(exc))
        st.stop()

    if dropped:
        st.warning(f"Dropped tickers with no data: {', '.join(dropped)}")

    requested_start = pd.Timestamp(start_date)
    if not prices.empty and prices.index.min() > requested_start + pd.Timedelta(days=5):
        st.info(
            f"Data only available from {prices.index.min().date()} onward for one or more "
            "tickers; the analysis uses this shorter overlapping date range."
        )

    returns = compute_daily_returns(prices)
    mean_returns, cov_matrix = annualize_returns_and_covariance(returns)

    try:
        max_sharpe_w = max_sharpe_weights(mean_returns, cov_matrix, risk_free_rate)
        min_var_w = min_variance_weights(mean_returns, cov_matrix)
        frontier = efficient_frontier(mean_returns, cov_matrix)
    except ValueError as exc:
        st.error(f"Optimization failed: {exc}")
        st.stop()

    mc_portfolios = random_portfolios(mean_returns, cov_matrix, risk_free_rate)

    tab_frontier, tab_corr, tab_backtest, tab_weights = st.tabs(
        ["Efficient Frontier & Monte Carlo", "Correlation", "Backtest vs Benchmark", "Portfolio Weights"]
    )

    with tab_frontier:
        max_sharpe_perf = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix, risk_free_rate)
        min_var_perf = portfolio_performance(min_var_w, mean_returns, cov_matrix, risk_free_rate)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mc_portfolios["volatility"], y=mc_portfolios["return"],
            mode="markers",
            marker=dict(size=5, color=mc_portfolios["sharpe"], colorscale="Viridis", showscale=True),
            name="Random portfolios",
        ))
        fig.add_trace(go.Scatter(
            x=[p["volatility"] for p in frontier], y=[p["return"] for p in frontier],
            mode="lines", line=dict(color="black", width=2), name="Efficient frontier",
        ))
        fig.add_trace(go.Scatter(
            x=[max_sharpe_perf[1]], y=[max_sharpe_perf[0]],
            mode="markers", marker=dict(size=14, color="red", symbol="star"), name="Max Sharpe",
        ))
        fig.add_trace(go.Scatter(
            x=[min_var_perf[1]], y=[min_var_perf[0]],
            mode="markers", marker=dict(size=14, color="blue", symbol="diamond"), name="Min Variance",
        ))
        fig.update_layout(xaxis_title="Volatility (annualized)", yaxis_title="Expected Return (annualized)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        corr = returns.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "Pairs closer to -1 (blue) diversify each other well; pairs close to +1 (red) "
            "move together and add less diversification benefit."
        )

    with tab_backtest:
        try:
            benchmark_prices = fetch_benchmark_prices(benchmark_ticker, start_date, end_date)
            benchmark_returns = benchmark_prices.pct_change().dropna()
        except InsufficientDataError as exc:
            benchmark_returns = None
            st.warning(f"{exc} Skipping backtest.")

        if benchmark_returns is not None:
            cumulative, summary = backtest_portfolio(returns, max_sharpe_w, benchmark_returns, risk_free_rate)
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=cumulative.index, y=cumulative["portfolio"], name="Max Sharpe Portfolio"))
            fig_bt.add_trace(go.Scatter(x=cumulative.index, y=cumulative["benchmark"], name="Benchmark"))
            fig_bt.update_layout(yaxis_title="Growth of 1 unit")
            st.plotly_chart(fig_bt, use_container_width=True)
            st.dataframe(pd.DataFrame(summary).T.style.format("{:.2%}"))

    with tab_weights:
        weights_df = pd.DataFrame({
            "Ticker": mean_returns.index,
            "Max Sharpe Weight": max_sharpe_w,
            "Min Variance Weight": min_var_w,
        })
        st.dataframe(
            weights_df.style.format({"Max Sharpe Weight": "{:.2%}", "Min Variance Weight": "{:.2%}"})
        )
        fig_pie = px.pie(weights_df, names="Ticker", values="Max Sharpe Weight", title="Max Sharpe Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)
```

- [ ] **Step 3: Add a launch config for the preview tool**

Create `.claude/launch.json`:

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "mpt-finance",
      "runtimeExecutable": "venv/Scripts/streamlit.exe",
      "runtimeArgs": ["run", "app.py"],
      "port": 8501
    }
  ]
}
```

- [ ] **Step 4: Start the app and verify it loads**

Use the preview tool to start the `mpt-finance` server, then load the page and confirm the sidebar and title render with no console errors.

- [ ] **Step 5: Manually verify the golden path**

In the running app: enter `PTT.BK, KBANK.BK, AOT.BK, CPALL.BK`, click "Build Portfolio", and confirm all four tabs render without error (Efficient Frontier chart, Correlation heatmap, Backtest chart, Weights table + pie chart).

- [ ] **Step 6: Manually verify edge cases**

- Enter a single ticker → confirm the "at least 2 tickers" error shows.
- Enter one invalid ticker alongside two valid ones (e.g. `AAPL, MSFT, NOTAREALTICKER`) → confirm the dropped-ticker warning shows and the app still completes.

- [ ] **Step 7: Commit**

```bash
git add app.py .streamlit/config.toml .claude/launch.json
git commit -m "feat: add Streamlit UI for portfolio optimizer"
```

---

## Task 8: README and Deployment Prep

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# MPT Finance — Portfolio Optimizer

A Streamlit app that helps build the best-performing investment portfolio for a given
level of risk, based on Modern Portfolio Theory (MPT). Enter a list of tickers (Thai SET
or global), and the app finds the asset mix that maximizes risk-adjusted return.

**Live demo:** _add your Streamlit Community Cloud URL here after deploying_

## Why this matters for investors

When advising a client on how to split money across several stocks, the goal isn't just
picking "good" stocks — it's finding the *combination* that gives the best return for the
risk the client is willing to take. Harry Markowitz's Modern Portfolio Theory (1952,
Nobel Prize 1990) formalizes this: for any set of assets, there's a curve of
"efficient" portfolios — no other combination gives higher return at the same risk. This
app plots that curve and highlights two useful points on it: the portfolio with the best
risk-adjusted return (Max Sharpe) and the portfolio with the lowest possible risk (Min
Variance).

## Features

- **Efficient Frontier & Monte Carlo** — visualizes thousands of possible portfolios and
  traces the optimal frontier, highlighting the Max Sharpe and Min Variance portfolios.
- **Correlation Heatmap** — shows how selected assets move together, to explain
  diversification benefit.
- **Backtest vs Benchmark** — simulates how the optimized portfolio would have performed
  historically against a benchmark index (e.g. SET Index).
- **Portfolio Weights** — the exact allocation percentages for each recommended portfolio.

## Tech Stack

Python, Streamlit, yfinance, NumPy, pandas, SciPy (`scipy.optimize`), Plotly, pytest.

## Project Structure

```
MPT-Finance/
  app.py                  # Streamlit UI
  mpt_finance/
    data_loader.py        # price fetching & validation
    optimizer.py           # efficient frontier, max Sharpe, min variance
    simulate.py             # Monte Carlo portfolio simulation
    backtest.py              # historical backtest vs benchmark
    metrics.py                # Sharpe, volatility, drawdown, VaR
  tests/                       # pytest unit tests for every module above
```

## Running Locally

```bash
python -m venv venv
source venv/Scripts/activate  # on Windows Git Bash
pip install -r requirements.txt
streamlit run app.py
```

## Running Tests

```bash
pytest -v
```

## Future Improvements

- Multi-period rebalancing instead of a single static allocation
- Transaction costs and taxes in the backtest
- Support for short positions (currently long-only)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README"
```

---

## Task 9: Final End-to-End Verification and Push

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests across `test_metrics.py`, `test_optimizer.py`, `test_simulate.py`, `test_data_loader.py`, `test_backtest.py` pass.

- [ ] **Step 2: Re-verify the app end-to-end in the browser**

Repeat Task 7 Step 5 and Step 6 checks once more against the final committed code, to confirm nothing regressed after README/config changes.

- [ ] **Step 3: Confirm with the user before creating a GitHub remote**

Ask the user for their GitHub username/repo name preference, then run (only after confirmation):

```bash
gh repo create <repo-name> --public --source=. --remote=origin
git push -u origin master
```

- [ ] **Step 4: Deploy to Streamlit Community Cloud**

Direct the user to share.streamlit.io, connect the new GitHub repo, and set `app.py` as
the entrypoint. Once deployed, update the `README.md` live-demo link with the real URL,
commit, and push.
