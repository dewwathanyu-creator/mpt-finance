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
