# MPT Finance — Design Spec

## Purpose

A portfolio-optimization web app built on Modern Portfolio Theory (MPT), designed as a public GitHub/Streamlit portfolio project for Wathanyu Sodsri's job applications (SME Relationship Manager / Wealth roles). It should demonstrate both finance domain knowledge (risk-return optimization) and software engineering discipline (modular code, tests), matching the bar set by his existing self-directed projects (Credit Risk Prediction, Credit Card Fraud Detection).

## Scope

- Users input multiple tickers (Thai SET tickers like `PTT.BK`, `KBANK.BK`, or global tickers like `AAPL`), a date range, a risk-free rate, and a benchmark index.
- The app computes and visualizes:
  1. Efficient Frontier with Max Sharpe and Min Variance portfolios highlighted
  2. Monte Carlo simulation cloud (5,000–10,000 randomly weighted portfolios) plotted behind the frontier
  3. Correlation heatmap across selected assets with a short diversification insight
  4. Historical backtest of the Max Sharpe portfolio vs. a benchmark index, with return/risk metrics
- Deployed publicly on Streamlit Community Cloud; source on GitHub.

Out of scope: multi-period rebalancing, transaction costs, taxes, short-selling constraints, real-money execution. These are noted as future improvements in the README.

## Tech Stack

Python 3.11, `yfinance`, `numpy`, `pandas`, `scipy.optimize`, `plotly`, `streamlit`, `pytest`.

## Repo Structure

```
MPT-Finance/
  app.py                      # Streamlit entrypoint (UI only, no business logic)
  mpt_finance/
    __init__.py
    data_loader.py            # fetch prices, compute returns, validate tickers/date range
    optimizer.py               # expected returns, covariance, efficient frontier, max Sharpe, min variance
    simulate.py                 # Monte Carlo random-weight portfolio generation
    backtest.py                 # historical backtest of chosen weights vs. benchmark
    metrics.py                   # Sharpe ratio, annualized volatility, max drawdown, VaR
  tests/
    test_optimizer.py
    test_metrics.py
    test_data_loader.py
  requirements.txt
  .streamlit/config.toml
  .gitignore
  README.md
```

`app.py` only wires UI to the `mpt_finance` package — no calculation logic lives in it, so the package can be tested independently of Streamlit.

## Data Flow

1. **Input** (sidebar): tickers, date range, risk-free rate, benchmark ticker (defaults to `^SET.BK` or `^GSPC` depending on market mix).
2. **`data_loader`**: fetches adjusted close prices via yfinance, computes daily returns, drops tickers with no data, aligns all series to their common overlapping date range.
3. **`optimizer`**: computes annualized expected returns and covariance matrix; runs `scipy.optimize.minimize` across a range of target returns to trace the efficient frontier; identifies the Max Sharpe and Min Variance portfolios.
4. **`simulate`**: randomly samples portfolio weights (5,000–10,000 draws), computes return/volatility/Sharpe for each, for the Monte Carlo scatter cloud.
5. **Correlation heatmap**: computed directly from the returns matrix.
6. **`backtest`**: applies the Max Sharpe weights to historical returns to build a cumulative portfolio value series, compares against the benchmark, and computes annualized return, volatility, Sharpe, and max drawdown via `metrics`.
7. **UI**: four tabs — Efficient Frontier & Monte Carlo, Correlation, Backtest vs Benchmark, Portfolio Weights (table + pie chart).

## Error Handling

All errors surface as `st.warning`/`st.error` messages, never raw tracebacks:

- Invalid/delisted ticker → dropped from the portfolio, user notified which ticker(s) were excluded.
- Fewer than 2 valid tickers remain → optimization blocked, explanatory message shown.
- Insufficient overlapping date range (e.g., a recently listed ticker) → data aligned to the common available range, user notified of the adjustment.
- Singular covariance matrix (e.g., duplicate tickers) → caught and reported instead of crashing.

## Testing

- `test_optimizer.py`: validates the optimizer against a synthetic 2-asset case with a known analytical MPT solution.
- `test_metrics.py`: validates Sharpe ratio, volatility, and max drawdown against hand-computed values on fixed sample data.
- `test_data_loader.py`: mocks yfinance calls to test edge cases (invalid ticker, insufficient data) without requiring network access.

## README / Portfolio Narrative

Written for two audiences — non-technical finance recruiters and technical reviewers:

1. Problem framing in RM/wealth-advisory language: helping a client allocate across assets for the best risk-adjusted return.
2. Screenshot/GIF of the running app plus a live Streamlit Cloud demo link.
3. Plain-language "How it works" section explaining the efficient frontier and diversification.
4. Tech stack, local run instructions, repo structure.
5. Future improvements section (transaction costs, multi-period rebalancing, short positions).

## Deployment

Push to GitHub, connect the repo to Streamlit Community Cloud (free tier). No secrets required — yfinance uses public data. The resulting public URL goes into the resume/LinkedIn alongside the GitHub link.

## Project Location

`C:\Users\acer\Desktop\Claude-Project\MPT-Finance` — new, separate git repository (sibling to the `Resume` folder, which holds only resume documents).
