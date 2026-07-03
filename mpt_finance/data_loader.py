"""Fetch and validate price data for portfolio construction."""
import pandas as pd
import yfinance as yf


class InsufficientDataError(Exception):
    """Raised when there isn't enough valid price data to build a portfolio."""


def _as_series(column_data):
    """Normalize a column extraction to a plain 1-D Series.

    Newer yfinance versions return MultiIndex columns from yf.download even for
    a single ticker, so data["Adj Close"] can come back as a single-column
    DataFrame instead of a Series. This flattens it back to a Series.
    """
    if isinstance(column_data, pd.DataFrame):
        return column_data.iloc[:, 0]
    return column_data


def fetch_price_data(tickers, start, end):
    valid_prices = {}
    dropped_tickers = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty or "Adj Close" not in data.columns or data["Adj Close"].dropna().empty:
            dropped_tickers.append(ticker)
            continue
        valid_prices[ticker] = _as_series(data["Adj Close"])

    if len(valid_prices) < 2:
        raise InsufficientDataError(
            f"Only {len(valid_prices)} valid ticker(s) after dropping {dropped_tickers}; need at least 2."
        )

    prices = pd.DataFrame(valid_prices).dropna(how="any")
    return prices, dropped_tickers


def fetch_benchmark_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty or "Adj Close" not in data.columns or data["Adj Close"].dropna().empty:
        raise InsufficientDataError(f"No data available for benchmark ticker '{ticker}'.")
    return _as_series(data["Adj Close"])


def compute_daily_returns(prices):
    return prices.pct_change().dropna(how="any")
