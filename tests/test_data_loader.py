import pandas as pd
import pytest

from mpt_finance import data_loader


def _fake_price_frame(values):
    dates = pd.date_range("2024-01-01", periods=len(values))
    return pd.DataFrame({"Adj Close": values}, index=dates)


def _fake_multiindex_price_frame(values, ticker):
    dates = pd.date_range("2024-01-01", periods=len(values))
    frame = pd.DataFrame({"Adj Close": values}, index=dates)
    frame.columns = pd.MultiIndex.from_product([frame.columns, [ticker]])
    return frame


def test_fetch_price_data_returns_prices_for_valid_tickers(monkeypatch):
    frames = {
        "AAPL": _fake_price_frame([100, 101, 102]),
        "MSFT": _fake_price_frame([200, 202, 201]),
    }

    def fake_download(ticker, start=None, end=None, progress=False, auto_adjust=False):
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

    def fake_download(ticker, start=None, end=None, progress=False, auto_adjust=False):
        return frames[ticker]

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    prices, dropped = data_loader.fetch_price_data(["AAPL", "MSFT", "FAKE"], "2024-01-01", "2024-01-03")

    assert dropped == ["FAKE"]
    assert list(prices.columns) == ["AAPL", "MSFT"]


def test_fetch_price_data_drops_ticker_with_all_nan_adj_close(monkeypatch):
    frames = {
        "AAPL": _fake_price_frame([100, 101, 102]),
        "MSFT": _fake_price_frame([200, 202, 201]),
        "DELISTED": pd.DataFrame(
            {"Adj Close": [float("nan"), float("nan"), float("nan")]},
            index=pd.date_range("2024-01-01", periods=3),
        ),
    }

    def fake_download(ticker, start=None, end=None, progress=False, auto_adjust=False):
        return frames[ticker]

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    prices, dropped = data_loader.fetch_price_data(
        ["AAPL", "MSFT", "DELISTED"], "2024-01-01", "2024-01-03"
    )

    assert dropped == ["DELISTED"]
    assert list(prices.columns) == ["AAPL", "MSFT"]


def test_fetch_price_data_raises_when_fewer_than_two_valid_tickers(monkeypatch):
    frames = {
        "AAPL": _fake_price_frame([100, 101, 102]),
        "FAKE1": pd.DataFrame(),
        "FAKE2": pd.DataFrame(),
    }

    def fake_download(ticker, start=None, end=None, progress=False, auto_adjust=False):
        return frames[ticker]

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    with pytest.raises(data_loader.InsufficientDataError):
        data_loader.fetch_price_data(["AAPL", "FAKE1", "FAKE2"], "2024-01-01", "2024-01-03")


def test_fetch_price_data_handles_multiindex_columns_from_real_yfinance(monkeypatch):
    frames = {
        "AAPL": _fake_multiindex_price_frame([100, 101, 102], "AAPL"),
        "MSFT": _fake_multiindex_price_frame([200, 202, 201], "MSFT"),
    }

    def fake_download(ticker, start=None, end=None, progress=False, auto_adjust=False):
        return frames[ticker]

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    prices, dropped = data_loader.fetch_price_data(["AAPL", "MSFT"], "2024-01-01", "2024-01-03")

    assert dropped == []
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert prices.shape == (3, 2)
    assert prices["AAPL"].tolist() == [100, 101, 102]
    assert prices["MSFT"].tolist() == [200, 202, 201]


def test_fetch_benchmark_prices_returns_series(monkeypatch):
    monkeypatch.setattr(
        data_loader.yf, "download",
        lambda ticker, start=None, end=None, progress=False, auto_adjust=False: _fake_price_frame([1000, 1010, 1005]),
    )

    series = data_loader.fetch_benchmark_prices("^GSPC", "2024-01-01", "2024-01-03")

    assert series.tolist() == [1000, 1010, 1005]


def test_fetch_benchmark_prices_handles_multiindex_columns_from_real_yfinance(monkeypatch):
    monkeypatch.setattr(
        data_loader.yf, "download",
        lambda ticker, start=None, end=None, progress=False, auto_adjust=False: _fake_multiindex_price_frame(
            [1000, 1010, 1005], "^GSPC"
        ),
    )

    series = data_loader.fetch_benchmark_prices("^GSPC", "2024-01-01", "2024-01-03")

    assert series.tolist() == [1000, 1010, 1005]


def test_fetch_benchmark_prices_raises_when_empty(monkeypatch):
    monkeypatch.setattr(
        data_loader.yf, "download",
        lambda ticker, start=None, end=None, progress=False, auto_adjust=False: pd.DataFrame(),
    )

    with pytest.raises(data_loader.InsufficientDataError):
        data_loader.fetch_benchmark_prices("BADTICKER", "2024-01-01", "2024-01-03")


def test_compute_daily_returns_matches_pct_change():
    prices = pd.DataFrame({"AAPL": [100, 110, 99], "MSFT": [200, 190, 209]})
    returns = data_loader.compute_daily_returns(prices)
    assert returns["AAPL"].tolist() == pytest.approx([0.10, -0.10])
    assert returns["MSFT"].tolist() == pytest.approx([-0.05, 0.10])
