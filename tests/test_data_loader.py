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
