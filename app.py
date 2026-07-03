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
