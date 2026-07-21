from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pairs_trading_lab.data import fetch_price_data, generate_demo_prices
from pairs_trading_lab.research import rank_pairs, walk_forward_pairs_backtest
from pairs_trading_lab.backtest import backtest_pair
from pairs_trading_lab.metrics import performance_summary, format_summary_table


st.set_page_config(
    page_title="Pairs Trading Research Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.13), transparent 24%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, #eef4f8 100%);
        }
        .hero {
            padding: 1.4rem 1.4rem 1rem 1.4rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.15rem;
        }
        .hero p {
            margin: 0;
            color: #475569;
            font-size: 1rem;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _default_tickers() -> str:
    return "XLE, XOP, XLF, KBE, JPM, BAC, WFC, GS"


@st.cache_data(show_spinner=False, ttl=3600)
def load_price_frame(tickers: tuple[str, ...], period: str, interval: str) -> tuple[pd.DataFrame, bool]:
    try:
        prices = fetch_price_data(list(tickers), period=period, interval=interval)
        if prices.shape[1] < 2 or prices.empty:
            raise ValueError("Not enough price data.")
        return prices, True
    except Exception:
        return generate_demo_prices(list(tickers), periods=756), False


def _equity_figure(result) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, mode="lines", name="Strategy"))
    if result.benchmark_equity is not None:
        fig.add_trace(go.Scatter(x=result.benchmark_equity.index, y=result.benchmark_equity.values, mode="lines", name="Static hedge"))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _drawdown_figure(result) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown.values, mode="lines", fill="tozeroy", name="Drawdown"))
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _spread_figure(result) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.spread.index, y=result.spread.values, mode="lines", name="Spread"))
    fig.add_trace(go.Scatter(x=result.zscore.index, y=result.zscore.values, mode="lines", name="Z-score", yaxis="y2"))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#94a3b8")
    fig.add_hline(y=2.0, line_width=1, line_dash="dash", line_color="#f59e0b")
    fig.add_hline(y=-2.0, line_width=1, line_dash="dash", line_color="#f59e0b")
    fig.update_layout(
        title="Spread and Z-Score",
        xaxis_title="Date",
        yaxis=dict(title="Spread"),
        yaxis2=dict(title="Z-Score", overlaying="y", side="right"),
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


st.markdown(
    """
    <div class="hero">
        <h1>Pairs Trading Research Lab</h1>
        <p>Discover mean-reverting pairs, run walk-forward backtests, and stress-test a systematic trading idea with realistic costs and out-of-sample validation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Research Controls")
    tickers_text = st.text_input("Universe", value=_default_tickers())
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    mode = st.radio("Mode", ["Walk-forward research", "Single pair backtest"], index=0)
    train_window = st.slider("Training window (days)", 60, 504, 252, 21)
    test_window = st.slider("Test window (days)", 21, 252, 63, 21)
    lookback = st.slider("Signal lookback", 20, 120, 60, 5)
    entry_z = st.slider("Entry Z-score", 1.0, 3.5, 2.0, 0.1)
    exit_z = st.slider("Exit Z-score", 0.0, 2.0, 0.5, 0.1)
    stop_z = st.slider("Stop Z-score", 2.0, 5.0, 3.5, 0.1)
    max_hold = st.slider("Max holding days", 5, 60, 20, 1)
    fee_bps = st.slider("Fee (bps per turnover)", 0.0, 20.0, 1.0, 0.5)
    slippage_bps = st.slider("Slippage (bps per turnover)", 0.0, 20.0, 1.0, 0.5)
    min_corr = st.slider("Min pair correlation", 0.0, 0.95, 0.55, 0.05)
    user_pair = st.text_input("Optional pair override", value="").strip().upper()

ticker_list = [token.strip().upper() for token in tickers_text.replace(",", " ").split() if token.strip()]
if len(ticker_list) < 2:
    st.warning("Please enter at least two tickers.")
    st.stop()

with st.spinner("Loading prices and running research..."):
    prices, live_data = load_price_frame(tuple(ticker_list), period, interval)
    prices = prices.loc[:, [c for c in ticker_list if c in prices.columns]]
    rank_table = rank_pairs(prices.tail(train_window), min_corr=min_corr)

    if rank_table.empty:
        st.error("No viable pairs were found in the current universe. Try a different set of tickers.")
        st.stop()

    best_pair = tuple(rank_table.iloc[0]["pair"])

    if mode == "Single pair backtest":
        if user_pair:
            pair_tokens = [token.strip().upper() for token in user_pair.replace("/", ",").replace("-", ",").split(",") if token.strip()]
            if len(pair_tokens) != 2 or any(token not in prices.columns for token in pair_tokens):
                st.warning("The override pair was not valid for this universe. Falling back to the top-ranked pair.")
            else:
                best_pair = (pair_tokens[0], pair_tokens[1])
        result = backtest_pair(
            prices[[best_pair[0], best_pair[1]]],
            best_pair[0],
            best_pair[1],
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold=max_hold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            train_prices=prices.tail(train_window),
        )
        summary = performance_summary(result.returns, result.positions, result.turnover, result.trades)
        fold_table = None
    else:
        wf = walk_forward_pairs_backtest(
            prices,
            train_window=train_window,
            test_window=test_window,
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold=max_hold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_corr=min_corr,
        )
        result = wf.result
        summary = performance_summary(result.returns, result.positions, result.turnover, result.trades)
        fold_table = wf.fold_summary
        best_pair = tuple(wf.best_pair)

st.markdown(
    f"""
    <div class="section-card">
        <strong>Data source:</strong> {"Live market data" if live_data else "Synthetic demo data"} |
        <strong>Selected pair:</strong> {best_pair[0]} / {best_pair[1]} |
        <strong>Hedge ratio:</strong> {result.hedge_ratio:.3f} |
        <strong>Half-life:</strong> {result.half_life:.1f} days
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(6)
metric_data = [
    ("CAGR", f"{summary['cagr']:.2%}"),
    ("Sharpe", f"{summary['sharpe']:.2f}"),
    ("Max Drawdown", f"{summary['max_drawdown']:.2%}"),
    ("Win Rate", f"{summary['win_rate']:.2%}"),
    ("Trades", f"{summary['trades']}"),
    ("Exposure", f"{summary['exposure']:.2%}"),
]
for col, (label, value) in zip(metric_cols, metric_data):
    with col:
        st.metric(label, value)

top_left, top_right = st.columns([1.15, 0.85])
with top_left:
    st.plotly_chart(_equity_figure(result), use_container_width=True)
with top_right:
    st.plotly_chart(_drawdown_figure(result), use_container_width=True)

mid_left, mid_right = st.columns([1.05, 0.95])
with mid_left:
    st.plotly_chart(_spread_figure(result), use_container_width=True)
with mid_right:
    st.subheader("Pair Ranking")
    st.dataframe(
        rank_table[["pair", "corr", "beta", "half_life", "score"]].head(10).style.format(
            {"corr": "{:.2f}", "beta": "{:.3f}", "half_life": "{:.1f}", "score": "{:.3f}"}
        ),
        use_container_width=True,
        height=360,
    )

st.subheader("Strategy Summary")
summary_table = format_summary_table(result)
st.dataframe(summary_table.style.format("{:.4f}"), use_container_width=True)

if fold_table is not None:
    st.subheader("Walk-Forward Folds")
    st.dataframe(
        fold_table.style.format(
            {
                "train_start": lambda x: x,
                "train_end": lambda x: x,
                "test_start": lambda x: x,
                "test_end": lambda x: x,
                "corr": "{:.2f}",
                "beta": "{:.3f}",
                "half_life": "{:.1f}",
                "test_cagr": "{:.2%}",
                "test_sharpe": "{:.2f}",
                "test_max_drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

st.subheader("Trade Log")
st.dataframe(result.trades.tail(30), use_container_width=True, height=280)

st.download_button(
    "Download strategy summary as CSV",
    data=summary_table.to_csv(index=True).encode("utf-8"),
    file_name="pairs_trading_summary.csv",
    mime="text/csv",
)
