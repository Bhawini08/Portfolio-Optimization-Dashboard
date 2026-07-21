# Pairs Trading Research Lab

A GitHub-ready quant research project for exploring a **sector-based pairs trading strategy** with:

- automated pair discovery
- hedge ratio estimation
- rolling z-score entry and exit rules
- realistic transaction costs and slippage
- walk-forward out-of-sample validation
- equity curve, drawdown, and trade log analysis
- a synthetic demo mode when live data is unavailable

The project is designed to look and feel like a real research workflow, not just a toy backtest.

## Why this is strong for quant roles

This project maps directly to how systematic trading ideas are evaluated in practice:

- you define a hypothesis about mean reversion
- you select candidate pairs from a universe
- you estimate a hedge ratio on training data
- you validate out-of-sample with walk-forward testing
- you include costs, slippage, and risk metrics
- you inspect trade-level behavior, not just a single headline Sharpe

That makes it especially relevant for:

- quant trading roles
- systematic research roles
- pair trading / stat arb teams
- hedge fund internships

## Features

- Live price download with `yfinance`
- Synthetic fallback dataset with a built-in mean-reverting pair
- Pair ranking across the candidate universe
- Static-hedge and strategy equity curve comparison
- Walk-forward testing across multiple folds
- Trade log and summary statistics
- CSV export for results

## Tech Stack

- Python
- NumPy
- Pandas
- SciPy
- Streamlit
- Plotly
- yfinance

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Suggested Universe

Good starter universes:

```text
XLE, XOP, XLF, KBE, JPM, BAC, WFC, GS
```

You can also try:

- `GOOG, GOOGL`
- `KO, PEP`
- `XLI, IYT, CAT, DE`
- `JPM, BAC, WFC, C`

## What the App Does

1. Downloads historical prices or uses a synthetic fallback.
2. Ranks candidate pairs based on correlation, hedge ratio, and mean reversion strength.
3. Either backtests the top pair directly or runs walk-forward research.
4. Applies realistic costs and slippage when positions change.
5. Visualizes equity, drawdown, spread behavior, and trade history.

## Project Structure

```text
.
├── app.py
├── requirements.txt
└── src/
    └── pairs_trading_lab/
        ├── __init__.py
        ├── data.py
        ├── backtest.py
        ├── research.py
        └── metrics.py
```

## Resume Bullet

You can use a sharper bullet like:

> Built a Streamlit pairs trading research lab with automated pair selection, rolling hedge ratio estimation, walk-forward backtesting, and realistic transaction cost modeling to evaluate stat-arb signals across sector universes.

