from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_prices(price_frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(price_frame.columns, pd.MultiIndex):
        if "Close" in price_frame.columns.get_level_values(-1):
            price_frame = price_frame.xs("Close", axis=1, level=-1)
        elif "Adj Close" in price_frame.columns.get_level_values(-1):
            price_frame = price_frame.xs("Adj Close", axis=1, level=-1)
        else:
            price_frame = price_frame.droplevel(0, axis=1)
    price_frame = price_frame.sort_index()
    price_frame = price_frame.ffill().dropna(axis=0, how="any")
    price_frame = price_frame.loc[:, price_frame.nunique() > 1]
    return price_frame


def fetch_price_data(tickers: list[str], period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No data returned by yfinance.")

    if len(tickers) == 1:
        if "Close" in raw.columns:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        elif "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            prices = raw[[raw.columns[0]]].rename(columns={raw.columns[0]: tickers[0]})
        return _clean_prices(prices)

    prices = raw
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(-1):
            prices = raw.xs("Close", axis=1, level=-1)
        elif "Adj Close" in raw.columns.get_level_values(-1):
            prices = raw.xs("Adj Close", axis=1, level=-1)
    prices = prices.loc[:, [col for col in tickers if col in prices.columns]]
    prices = _clean_prices(prices)
    if prices.empty:
        raise ValueError("Cleaned price data is empty.")
    return prices


def generate_demo_prices(tickers: list[str], periods: int = 756, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    base_trend = np.cumsum(rng.normal(0.0003, 0.01, size=periods))
    mean_reverting_spread = np.zeros(periods)
    for i in range(1, periods):
        mean_reverting_spread[i] = 0.92 * mean_reverting_spread[i - 1] + rng.normal(0, 0.02)

    log_prices = np.zeros((periods, n_assets))
    log_prices[:, 0] = 4.5 + base_trend + 0.5 * mean_reverting_spread
    if n_assets > 1:
        log_prices[:, 1] = 4.5 + base_trend - 0.5 * mean_reverting_spread

    for j in range(2, n_assets):
        idio = np.cumsum(rng.normal(0.00025 + 0.00005 * j, 0.012 + 0.002 * j, size=periods))
        factor_loading = 0.45 + 0.05 * rng.random()
        log_prices[:, j] = 4.4 + factor_loading * base_trend + idio

    prices = np.exp(log_prices)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def price_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="any")
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return returns


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.replace(0, np.nan)).dropna(how="any")

