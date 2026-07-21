from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import log_prices
from .metrics import performance_summary


@dataclass
class BacktestResult:
    pair: tuple[str, str]
    hedge_ratio: float
    intercept: float
    half_life: float
    spread: pd.Series
    zscore: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series
    equity_curve: pd.Series
    benchmark_equity: pd.Series
    drawdown: pd.Series
    positions: pd.Series
    turnover: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


def estimate_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float]:
    x = np.column_stack([log_b.values, np.ones(len(log_b))])
    beta, intercept = np.linalg.lstsq(x, log_a.values, rcond=None)[0]
    return float(beta), float(intercept)


def spread_half_life(spread: pd.Series) -> float:
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([lagged, delta], axis=1).dropna()
    if aligned.shape[0] < 10:
        return float("nan")
    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return float("inf")
    return float(-np.log(2.0) / slope)


def _base_weights(beta: float) -> np.ndarray:
    weights = np.array([1.0, -beta], dtype=float)
    gross = np.abs(weights).sum()
    if gross == 0:
        return np.array([0.5, -0.5], dtype=float)
    return weights / gross


def backtest_pair(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_hold: int = 20,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    train_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    pair_prices = prices[[asset_a, asset_b]].dropna()
    if len(pair_prices) < lookback + 5:
        raise ValueError("Not enough observations for the requested lookback.")

    fit_prices = train_prices[[asset_a, asset_b]].dropna() if train_prices is not None else pair_prices
    fit_logs = log_prices(fit_prices)
    beta, intercept = estimate_hedge_ratio(fit_logs[asset_a], fit_logs[asset_b])

    logs = log_prices(pair_prices)
    spread = logs[asset_a] - (intercept + beta * logs[asset_b])
    half_life = spread_half_life(spread)
    returns = pair_prices.pct_change().fillna(0.0)
    base = _base_weights(beta)
    total_cost_per_turnover = (fee_bps + slippage_bps) / 10000.0

    strategy_returns = pd.Series(0.0, index=pair_prices.index)
    benchmark_returns = pd.Series(0.0, index=pair_prices.index)
    positions = pd.Series(0, index=pair_prices.index, dtype=int)
    turnover = pd.Series(0.0, index=pair_prices.index)
    zscores = pd.Series(np.nan, index=pair_prices.index)

    current_position = 0
    current_weights = np.zeros(2)
    entry_date = None
    entry_idx = None
    open_trade_returns: list[float] = []
    trade_rows: list[dict[str, object]] = []

    for i in range(1, len(pair_prices)):
        date = pair_prices.index[i]
        hist = spread.iloc[max(0, i - lookback):i]
        if len(hist) < max(10, lookback // 3):
            continue

        mu = hist.mean()
        sigma = hist.std(ddof=0)
        z = 0.0 if sigma == 0 else float((spread.iloc[i - 1] - mu) / sigma)
        zscores.iloc[i] = z

        desired_position = current_position
        holding_days = 0 if entry_idx is None else (i - entry_idx)

        if current_position == 0:
            if z <= -entry_z:
                desired_position = 1
                entry_date = date
                entry_idx = i
                open_trade_returns = []
            elif z >= entry_z:
                desired_position = -1
                entry_date = date
                entry_idx = i
                open_trade_returns = []
        elif current_position == 1:
            if z >= -exit_z or z <= -stop_z or holding_days >= max_hold:
                desired_position = 0
        elif current_position == -1:
            if z <= exit_z or z >= stop_z or holding_days >= max_hold:
                desired_position = 0

        target_weights = desired_position * base
        day_turnover = float(np.abs(target_weights - current_weights).sum())
        trade_cost = day_turnover * total_cost_per_turnover
        current_weights = target_weights
        turnover.iloc[i] = day_turnover
        positions.iloc[i] = desired_position

        daily_return = float(current_weights @ returns.iloc[i].values) - trade_cost
        strategy_returns.iloc[i] = daily_return
        benchmark_returns.iloc[i] = float(base @ returns.iloc[i].values)

        if current_position == 0 and desired_position != 0:
            current_position = desired_position
        elif current_position != 0 and desired_position == 0:
            trade_rows.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long spread" if current_position == 1 else "short spread",
                    "holding_days": int(i - entry_idx if entry_idx is not None else 0),
                    "trade_return": float(np.prod(1 + np.array(open_trade_returns)) - 1.0) if open_trade_returns else 0.0,
                }
            )
            current_position = 0
            entry_date = None
            entry_idx = None
            open_trade_returns = []
        elif current_position != 0 and desired_position != current_position:
            trade_rows.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long spread" if current_position == 1 else "short spread",
                    "holding_days": int(i - entry_idx if entry_idx is not None else 0),
                    "trade_return": float(np.prod(1 + np.array(open_trade_returns)) - 1.0) if open_trade_returns else 0.0,
                }
            )
            current_position = desired_position
            entry_date = date if desired_position != 0 else None
            entry_idx = i if desired_position != 0 else None
            open_trade_returns = []

        if current_position != 0:
            open_trade_returns.append(daily_return)

    if current_position != 0 and entry_date is not None:
        trade_rows.append(
            {
                "entry_date": entry_date,
                "exit_date": pair_prices.index[-1],
                "direction": "long spread" if current_position == 1 else "short spread",
                "holding_days": int(len(open_trade_returns)),
                "trade_return": float(np.prod(1 + np.array(open_trade_returns)) - 1.0) if open_trade_returns else 0.0,
            }
        )

    equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    trades = pd.DataFrame(trade_rows)
    metrics = performance_summary(strategy_returns, positions, turnover, trades)

    return BacktestResult(
        pair=(asset_a, asset_b),
        hedge_ratio=beta,
        intercept=intercept,
        half_life=half_life,
        spread=spread,
        zscore=zscores,
        returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        equity_curve=equity,
        benchmark_equity=benchmark_equity,
        drawdown=drawdown,
        positions=positions,
        turnover=turnover,
        trades=trades,
        metrics=metrics,
    )
