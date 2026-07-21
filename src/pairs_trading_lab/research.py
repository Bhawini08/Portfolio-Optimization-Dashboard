from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest_pair, estimate_hedge_ratio, spread_half_life
from .data import log_prices


def pair_score(log_returns: pd.DataFrame, logs: pd.DataFrame, asset_a: str, asset_b: str) -> dict[str, float]:
    corr = float(log_returns[asset_a].corr(log_returns[asset_b]))
    beta, intercept = estimate_hedge_ratio(logs[asset_a], logs[asset_b])
    spread = logs[asset_a] - (intercept + beta * logs[asset_b])
    half_life = spread_half_life(spread)
    spread_vol = float(spread.std(ddof=0))
    stability = 1.0 / (1.0 + (half_life if np.isfinite(half_life) else 999.0))
    score = abs(corr) * stability / (1.0 + spread_vol)
    return {
        "corr": corr,
        "beta": beta,
        "intercept": intercept,
        "half_life": half_life,
        "spread_vol": spread_vol,
        "score": score,
    }


def rank_pairs(prices: pd.DataFrame, min_corr: float = 0.55) -> pd.DataFrame:
    prices = prices.dropna()
    if prices.shape[1] < 2 or prices.shape[0] < 30:
        return pd.DataFrame(columns=["pair", "corr", "beta", "intercept", "half_life", "spread_vol", "score"])

    logs = log_prices(prices)
    log_returns = logs.diff().dropna()
    rows = []
    for a, b in itertools.combinations(prices.columns, 2):
        stats = pair_score(log_returns, logs, a, b)
        if abs(stats["corr"]) < min_corr:
            continue
        rows.append(
            {
                "pair": (a, b),
                **stats,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["pair", "corr", "beta", "intercept", "half_life", "spread_vol", "score"])
    return pd.DataFrame(rows).sort_values(["score", "corr"], ascending=False).reset_index(drop=True)


@dataclass
class WalkForwardResult:
    result: BacktestResult
    fold_summary: pd.DataFrame
    best_pair: tuple[str, str]


def walk_forward_pairs_backtest(
    prices: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 63,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_hold: int = 20,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    min_corr: float = 0.55,
) -> WalkForwardResult:
    prices = prices.dropna()
    if prices.shape[0] < train_window + test_window + 10:
        raise ValueError("Not enough history for walk-forward testing.")

    all_strategy_returns: list[pd.Series] = []
    all_benchmark_returns: list[pd.Series] = []
    all_positions: list[pd.Series] = []
    all_turnover: list[pd.Series] = []
    all_spreads: list[pd.Series] = []
    all_zscores: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    best_pair: tuple[str, str] | None = None
    last_result: BacktestResult | None = None

    cursor = train_window
    while cursor + test_window <= len(prices):
        train_prices = prices.iloc[cursor - train_window:cursor]
        ranking = rank_pairs(train_prices, min_corr=min_corr)
        if ranking.empty:
            cursor += test_window
            continue
        pair = tuple(ranking.iloc[0]["pair"])
        if best_pair is None:
            best_pair = pair

        window_prices = pd.concat([train_prices.tail(lookback), prices.iloc[cursor:cursor + test_window]])
        result = backtest_pair(
            window_prices[[pair[0], pair[1]]],
            pair[0],
            pair[1],
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold=max_hold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            train_prices=train_prices,
        )
        last_result = result
        test_index = prices.index[cursor:cursor + test_window]
        test_returns = result.returns.loc[test_index]
        test_positions = result.positions.loc[test_index]
        test_turnover = result.turnover.loc[test_index]
        test_spread = result.spread.loc[test_index]
        test_zscore = result.zscore.loc[test_index]
        test_trades = result.trades[
            (result.trades["entry_date"] >= test_index.min()) & (result.trades["entry_date"] <= test_index.max())
        ] if not result.trades.empty else result.trades

        from .metrics import performance_summary

        fold_metrics = performance_summary(test_returns, test_positions, test_turnover, test_trades)
        fold_rows.append(
            {
                "train_start": train_prices.index[0],
                "train_end": train_prices.index[-1],
                "test_start": test_index[0],
                "test_end": test_index[-1],
                "pair": f"{pair[0]} / {pair[1]}",
                "corr": ranking.iloc[0]["corr"],
                "beta": ranking.iloc[0]["beta"],
                "half_life": ranking.iloc[0]["half_life"],
                "test_cagr": fold_metrics["cagr"],
                "test_sharpe": fold_metrics["sharpe"],
                "test_max_drawdown": fold_metrics["max_drawdown"],
                "test_trades": fold_metrics["trades"],
            }
        )

        all_strategy_returns.append(test_returns)
        all_benchmark_returns.append(result.benchmark_returns.loc[test_index])
        all_positions.append(test_positions)
        all_turnover.append(test_turnover)
        all_spreads.append(test_spread)
        all_zscores.append(test_zscore)
        all_trades.append(test_trades)
        cursor += test_window

    if not all_strategy_returns or last_result is None or best_pair is None:
        raise ValueError("Walk-forward research did not produce any folds.")

    strategy_returns = pd.concat(all_strategy_returns).sort_index()
    benchmark_returns = pd.concat(all_benchmark_returns).sort_index()
    positions = pd.concat(all_positions).sort_index()
    turnover = pd.concat(all_turnover).sort_index()
    spreads = pd.concat(all_spreads).sort_index()
    zscores = pd.concat(all_zscores).sort_index()
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    stitched = BacktestResult(
        pair=best_pair,
        hedge_ratio=last_result.hedge_ratio,
        intercept=last_result.intercept,
        half_life=last_result.half_life,
        spread=spreads,
        zscore=zscores,
        returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        equity_curve=(1.0 + strategy_returns).cumprod(),
        benchmark_equity=(1.0 + benchmark_returns).cumprod(),
        drawdown=(1.0 + strategy_returns).cumprod() / (1.0 + strategy_returns).cumprod().cummax() - 1.0,
        positions=positions,
        turnover=turnover,
        trades=trades,
        metrics={},
    )
    from .metrics import performance_summary

    stitched.metrics = performance_summary(stitched.returns, stitched.positions, stitched.turnover, stitched.trades)
    fold_summary = pd.DataFrame(fold_rows)
    return WalkForwardResult(result=stitched, fold_summary=fold_summary, best_pair=best_pair)
