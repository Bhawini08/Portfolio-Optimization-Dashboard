from __future__ import annotations

import math

import pandas as pd


def _max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    return float(drawdown.min())


def performance_summary(
    returns: pd.Series,
    positions: pd.Series,
    turnover: pd.Series,
    trades: pd.DataFrame,
) -> dict[str, float]:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    periods_per_year = 252
    total_days = max(len(returns), 1)
    cagr = float(equity.iloc[-1] ** (periods_per_year / total_days) - 1.0) if len(equity) else 0.0
    annual_vol = float(returns.std(ddof=0) * math.sqrt(periods_per_year))
    sharpe = float((returns.mean() * periods_per_year) / annual_vol) if annual_vol > 0 else 0.0
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * math.sqrt(periods_per_year)) if len(downside) else 0.0
    sortino = float((returns.mean() * periods_per_year) / downside_vol) if downside_vol > 0 else 0.0
    drawdown = _max_drawdown(equity)
    calmar = float(cagr / abs(drawdown)) if drawdown < 0 else 0.0
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    exposure = float((positions != 0).mean()) if len(positions) else 0.0
    avg_turnover = float(turnover.mean()) if len(turnover) else 0.0
    profit_factor = float(returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if returns[returns < 0].sum() != 0 else 0.0

    return {
        "cagr": cagr,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "exposure": exposure,
        "avg_turnover": avg_turnover,
        "profit_factor": profit_factor,
        "trades": float(len(trades)),
    }


def format_summary_table(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "CAGR",
                "Value": result.metrics["cagr"],
            },
            {
                "Metric": "Annual Vol",
                "Value": result.metrics["annual_vol"],
            },
            {
                "Metric": "Sharpe",
                "Value": result.metrics["sharpe"],
            },
            {
                "Metric": "Sortino",
                "Value": result.metrics["sortino"],
            },
            {
                "Metric": "Max Drawdown",
                "Value": result.metrics["max_drawdown"],
            },
            {
                "Metric": "Calmar",
                "Value": result.metrics["calmar"],
            },
            {
                "Metric": "Win Rate",
                "Value": result.metrics["win_rate"],
            },
            {
                "Metric": "Exposure",
                "Value": result.metrics["exposure"],
            },
            {
                "Metric": "Avg Turnover",
                "Value": result.metrics["avg_turnover"],
            },
            {
                "Metric": "Profit Factor",
                "Value": result.metrics["profit_factor"],
            },
        ]
    ).set_index("Metric")

