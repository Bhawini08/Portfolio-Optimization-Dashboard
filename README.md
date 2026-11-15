# Pairs Trading Research Lab

This Streamlit app tests long/short spreads using pairs and hedge ratios selected on training data before each test window.

## Results from three walk-forward folds

Adjusted Yahoo Finance daily prices for XLE, XOP, XLF, KBE, JPM, BAC, WFC and GS produced **6.61% annualized return** after fees and slippage, a **1.18 Sharpe** and **-4.22% maximum drawdown**. One fold lost money, and there were only seven logged trades.

| Test window | Pair | Annualized return | Sharpe | Logged trades |
| --- | --- | ---: | ---: | ---: |
| Nov 18, 2025–Feb 19, 2026 | XLF / GS | 8.28% | 1.44 | 3 |
| Feb 20–May 20, 2026 | XLF / JPM | -5.19% | -0.83 | 1 |
| May 21–Aug 20, 2026 | JPM / BAC | 18.02% | 3.56 | 3 |

The input period was November 14, 2024–October 30, 2026. The run uses a 252-session training window, a 63-session test window, a 60-session signal lookback, a 0.55 absolute return-correlation filter and 1 bp each for fees and slippage per unit of turnover. The remaining observations did not fill a fourth test window.

The backtest applies the prior-close signal to the next close-to-close return. It does not model a separate next-open fill, borrow costs or a liquidation charge at fold boundaries. The single-pair view is exploratory and should not be read as out-of-sample evidence.

### Open the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app falls back to labeled synthetic prices when Yahoo Finance is unavailable. If the demo finds no eligible pair under the default filter, lower the minimum correlation.
