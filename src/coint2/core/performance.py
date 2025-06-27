"""Performance metrics for trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def sharpe_ratio(pnl: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio of a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Profit and loss series.
    risk_free_rate : float, optional
        Daily risk free rate, by default ``0.0``.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    excess_returns = pnl - risk_free_rate
    if excess_returns.std(ddof=0) == 0:
        return np.nan
    daily_sharpe = excess_returns.mean() / excess_returns.std(ddof=0)
    return daily_sharpe * np.sqrt(TRADING_DAYS)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Calculate maximum drawdown from a cumulative PnL series.

    Parameters
    ----------
    cumulative_pnl : pd.Series
        Cumulative profit and loss series.

    Returns
    -------
    float
        Maximum drawdown as a non-positive number.
    """
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    return drawdown.min()
