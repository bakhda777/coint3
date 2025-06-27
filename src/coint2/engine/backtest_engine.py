"""Vectorized pair backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coint2.core.math_utils import rolling_beta, rolling_zscore
from coint2.core.performance import max_drawdown, sharpe_ratio


class PairBacktester:
    """Run a simple mean-reversion backtest on a pair of prices."""

    def __init__(self, prices: pd.DataFrame, window: int, z_threshold: float) -> None:
        self.prices = prices.copy()
        self.window = window
        self.z_threshold = z_threshold
        self.results: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        """Execute backtest and return dataframe with calculations."""
        df = self.prices.copy()
        s1, s2 = df.iloc[:, 0], df.iloc[:, 1]

        df["beta"] = rolling_beta(s1, s2, self.window)
        df["spread"] = s1 - df["beta"] * s2
        df["z_score"] = rolling_zscore(df["spread"], self.window)

        signals = pd.Series(0, index=df.index, dtype=float)
        signals[df["z_score"] > self.z_threshold] = -1
        signals[df["z_score"] < -self.z_threshold] = 1
        df["signal"] = signals

        positions = signals.replace(0, np.nan).ffill()
        positions[signals == 0] = 0
        df["position"] = positions.fillna(0)

        df["pnl"] = df["position"].shift().fillna(0) * df["spread"].diff().fillna(0)

        self.results = df
        return df

    def get_performance_metrics(self) -> dict[str, float]:
        """Return Sharpe ratio and max drawdown of the strategy."""
        if self.results is None:
            raise ValueError("run() must be called before computing metrics")
        pnl = self.results["pnl"]
        cumulative = pnl.cumsum()
        return {
            "sharpe_ratio": sharpe_ratio(pnl),
            "max_drawdown": max_drawdown(cumulative),
        }

