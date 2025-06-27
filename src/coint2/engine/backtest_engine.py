"""Simple pair backtester stub used by the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd  # type: ignore

from coint2.core.math_utils import rolling_zscore
from coint2.core.performance import max_drawdown, sharpe_ratio
from coint2.utils.config import BacktestConfig


@dataclass
class PairBacktester:
    """Very lightweight pair trading backtester.

    This implementation is intentionally simplistic and serves as a
    placeholder until the full engine is implemented in another task.
    """

    data: pd.DataFrame
    config: BacktestConfig

    def run(self) -> Dict[str, float]:
        """Execute a naive mean reversion strategy and return metrics."""

        s1, s2 = self.data.columns[:2]
        spread = self.data[s1] - self.data[s2]
        z = rolling_zscore(spread, self.config.rolling_window)

        long = z < -self.config.zscore_threshold
        short = z > self.config.zscore_threshold
        positions = long.astype(int) - short.astype(int)

        returns = spread.diff().shift(-1)
        pnl = (positions * returns).dropna()

        metrics = {
            "cumulative_return": pnl.sum(),
            "sharpe": sharpe_ratio(pnl),
            "max_drawdown": max_drawdown(pnl.cumsum()),
        }
        return metrics

