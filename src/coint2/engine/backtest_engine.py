import pandas as pd
from ..core import math_utils
from ..core import performance


class PairBacktester:
    """Vectorized backtester for a single pair."""

    def __init__(self, pair_data: pd.DataFrame, window: int, z_threshold: float):
        """Initialize backtester.

        Parameters
        ----------
        pair_data : pd.DataFrame
            DataFrame with columns 'x' and 'y' containing price series.
        window : int
            Rolling window for beta and z-score calculation.
        z_threshold : float
            Z-score absolute threshold for entry signals.
        """
        self.pair_data = pair_data.copy()
        self.window = window
        self.z_threshold = z_threshold
        self.results: pd.DataFrame | None = None

    def run(self) -> None:
        """Run backtest and store results in ``self.results``."""
        df = self.pair_data.copy()

        # rolling beta of y ~ x
        df["beta"] = math_utils.rolling_beta(df["y"], df["x"], self.window)

        # compute spread y - beta * x
        df["spread"] = df["y"] - df["beta"] * df["x"]

        # compute z-score of spread
        df["z_score"] = math_utils.zscore(df["spread"], self.window)

        # generate long/short signals: long when z_score < -threshold, short when z_score > threshold
        df["signal"] = 0
        df.loc[df["z_score"] > self.z_threshold, "signal"] = -1
        df.loc[df["z_score"] < -self.z_threshold, "signal"] = 1

        # forward fill signals to maintain positions until exit (when sign flips or crosses 0)
        df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)

        # shift position by 1 to avoid lookahead bias (enter at next period's open)
        df["position"] = df["position"].shift().fillna(0)

        # compute daily returns of spread trade
        df["spread_return"] = df["spread"].diff()
        df["pnl"] = df["position"] * df["spread_return"]
        df["cumulative_pnl"] = df["pnl"].cumsum()

        self.results = df

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Backtest not yet run")
        return self.results[["spread", "z_score", "position", "pnl", "cumulative_pnl"]]

    def get_performance_metrics(self) -> dict:
        if self.results is None:
            raise ValueError("Backtest not yet run")
        pnl = self.results["pnl"].dropna()
        cum_pnl = self.results["cumulative_pnl"].dropna()
        return {
            "sharpe": performance.sharpe_ratio(pnl),
            "max_drawdown": performance.max_drawdown(cum_pnl),
            "total_pnl": cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0,
        }
