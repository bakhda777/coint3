import pandas as pd
# Импортируем модули, которые уже есть в main
from ..core import math_utils
from ..core import performance


class PairBacktester:
    """Vectorized backtester for a single pair."""

    def __init__(self, pair_data: pd.DataFrame, window: int, z_threshold: float):
        """Initialize backtester.

        Parameters
        ----------
        pair_data : pd.DataFrame
            DataFrame with two columns containing price series for the pair.
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
        # Улучшение: делаем код независимым от имен колонок,
        # переименовывая их в 'y' и 'x' для удобства.
        df = self.pair_data.rename(columns={
            self.pair_data.columns[0]: 'y',
            self.pair_data.columns[1]: 'x'
        })

        # rolling beta of y ~ x
        df["beta"] = math_utils.rolling_beta(df["y"], df["x"], self.window)

        # compute spread y - beta * x
        df["spread"] = df["y"] - df["beta"] * df["x"]

        # ИСПРАВЛЕНИЕ: Используем правильное имя функции `rolling_zscore` из `main`
        df["z_score"] = math_utils.rolling_zscore(df["spread"], self.window)

        # generate long/short signals: long when z_score < -threshold, short when z_score > threshold
        df["signal"] = 0
        df.loc[df["z_score"] > self.z_threshold, "signal"] = -1 # Продаем дорогой спред
        df.loc[df["z_score"] < -self.z_threshold, "signal"] = 1  # Покупаем дешевый спред

        # forward fill signals to maintain positions until exit
        # ПРИМЕЧАНИЕ: Это простая логика удержания. Выход происходит при появлении
        # противоположного сигнала или возврате к "нейтральной" зоне.
        df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)

        # shift position by 1 to avoid lookahead bias (входим по цене следующего периода)
        df["position"] = df["position"].shift().fillna(0)

        # compute pnl
        df["spread_return"] = df["spread"].diff()
        df["pnl"] = df["position"] * df["spread_return"]
        df["cumulative_pnl"] = df["pnl"].cumsum()

        self.results = df

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Backtest not yet run")
        return self.results[["spread", "z_score", "position", "pnl", "cumulative_pnl"]]

    def get_performance_metrics(self) -> dict:
        if self.results is None or self.results.empty:
            raise ValueError("Backtest not yet run")
        pnl = self.results["pnl"].dropna()
        cum_pnl = self.results["cumulative_pnl"].dropna()
        return {
            "sharpe_ratio": performance.sharpe_ratio(pnl),
            "max_drawdown": performance.max_drawdown(cum_pnl),
            "total_pnl": cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0,
        }