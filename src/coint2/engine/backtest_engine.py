import pandas as pd
from ..core import performance


class PairBacktester:
    """Vectorized backtester for a single pair."""

    def __init__(
        self,
        pair_data: pd.DataFrame,
        beta: float,
        spread_mean: float,
        spread_std: float,
        z_threshold: float,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        """Initialize backtester with pre-computed parameters.

        Parameters
        ----------
        pair_data : pd.DataFrame
            DataFrame with two columns containing price series for the pair.
        beta : float
            Regression coefficient between ``y`` and ``x`` estimated on the
            training period.
        spread_mean : float
            Mean of the spread from the training period.
        spread_std : float
            Standard deviation of the spread from the training period.
        z_threshold : float
            Z-score absolute threshold for entry signals.
        """
        self.pair_data = pair_data.copy()
        self.beta = beta
        self.mean = spread_mean
        self.std = spread_std
        self.z_threshold = z_threshold
        self.results: pd.DataFrame | None = None
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self) -> None:
        """Run backtest and store results in ``self.results``."""
        # Улучшение: делаем код независимым от имен колонок,
        # переименовывая их в 'y' и 'x' для удобства.
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            # Создаем пустой DataFrame с нужными колонками, чтобы избежать ошибок дальше
            self.results = pd.DataFrame(columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"])
            return

        df = self.pair_data.rename(
            columns={
                self.pair_data.columns[0]: "y",
                self.pair_data.columns[1]: "x",
            }
        )

        # compute spread using pre-calculated beta
        df["spread"] = df["y"] - self.beta * df["x"]

        # z-score using fixed mean and std from training period
        df["z_score"] = (df["spread"] - self.mean) / self.std

        # generate long/short signals: long when z_score < -threshold, short when z_score > threshold
        df["signal"] = 0
        df.loc[df["z_score"] > self.z_threshold, "signal"] = -1 # Продаем дорогой спред
        df.loc[df["z_score"] < -self.z_threshold, "signal"] = 1  # Покупаем дешевый спред

        # forward fill signals to maintain positions until exit
        # (Выход происходит при появлении противоположного сигнала или возврате к "нейтральной" зоне).
        df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)

        # shift position by 1 to avoid lookahead bias (входим по цене следующего периода)
        df["position"] = df["position"].shift().fillna(0)

        # compute pnl with transaction costs
        df["trades"] = df["position"].diff().abs()
        df["gross_pnl"] = df["position"] * df["spread"].diff()
        total_cost_pct = self.commission_pct + self.slippage_pct
        df["costs"] = df["trades"] * df["y"] * total_cost_pct
        df["pnl"] = df["gross_pnl"] - df["costs"]
        df["cumulative_pnl"] = df["pnl"].cumsum()

        self.results = df

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Backtest not yet run")
        return self.results[["spread", "z_score", "position", "pnl", "cumulative_pnl"]]

    def get_performance_metrics(self) -> dict:
        if self.results is None or self.results.empty:
            raise ValueError("Backtest has not been run or produced no results")

        pnl = self.results["pnl"].dropna()
        cum_pnl = self.results["cumulative_pnl"].dropna()

        # Если после dropna ничего не осталось, возвращаем нулевые метрики
        if pnl.empty:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
            }

        return {
            "sharpe_ratio": performance.sharpe_ratio(pnl),
            "max_drawdown": performance.max_drawdown(cum_pnl),
            "total_pnl": cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0,
        }