import numpy as np
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
        annualizing_factor: int = 365,
        capital_at_risk: float = 1.0,
        stop_loss_multiplier: float = 2.0,
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
        self.annualizing_factor = annualizing_factor
        self.capital_at_risk = capital_at_risk
        self.stop_loss_multiplier = stop_loss_multiplier

    def run(self) -> None:
        """Run backtest and store results in ``self.results``."""
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
            return

        df = self.pair_data.rename(
            columns={
                self.pair_data.columns[0]: "y",
                self.pair_data.columns[1]: "x",
            }
        )

        df["spread"] = df["y"] - self.beta * df["x"]

        if self.std == 0:
            df["z_score"] = 0.0
        else:
            df["z_score"] = (df["spread"] - self.mean) / self.std

        df["position"] = 0.0
        df["trades"] = 0.0
        df["costs"] = 0.0
        df["pnl"] = 0.0

        total_cost_pct = self.commission_pct + self.slippage_pct

        position = 0.0
        entry_z = 0.0
        stop_loss_z = 0.0

        for i in range(1, len(df)):
            spread_prev = df["spread"].iat[i - 1]
            spread_curr = df["spread"].iat[i]
            z_curr = df["z_score"].iat[i]

            pnl = position * (spread_curr - spread_prev)

            new_position = position

            if position > 0 and z_curr <= stop_loss_z:
                new_position = 0.0
            elif position < 0 and z_curr >= stop_loss_z:
                new_position = 0.0

            signal = 0
            if z_curr > self.z_threshold:
                signal = -1
            elif z_curr < -self.z_threshold:
                signal = 1

            if new_position == 0 and signal != 0:
                entry_z = z_curr
                stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
                stop_loss_price = self.mean + stop_loss_z * self.std
                risk_per_unit = abs(spread_curr - stop_loss_price)
                size = self.capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
                new_position = signal * size
            elif new_position != 0 and signal != 0 and np.sign(new_position) != signal:
                entry_z = z_curr
                stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
                stop_loss_price = self.mean + stop_loss_z * self.std
                risk_per_unit = abs(spread_curr - stop_loss_price)
                size = self.capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
                new_position = signal * size

            trades = abs(new_position - position)
            trade_value = df["y"].iat[i] + abs(self.beta) * df["x"].iat[i]
            costs = trades * trade_value * total_cost_pct

            df.iat[i, df.columns.get_loc("position")] = new_position
            df.iat[i, df.columns.get_loc("trades")] = trades
            df.iat[i, df.columns.get_loc("costs")] = costs
            df.iat[i, df.columns.get_loc("pnl")] = pnl - costs

            position = new_position

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
            "sharpe_ratio": performance.sharpe_ratio(pnl, self.annualizing_factor),
            "max_drawdown": performance.max_drawdown(cum_pnl),
            "total_pnl": cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0,
        }
