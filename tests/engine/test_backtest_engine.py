import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import pandas as pd
import numpy as np

from coint2.engine.backtest_engine import PairBacktester
from coint2.core import math_utils, performance


def manual_backtest(df: pd.DataFrame, window: int, z_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["beta"] = math_utils.rolling_beta(df["y"], df["x"], window)
    df["spread"] = df["y"] - df["beta"] * df["x"]
    df["z_score"] = math_utils.zscore(df["spread"], window)
    df["signal"] = 0
    df.loc[df["z_score"] > z_threshold, "signal"] = -1
    df.loc[df["z_score"] < -z_threshold, "signal"] = 1
    df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)
    df["position"] = df["position"].shift().fillna(0)
    df["spread_return"] = df["spread"].diff()
    df["pnl"] = df["position"] * df["spread_return"]
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df


def test_backtester_outputs():
    np.random.seed(0)
    x = np.linspace(1, 20, 20)
    noise = np.random.normal(0, 0.5, size=len(x))
    y = x + noise
    data = pd.DataFrame({"x": x, "y": y})

    window = 5
    z_threshold = 1.0

    bt = PairBacktester(data, window=window, z_threshold=z_threshold)
    bt.run()
    result = bt.get_results()

    expected = manual_backtest(data, window, z_threshold)

    pd.testing.assert_series_equal(result["spread"], expected["spread"])
    pd.testing.assert_series_equal(result["z_score"], expected["z_score"])
    pd.testing.assert_series_equal(result["position"], expected["position"])
    pd.testing.assert_series_equal(result["pnl"], expected["pnl"])
    pd.testing.assert_series_equal(result["cumulative_pnl"], expected["cumulative_pnl"])

    metrics = bt.get_performance_metrics()
    expected_metrics = {
        "sharpe": performance.sharpe_ratio(expected["pnl"].dropna()),
        "max_drawdown": performance.max_drawdown(expected["cumulative_pnl"].dropna()),
        "total_pnl": expected["cumulative_pnl"].dropna().iloc[-1],
    }
    assert metrics == expected_metrics
