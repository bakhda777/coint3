import numpy as np
import pandas as pd
from scipy.stats import linregress

from coint2.engine.backtest_engine import PairBacktester
from coint2.core.math_utils import rolling_zscore
from coint2.core.performance import sharpe_ratio, max_drawdown


def test_pair_backtester_basic():
    idx = pd.date_range("2021-01-01", periods=8, freq="D")
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=idx)
    noise = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2, -0.2, 0.1, 0], index=idx)
    y = x * 1.5 + noise
    df = pd.DataFrame({"Y": y, "X": x})

    window = 3
    thresh = 1.0

    bt = PairBacktester(df, window, thresh)
    result = bt.run()

    # manual beta using linregress
    expected_beta = []
    for i in range(len(df)):
        if i + 1 < window:
            expected_beta.append(np.nan)
        else:
            sl, _, _, _, _ = linregress(
                x.iloc[i - window + 1 : i + 1], y.iloc[i - window + 1 : i + 1]
            )
            expected_beta.append(sl)
    expected_beta = pd.Series(expected_beta, index=idx)
    pd.testing.assert_series_equal(result["beta"], expected_beta)

    expected_spread = y - expected_beta * x
    pd.testing.assert_series_equal(result["spread"], expected_spread)

    expected_z = rolling_zscore(expected_spread, window)
    pd.testing.assert_series_equal(result["z_score"], expected_z)

    signals = pd.Series(0.0, index=idx)
    signals[expected_z > thresh] = -1
    signals[expected_z < -thresh] = 1
    pd.testing.assert_series_equal(result["signal"], signals)

    positions = signals.replace(0, np.nan).ffill()
    positions[signals == 0] = 0
    positions = positions.fillna(0)
    pd.testing.assert_series_equal(result["position"], positions)

    expected_pnl = positions.shift().fillna(0) * expected_spread.diff().fillna(0)
    pd.testing.assert_series_equal(result["pnl"], expected_pnl)

    metrics = bt.get_performance_metrics()
    assert metrics["sharpe_ratio"] == sharpe_ratio(expected_pnl)
    assert metrics["max_drawdown"] == max_drawdown(expected_pnl.cumsum())
