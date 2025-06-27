# ВНИМАНИЕ: Строки с sys.path.insert удалены! Они больше не нужны благодаря conftest.py.

import pandas as pd
import numpy as np

# Импортируем код проекта напрямую
from coint2.engine.backtest_engine import PairBacktester
from coint2.core import math_utils, performance

def manual_backtest(df: pd.DataFrame, window: int, z_threshold: float) -> pd.DataFrame:
    """Эталонная реализация логики бэктеста для проверки."""
    df = df.copy()
    # Делаем код независимым от имен колонок, как в основном классе
    y_col, x_col = df.columns[0], df.columns[1]

    df["beta"] = math_utils.rolling_beta(df[y_col], df[x_col], window)
    df["spread"] = df[y_col] - df["beta"] * df[x_col]
    
    # ИСПРАВЛЕНИЕ: Используем правильное имя функции `rolling_zscore`
    df["z_score"] = math_utils.rolling_zscore(df["spread"], window)
    
    df["signal"] = 0
    df.loc[df["z_score"] > z_threshold, "signal"] = -1
    df.loc[df["z_score"] < -z_threshold, "signal"] = 1
    df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)
    df["position"] = df["position"].shift(1).fillna(0)
    df["spread_return"] = df["spread"].diff()
    df["pnl"] = df["position"] * df["spread_return"]
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df

def test_backtester_outputs():
    """Проверяет, что каждый столбец и метрика бэктестера совпадают с эталоном."""
    np.random.seed(0)
    # Используем произвольные имена колонок для проверки надежности
    data = pd.DataFrame({
        "ASSET_Y": np.linspace(1, 20, 20) + np.random.normal(0, 0.5, size=20),
        "ASSET_X": np.linspace(1, 20, 20)
    })

    window = 5
    z_threshold = 1.0

    bt = PairBacktester(data, window=window, z_threshold=z_threshold)
    bt.run()
    result = bt.get_results()

    # Сравниваем с эталоном
    expected = manual_backtest(data, window, z_threshold)
    expected_for_comparison = expected[["spread", "z_score", "position", "pnl", "cumulative_pnl"]]
    
    pd.testing.assert_frame_equal(result, expected_for_comparison)

    # Проверяем метрики
    metrics = bt.get_performance_metrics()
    
    expected_pnl = expected["pnl"].dropna()
    expected_cum_pnl = expected["cumulative_pnl"].dropna()
    expected_metrics = {
        # ИСПРАВЛЕНИЕ: Используем правильные ключи из `get_performance_metrics`
        "sharpe_ratio": performance.sharpe_ratio(expected_pnl),
        "max_drawdown": performance.max_drawdown(expected_cum_pnl),
        "total_pnl": expected_cum_pnl.iloc[-1] if not expected_cum_pnl.empty else 0.0,
    }

    # Надежное сравнение словарей с float-числами
    assert metrics.keys() == expected_metrics.keys()
    assert np.isclose(metrics["sharpe_ratio"], expected_metrics["sharpe_ratio"])
    assert np.isclose(metrics["max_drawdown"], expected_metrics["max_drawdown"])
    assert np.isclose(metrics["total_pnl"], expected_metrics["total_pnl"])