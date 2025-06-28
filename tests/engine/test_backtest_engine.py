# ВНИМАНИЕ: Строки с sys.path.insert удалены! Они больше не нужны благодаря conftest.py.

import pandas as pd
import numpy as np

# Импортируем код проекта напрямую
from coint2.engine.backtest_engine import PairBacktester
from coint2.core import performance

def calc_params(df: pd.DataFrame) -> tuple[float, float, float]:
    """Calculate beta, mean and std of spread for the given DataFrame."""
    y_col, x_col = df.columns[0], df.columns[1]
    beta = df[y_col].cov(df[x_col]) / df[x_col].var()
    spread = df[y_col] - beta * df[x_col]
    return beta, spread.mean(), spread.std()


def manual_backtest(
    df: pd.DataFrame,
    beta: float,
    mean: float,
    std: float,
    z_threshold: float,
    commission_pct: float,
    slippage_pct: float,
) -> pd.DataFrame:
    """Эталонная реализация логики бэктеста для проверки."""
    df = df.copy()
    y_col, x_col = df.columns[0], df.columns[1]

    df["spread"] = df[y_col] - beta * df[x_col]
    df["z_score"] = (df["spread"] - mean) / std
    
    df["signal"] = 0
    df.loc[df["z_score"] > z_threshold, "signal"] = -1
    df.loc[df["z_score"] < -z_threshold, "signal"] = 1
    df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)
    df["position"] = df["position"].shift(1).fillna(0)
    df["trades"] = df["position"].diff().abs()
    df["gross_pnl"] = df["position"] * df["spread"].diff()
    total_cost_pct = commission_pct + slippage_pct
    df["costs"] = df["trades"] * df[y_col] * total_cost_pct
    df["pnl"] = df["gross_pnl"] - df["costs"]
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

    z_threshold = 1.0
    commission = 0.001
    slippage = 0.0005

    beta, mean, std = calc_params(data)

    bt = PairBacktester(
        data,
        beta=beta,
        spread_mean=mean,
        spread_std=std,
        z_threshold=z_threshold,
        commission_pct=commission,
        slippage_pct=slippage,
    )
    bt.run()
    result = bt.get_results()

    # Сравниваем с эталоном
    expected = manual_backtest(
        data,
        beta,
        mean,
        std,
        z_threshold,
        commission_pct=commission,
        slippage_pct=slippage,
    )
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