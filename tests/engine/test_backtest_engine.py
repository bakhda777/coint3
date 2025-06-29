# ВНИМАНИЕ: Строки с sys.path.insert удалены! Они больше не нужны благодаря conftest.py.

import numpy as np
import pandas as pd

from coint2.core import performance

# Импортируем код проекта напрямую
from coint2.engine.backtest_engine import PairBacktester


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
    capital_at_risk: float,
    stop_loss_multiplier: float,
) -> pd.DataFrame:
    """Эталонная реализация логики бэктеста для проверки."""
    df = df.copy()
    y_col, x_col = df.columns[0], df.columns[1]

    df["spread"] = df[y_col] - beta * df[x_col]
    if std == 0:
        df["z_score"] = 0.0
    else:
        df["z_score"] = (df["spread"] - mean) / std
    
    df["position"] = 0.0
    df["trades"] = 0.0
    df["costs"] = 0.0
    df["pnl"] = 0.0

    total_cost_pct = commission_pct + slippage_pct

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
        if z_curr > z_threshold:
            signal = -1
        elif z_curr < -z_threshold:
            signal = 1

        if new_position == 0 and signal != 0:
            entry_z = z_curr
            stop_loss_z = float(np.sign(entry_z) * stop_loss_multiplier)
            stop_loss_price = mean + stop_loss_z * std
            risk_per_unit = abs(spread_curr - stop_loss_price)
            size = capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
            new_position = signal * size
        elif new_position != 0 and signal != 0 and np.sign(new_position) != signal:
            entry_z = z_curr
            stop_loss_z = float(np.sign(entry_z) * stop_loss_multiplier)
            stop_loss_price = mean + stop_loss_z * std
            risk_per_unit = abs(spread_curr - stop_loss_price)
            size = capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
            new_position = signal * size

        trades = abs(new_position - position)
        trade_value = df[y_col].iat[i] + abs(beta) * df[x_col].iat[i]
        costs = trades * trade_value * total_cost_pct

        df.iat[i, df.columns.get_loc("position")] = new_position
        df.iat[i, df.columns.get_loc("trades")] = trades
        df.iat[i, df.columns.get_loc("costs")] = costs
        df.iat[i, df.columns.get_loc("pnl")] = pnl - costs

        position = new_position

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
    annualizing_factor = 365

    beta, mean, std = calc_params(data)

    bt = PairBacktester(
        data,
        beta=beta,
        spread_mean=mean,
        spread_std=std,
        z_threshold=z_threshold,
        commission_pct=commission,
        slippage_pct=slippage,
        annualizing_factor=annualizing_factor,
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
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
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
    )
    expected_for_comparison = expected[["spread", "z_score", "position", "pnl", "cumulative_pnl"]]
    
    pd.testing.assert_frame_equal(result, expected_for_comparison)

    # Проверяем метрики
    metrics = bt.get_performance_metrics()
    
    expected_pnl = expected["pnl"].dropna()
    expected_cum_pnl = expected["cumulative_pnl"].dropna()
    expected_metrics = {
        "sharpe_ratio": performance.sharpe_ratio(
            expected_pnl, annualizing_factor
        ),
        "max_drawdown": performance.max_drawdown(expected_cum_pnl),
        "total_pnl": expected_cum_pnl.iloc[-1] if not expected_cum_pnl.empty else 0.0,
    }

    # Надежное сравнение словарей с float-числами
    assert metrics.keys() == expected_metrics.keys()
    assert np.isclose(metrics["sharpe_ratio"], expected_metrics["sharpe_ratio"])
    assert np.isclose(metrics["max_drawdown"], expected_metrics["max_drawdown"])
    assert np.isclose(metrics["total_pnl"], expected_metrics["total_pnl"])


def test_zero_std_handling() -> None:
    """Проверяет корректность работы при нулевом стандартном отклонении спреда."""
    data = pd.DataFrame({"Y": 2 * np.arange(1, 11), "X": np.arange(1, 11)})

    beta, mean, std = calc_params(data)
    assert std == 0

    bt = PairBacktester(
        data,
        beta=beta,
        spread_mean=mean,
        spread_std=std,
        z_threshold=1.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        annualizing_factor=365,
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
    )
    bt.run()
    result = bt.get_results()

    expected = manual_backtest(
        data,
        beta,
        mean,
        std,
        1.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
    )
    expected_for_comparison = expected[["spread", "z_score", "position", "pnl", "cumulative_pnl"]]

    pd.testing.assert_frame_equal(result, expected_for_comparison)

    metrics = bt.get_performance_metrics()
    assert metrics == {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0}
