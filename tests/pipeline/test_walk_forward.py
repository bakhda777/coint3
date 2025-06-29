from pathlib import Path

import pandas as pd

from coint2.core import performance
from coint2.core.data_loader import DataHandler
from coint2.engine.backtest_engine import PairBacktester
from coint2.pipeline import walk_forward_orchestrator as wf
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    PortfolioConfig,
    WalkForwardConfig,
)


def create_dataset(base_dir: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=12, freq="D")
    a = pd.Series(range(len(idx)), index=idx)
    b = a + 0.1

    for sym, series in [("A", a), ("B", b)]:
        part_dir = base_dir / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


def manual_walk_forward(handler: DataHandler, cfg: AppConfig) -> dict:
    full_start = pd.Timestamp(cfg.walk_forward.start_date) - pd.Timedelta(days=cfg.walk_forward.training_period_days)
    master = handler.preload_all_data(full_start, pd.Timestamp(cfg.walk_forward.end_date))

    overall = pd.Series(dtype=float)
    equity = cfg.portfolio.initial_capital
    current = pd.Timestamp(cfg.walk_forward.start_date)
    end = pd.Timestamp(cfg.walk_forward.end_date)
    while current < end:
        train_end = current + pd.Timedelta(days=cfg.walk_forward.training_period_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)
        if test_end > end:
            break
        train = master.loc[current:train_end, ["A", "B"]].dropna()
        beta = train["A"].cov(train["B"]) / train["B"].var()
        spread = train["A"] - beta * train["B"]
        mean = spread.mean()
        std = spread.std()

        pairs = [("A", "B", beta, mean, std)]
        active_pairs = pairs[: cfg.portfolio.max_active_positions]

        step_pnl = pd.Series(dtype=float)
        total_step_pnl = 0.0

        if active_pairs:
            capital_per_pair = equity * cfg.portfolio.risk_per_position_pct
        else:
            capital_per_pair = 0.0

        for _s1, _s2, _beta, _mean, _std in active_pairs:
            data = master.loc[test_start:test_end, ["A", "B"]].dropna()
            bt = PairBacktester(
                data,
                beta=_beta,
                spread_mean=_mean,
                spread_std=_std,
                z_threshold=cfg.backtest.zscore_threshold,
                commission_pct=cfg.backtest.commission_pct,
                slippage_pct=cfg.backtest.slippage_pct,
                annualizing_factor=cfg.backtest.annualizing_factor,
            )
            bt.run()
            pnl_series = bt.get_results()["pnl"] * capital_per_pair
            step_pnl = step_pnl.add(pnl_series, fill_value=0)
            total_step_pnl += pnl_series.sum()

        overall = pd.concat([overall, step_pnl])
        equity += total_step_pnl
        current = test_end

    overall = overall.dropna()
    if overall.empty:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0}
    cum = overall.cumsum()
    return {
        "sharpe_ratio": performance.sharpe_ratio(
            overall, cfg.backtest.annualizing_factor
        ),
        "max_drawdown": performance.max_drawdown(cum),
        "total_pnl": cum.iloc[-1],
    }


def test_walk_forward(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path / "results",
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=5,
            coint_pvalue_threshold=0.05,
            ssd_top_n=1,
            min_half_life_days=1,
            max_half_life_days=30,
            min_mean_crossings=12,
        ),
        backtest=BacktestConfig(
            timeframe="1d",
            rolling_window=3,
            zscore_threshold=1.0,
            stop_loss_multiplier=3.0,
            fill_limit_pct=0.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            annualizing_factor=365,
        ),
        walk_forward=WalkForwardConfig(
            start_date="2021-01-01",
            end_date="2021-01-11",
            training_period_days=2,
            testing_period_days=2,
        ),
    )

    metrics = wf.run_walk_forward(cfg)

    expected_metrics = manual_walk_forward(DataHandler(cfg), cfg)

    assert metrics == expected_metrics
