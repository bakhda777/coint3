from pathlib import Path
from uuid import uuid4

import pandas as pd

from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    PortfolioConfig,
    WalkForwardConfig,
)


def create_random_shards(base: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    for sym in ["AAA", "BBB"]:
        part = base / f"symbol={sym}" / "year=2021" / "month=01"
        part.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": range(len(idx))})
        df.to_parquet(part / f"{uuid4().hex}.parquet")


def test_rglob_finds_all_files(tmp_path: Path) -> None:
    create_random_shards(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(initial_capital=1, risk_per_position_pct=0.1, max_active_positions=1),
        pair_selection=PairSelectionConfig(
            lookback_days=2,
            coint_pvalue_threshold=0.05,
            ssd_top_n=1,
            min_half_life_days=1,
            max_half_life_days=30,
            min_mean_crossings=12,
        ),
        backtest=BacktestConfig(
            timeframe="1d",
            rolling_window=1,
            zscore_threshold=1.0,
            stop_loss_multiplier=3.0,
            fill_limit_pct=0.1,
            commission_pct=0.0,
            slippage_pct=0.0,
            annualizing_factor=365,
        ),
        walk_forward=WalkForwardConfig(
            start_date="2021-01-01",
            end_date="2021-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    df = handler.load_all_data_for_period()

    assert not df.empty
