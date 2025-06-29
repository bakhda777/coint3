import pandas as pd
from pathlib import Path

from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PortfolioConfig,
    PairSelectionConfig,
    WalkForwardConfig,
)


def create_dataset_with_duplicates(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    for sym, shift in [("AAA", 0), ("BBB", 100)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "timestamp": list(idx) + [idx[1]],  # duplicate second day
            "close": [shift + i for i in range(3)] + [shift + 99],
        })
        df.to_parquet(part_dir / "data.parquet")


def _create_handler(tmp_path: Path) -> DataHandler:
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_trade_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=1,
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
            end_date="2021-01-03",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    return DataHandler(cfg)


def test_pivot(tmp_path: Path) -> None:
    create_dataset_with_duplicates(tmp_path)
    handler = _create_handler(tmp_path)

    result = handler.load_all_data_for_period(lookback_days=3)

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    end_date = pdf["timestamp"].max()
    start_date = end_date - pd.Timedelta(days=3)
    filtered = pdf[pdf["timestamp"] >= start_date]
    expected = filtered.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    )
    expected = expected.sort_index()

    pd.testing.assert_frame_equal(result, expected)
