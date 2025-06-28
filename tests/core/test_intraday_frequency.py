import pandas as pd
from pathlib import Path

from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    PortfolioConfig,
    PairSelectionConfig,
    BacktestConfig,
    WalkForwardConfig,
)


def create_intraday_dataset(tmp_path: Path) -> None:
    day1 = pd.date_range("2021-01-01 09:30", "2021-01-01 16:00", freq="15T")
    day2 = pd.date_range("2021-01-02 09:30", "2021-01-02 16:00", freq="15T")
    idx = day1.append(day2)
    for sym, shift in [("AAA", 0), ("BBB", 1)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        series = pd.Series(range(len(idx)), index=idx) + shift
        df = pd.DataFrame({"timestamp": idx, "close": series.values})
        df.to_parquet(part_dir / "data.parquet")


def make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
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
            timeframe="15T",
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


def test_intraday_frequency(tmp_path: Path) -> None:
    create_intraday_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    handler = DataHandler(cfg)

    start = pd.Timestamp("2021-01-01 09:30")
    end = pd.Timestamp("2021-01-02 16:00")
    result = handler.load_pair_data("AAA", "BBB", start, end)

    day1 = pd.date_range("2021-01-01 09:30", "2021-01-01 16:00", freq="15T")
    day2 = pd.date_range("2021-01-02 09:30", "2021-01-02 16:00", freq="15T")
    idx = day1.append(day2)
    assert list(result.index) == list(idx)
    assert handler.freq is None
