from pathlib import Path
import types

import numpy as np
import pandas as pd

from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    BacktestConfig,
    PairSelectionConfig,
    PortfolioConfig,
    WalkForwardConfig,
)


def create_dataset(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    for sym, shift in [("AAA", 0), ("BBB", 1)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        series = pd.Series(range(5), index=idx) + shift
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


def test_load_all_data_for_period(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
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

    result = handler.load_all_data_for_period()

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.sort_values("timestamp")
    end_date = pdf["timestamp"].max()
    start_date = end_date - pd.Timedelta(days=2)
    filtered = pdf[pdf["timestamp"] >= start_date]
    expected = filtered.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.sort_index()

    pd.testing.assert_frame_equal(result, expected)


def test_load_pair_data(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=10,
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

    result = handler.load_pair_data(
        "AAA",
        "BBB",
        pd.Timestamp("2021-01-02"),
        pd.Timestamp("2021-01-04"),
    )

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf = pdf[pdf["symbol"].isin(["AAA", "BBB"])]
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    expected = pdf.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.asfreq(pd.infer_freq(expected.index) or "D")
    limit = int(len(expected) * 0.1)
    expected = expected.ffill(limit=limit).bfill(limit=limit)
    expected = expected[["AAA", "BBB"]].dropna()
    expected = expected.loc[pd.Timestamp("2021-01-02"): pd.Timestamp("2021-01-04")]

    pd.testing.assert_frame_equal(result, expected)


def test_load_and_normalize_data(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=10,
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

    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2021-01-05")
    result = handler.load_and_normalize_data(start, end)

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    mask = (pdf["timestamp"] >= start) & (pdf["timestamp"] <= end)
    pdf = pdf.loc[mask]
    expected = pdf.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.sort_index()

    for col in expected.columns:
        series = expected[col]
        max_val = series.max()
        min_val = series.min()
        if pd.isna(max_val) or pd.isna(min_val) or max_val == min_val:
            expected[col] = 0.0
        else:
            expected[col] = (series - min_val) / (max_val - min_val)

    pd.testing.assert_frame_equal(result, expected)
    assert (result >= 0).all().all()
    assert (result <= 1).all().all()


def test_clear_cache(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
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
            end_date="2021-01-02",
            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    initial = handler.load_all_data_for_period()
    assert "CCC" not in initial.columns

    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    part_dir = tmp_path / "symbol=CCC" / "year=2021" / "month=01"
    part_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"timestamp": idx, "close": range(5)})
    df.to_parquet(part_dir / "data.parquet")

    handler.clear_cache()
    result = handler.load_all_data_for_period()

    pdf = pd.read_parquet(tmp_path, engine="pyarrow")
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.sort_values("timestamp")
    end_date = pdf["timestamp"].max()
    start_date = end_date - pd.Timedelta(days=10)
    filtered = pdf[pdf["timestamp"] >= start_date]
    expected = filtered.pivot_table(index="timestamp", columns="symbol", values="close")
    expected = expected.sort_index()

    pd.testing.assert_frame_equal(result, expected)
    assert "CCC" in result.columns


def create_large_dataset_with_gaps(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    a = pd.Series(range(100), index=idx, dtype=float)
    b = a + 1
    a[50:60] = np.nan
    b[60:70] = np.nan
    for sym, series in [("AAA", a), ("BBB", b)]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2021" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": series})
        df.to_parquet(part_dir / "data.parquet")


def test_fill_limit_pct_application(tmp_path: Path) -> None:
    create_large_dataset_with_gaps(tmp_path)

    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
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

            end_date="2021-04-10",

            training_period_days=1,
            testing_period_days=1,
        ),
    )
    handler = DataHandler(cfg)

    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2021-04-10")
    result = handler.load_pair_data("AAA", "BBB", start, end)

    expected_a = pd.Series(np.arange(100, dtype=float), index=pd.date_range("2021-01-01", periods=100, freq="D"))
    expected_b = expected_a + 1
    expected_a[50:60] = np.nan
    expected_b[60:70] = np.nan
    expected = pd.DataFrame({"AAA": expected_a, "BBB": expected_b})
    limit = int(len(expected) * 0.1)
    expected = expected.ffill(limit=limit).bfill(limit=limit)
    expected = expected[["AAA", "BBB"]].dropna()

    pd.testing.assert_frame_equal(result, expected)




def create_future_dataset(tmp_path: Path) -> None:
    idx = pd.date_range("2025-01-11", periods=5, freq="D")
    for sym in ["AAA", "BBB"]:
        part_dir = tmp_path / f"symbol={sym}" / "year=2025" / "month=01"
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": idx, "close": range(5)})
        df.to_parquet(part_dir / "data.parquet")


def test__load_full_dataset(tmp_path: Path) -> None:
    create_future_dataset(tmp_path)
    cfg = types.SimpleNamespace(
        data_dir=tmp_path,
        backtest=types.SimpleNamespace(fill_limit_pct=0.1),
        pair_selection=types.SimpleNamespace(lookback_days=10),
        max_shards=None,
    )
    loader = DataHandler(cfg)
    end_date = pd.Timestamp("2025-01-15")

    ddf = loader._load_full_dataset()
    df = ddf.compute()

    assert not df.empty
    assert df["timestamp"].min() >= end_date - pd.Timedelta(days=10)
