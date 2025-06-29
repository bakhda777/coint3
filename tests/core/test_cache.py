import pandas as pd
from pathlib import Path
import threading
import time
import dask.dataframe as dd
from coint2.core.data_loader import DataHandler
from coint2.utils.config import (
    AppConfig,
    PortfolioConfig,
    PairSelectionConfig,
    BacktestConfig,
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


def test_load_all_data_cache(monkeypatch, tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    handler = DataHandler(cfg)

    calls = 0
    original = dd.read_parquet

    def counting_read_parquet(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dd, "read_parquet", counting_read_parquet)

    handler.load_all_data_for_period(lookback_days=2)
    assert calls == 1

    handler.load_all_data_for_period(lookback_days=2)
    assert calls == 1


def test_threaded_cache_reload(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    handler = DataHandler(cfg)

    # preload and then clear to force reload
    handler.load_all_data_for_period(lookback_days=2)
    handler.clear_cache()

    results: list[pd.DataFrame] = []

    def worker() -> None:
        results.append(handler.load_all_data_for_period(lookback_days=2))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    pd.testing.assert_frame_equal(results[0], results[1])
    assert handler.freq == pd.infer_freq(results[0].index)


def test_cache_autorefresh(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    handler = DataHandler(cfg)

    initial = handler.load_all_data_for_period(lookback_days=10)
    assert len(initial) == 5

    part_dir = tmp_path / "symbol=AAA" / "year=2021" / "month=01"
    new_idx = pd.date_range("2021-01-01", periods=6, freq="D")
    df = pd.DataFrame({"timestamp": new_idx, "close": range(6)})
    time.sleep(1)
    df.to_parquet(part_dir / "data.parquet")

    updated = handler.load_all_data_for_period(lookback_days=10)
    assert len(updated) == 6
    assert pd.Timestamp("2021-01-06") in updated.index
