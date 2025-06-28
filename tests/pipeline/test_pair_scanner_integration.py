import pandas as pd
from pathlib import Path


from coint2.core.data_loader import DataHandler
import coint2.pipeline.pair_scanner as pair_scanner
from coint2.utils.config import (
    AppConfig,
    PairSelectionConfig,
    BacktestConfig,
    PortfolioConfig,
    WalkForwardConfig,
)


def create_parquet_files(tmp_path: Path) -> None:
    idx = pd.date_range('2021-01-01', periods=20, freq='D')
    a = pd.Series(range(20), index=idx)
    b = a + 0.1  # cointegrated with A
    c = pd.Series(range(20, 0, -1), index=idx)

    for sym, series in [('A', a), ('B', b), ('C', c)]:
        part_dir = tmp_path / f'symbol={sym}' / 'year=2021' / 'month=01'
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'timestamp': idx, 'close': series})
        df.to_parquet(part_dir / 'data.parquet')


def test_find_cointegrated_pairs(tmp_path: Path) -> None:
    create_parquet_files(tmp_path)
    cfg = AppConfig(
        data_dir=tmp_path,
        results_dir=tmp_path,
        portfolio=PortfolioConfig(
            initial_capital=10000.0,
            risk_per_trade_pct=0.01,
            max_active_positions=5,
        ),
        pair_selection=PairSelectionConfig(
            lookback_days=20, coint_pvalue_threshold=0.05, ssd_top_n=1
        ),
        backtest=BacktestConfig(
            timeframe="1d",
            rolling_window=1,
            zscore_threshold=1.0,
            stop_loss_multiplier=3.0,
            fill_limit_pct=0.1,
            commission_pct=0.001,
            slippage_pct=0.0005,
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
    data = handler.load_all_data_for_period(lookback_days=20)

    beta = data["A"].cov(data["B"]) / data["B"].var()
    spread = data["A"] - beta * data["B"]
    expected = ("A", "B", beta, spread.mean(), spread.std())

    start = data.index.min()
    end = data.index.max()
    pairs = pair_scanner.find_cointegrated_pairs(handler, start, end, cfg)

    assert pairs == [expected]
