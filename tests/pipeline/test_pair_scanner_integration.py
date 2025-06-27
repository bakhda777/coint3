import pandas as pd
from pathlib import Path

from coint2.core.data_loader import DataHandler
from coint2.pipeline.pair_scanner import find_cointegrated_pairs


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
    handler = DataHandler(tmp_path, '1d', fill_limit_pct=0.1)
    data = handler.load_all_data_for_period(lookback_days=20)
    pairs = find_cointegrated_pairs(data, p_value_threshold=0.05)
    assert ('A', 'B') in pairs or ('B', 'A') in pairs
